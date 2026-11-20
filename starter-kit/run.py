"""Trusted development runner; participant code executes in a separate service."""
import argparse
import asyncio
import os
from datetime import datetime
from importlib.metadata import version
from uuid import uuid4
from zoneinfo import ZoneInfo
import yaml
from environments import DATASETS, get_environment
import hashlib
import json
import time
from pathlib import Path


def jsonable(value):
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def summarize(rows, expected_tasks):
    complete = len(rows) == expected_tasks and all(r.get('status') in ('success', 'failed') for r in rows)
    return {'expected_tasks': expected_tasks, 'attempted_tasks': len(rows), 'complete': complete,
            'primary_metric': 'score',
            'score': sum(r['score'] for r in rows) / expected_tasks if complete and expected_tasks else None,
            'successful_tasks': sum(r.get('success') is True for r in rows),
            'errors': sum(r.get('status') not in ('success', 'failed') for r in rows)}


class Recorder:
    def __init__(self, directory, secret="", expected_tasks=0):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.secret = secret
        self.expected_tasks = expected_tasks
        self.episodes = []

    def encode(self, obj):
        def redact(value):
            if isinstance(value, str):
                return value.replace(self.secret, "[REDACTED]") if self.secret else value
            if isinstance(value, dict):
                return {redact(str(k)): redact(v) for k, v in value.items()}
            if isinstance(value, list):
                return [redact(v) for v in value]
            if value is None or isinstance(value, (int, float, bool)):
                return value
            return redact(str(value))
        return json.dumps(redact(jsonable(obj)), ensure_ascii=False)

    def save(self, name, obj):
        (self.directory / name).write_text(self.encode(obj) + "\n", encoding="utf-8")

    def append(self, name, obj):
        with (self.directory / name).open("a", encoding="utf-8") as stream:
            stream.write(self.encode(obj) + "\n")

    def trace(self, task_id, event, **payload):
        self.append("traces.jsonl", {"time": time.time(), "task_id": task_id,
                                    "event": event, **payload})

    def summary(self):
        return summarize(self.episodes, self.expected_tasks)

    def episode(self, result):
        self.episodes.append(result)
        self.append("episodes.jsonl", result)
        self.save("summary.json", self.summary())



def config_for_recording(config):
    return {**config, "model": {k:v for k,v in config["model"].items() if k != "api_key"}}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

ROOT = Path(__file__).resolve().parent

def model_credentials(config):
    model = config['model']
    key_env = model.get('api_key_env', 'MODEL_API_KEY')
    key = os.environ.get(key_env, '')
    # The launcher mounts a secret using a provider-neutral container variable.
    # Native runs may use the configured variable or its _FILE companion.
    key_file = os.environ.get('MODEL_API_KEY_FILE') or os.environ.get(key_env + '_FILE')
    if key_file:
        key = Path(key_file).read_text().strip()
    key = key.strip()
    if not key:
        raise ValueError('Provide the model API key to the official container only')
    return {'base': model['base_url'], 'provider': model['provider'],
            'model': model['name'], 'key': key}


def remote_algorithm(config, method):
    from environments.algorithm_client import RemoteAlgorithm
    endpoint = os.environ.get('ALGORITHM_URL', config['algorithm'].get('endpoint', 'http://algorithm:8765'))
    return RemoteAlgorithm(endpoint, str(method), domain=config['data']['dataset'],
                           seed=config['data']['seed'], options=config['algorithm'].get('options'))


def _private_registration(environment, root, rows):
    from contextlib import nullcontext
    if not rows:
        return nullcontext()
    import importlib.util
    spec = importlib.util.spec_from_file_location('organizer_registration', ROOT / 'test/evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.registered_tasks(environment, root, rows, {})


async def _execute_worker(spec):
    # Configure logging before framework imports create file handlers.
    from openjiuwen.core.common.logging.log_config import configure_log_config
    configure_log_config({'level': 'WARNING', 'output': ['console'],
                         'interface_output': ['console'], 'performance_output': ['console']})
    from environments.agent import execute_episode
    from environments.algorithm_client import AlgorithmError, AlgorithmUnavailable
    from openjiuwen.core.runner import Runner
    config = spec['config']
    credentials = model_credentials(config)
    environment = get_environment(config['data']['dataset'])
    algorithm = remote_algorithm(config, spec['method'])
    recorder = Recorder.__new__(Recorder)
    recorder.directory, recorder.secret = Path(spec['directory']), credentials['key']
    stats = dict.fromkeys(('llm_requests', 'input_tokens', 'output_tokens', 'reasoning_tokens',
                          'usage_missing_calls', 'guidance_seconds', 'guidance_calls',
                          'retrieval_initialization_seconds'), 0)
    api_calls, env, result, error = [], None, None, None
    status = 'error'
    started = time.monotonic()
    try:
        await Runner.start()
        with _private_registration(environment, Path(config['data']['root']), spec.get('private_rows')):
            env = environment.Adapter(Path(config['data']['root']), spec['task_id'], config['limits']['max_actions'])
            runtime = {**config_for_recording(config)['model'], **config['limits']}
            result = await execute_episode(env, algorithm, [], runtime, credentials, recorder, stats, api_calls)
            if isinstance(result, dict) and result.get('result_type') == 'error':
                status, error = 'error', str(result.get('output'))
            else:
                status = 'success' if env.success else 'failed'
    except AlgorithmUnavailable as exc:
        status, error = 'error', str(exc)
    except AlgorithmError as exc:
        status, error = 'failed', str(exc)
    except Exception as exc:
        status, error = 'error', type(exc).__name__ + ': ' + str(exc)
    failures = [c['failure'] for c in api_calls if c.get('failure')]
    if stats.get('algorithm_error'):
        status, error = 'failed', 'Algorithm service failed during the episode'
    if stats.get('algorithm_unavailable'):
        status, error = 'error', 'Algorithm connection unavailable; attribution requires review'
    if failures:
        status = 'failed' if all(f['kind'] == 'output' for f in failures) else 'error'
        error = 'Model request failed: ' + json.dumps(failures)
    reason = 'environment_finished' if env and (env.success or getattr(env, 'submitted', False)) else (
        'action_limit' if env and env.attempts >= env.max_actions else (
        'model_call_limit' if stats['llm_requests'] >= config['limits']['max_model_calls'] else
        'algorithm_error' if error and status == 'failed' and not failures else
        'error' if status == 'error' else 'model_stopped'))
    row = {'task_id': spec['task_id'], 'domain': environment.DOMAIN, 'status': status,
           'success': bool(env and env.success and status in ('success', 'failed')),
           'score': float(env.score) if env and status in ('success', 'failed') else 0.0,
           'termination_reason': reason, 'error': error,
           'retryable_model_failure': bool(failures) and all(f['kind'] == 'service' for f in failures),
           'model_failures': failures, 'elapsed_seconds': time.monotonic() - started,
           'action_attempts': env.attempts if env else 0, 'environment_steps': env.steps if env else 0,
           'goal': env.goal if env else '', 'actions': [r['action'] for r in env.history if r['valid']] if env else [],
           'result': result, **stats}
    recorder.save(spec['result_name'], row)
    if env:
        env.close()
    await Runner.stop()


async def supervise_process(command, *, timeout, log_path):
    """External to the episode: kill its entire process group at the deadline."""
    import signal
    with Path(log_path).open('wb') as log:
        process = await asyncio.create_subprocess_exec(*command, stdout=log, stderr=log,
                                                       start_new_session=True)
        expired = False
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            expired = True
        finally:
            # Also reap descendants left behind after a normal episode exit.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
    return process.returncode, expired


async def restart_algorithm(algorithm):
    """Ask the trusted host to replace the container; no Docker socket in either service."""
    control_root = os.environ.get('ALGORITHM_CONTROL_DIR')
    if not control_root:
        await asyncio.to_thread(algorithm.cancel)  # Native trusted-service tests only.
        return
    control = Path(control_root)
    control.mkdir(parents=True, exist_ok=True)
    request_id = uuid4().hex
    temporary = control / 'request.tmp'
    temporary.write_text(json.dumps({'id': request_id, 'operation': 'restart_algorithm'}))
    temporary.replace(control / 'request.json')
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            reply = json.loads((control / 'reply.json').read_text())
            if reply.get('id') == request_id:
                if reply.get('ok') is not True:
                    raise RuntimeError('Host failed to restart the algorithm container')
                return
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        await asyncio.sleep(.2)
    raise RuntimeError('No host acknowledgement for algorithm-container restart')


async def experiment(config, tasks, corpus, data_report, algorithm, project_root, private_rows=None):
    import sys
    model_credentials(config)  # Check before starting unbounded index preparation.
    environment = get_environment(config['data']['dataset'])
    stamp = datetime.now(ZoneInfo('UTC')).strftime('%Y%m%d_%H%M%S')
    directory = (project_root / config['output']['directory']).resolve() / (stamp + '_' + environment.DOMAIN + '_' + uuid4().hex[:8])
    recorder = Recorder(directory, secret=model_credentials(config)['key'], expected_tasks=len(tasks))
    method = config['algorithm']['method']
    recorder.save('config.json', {'config': config_for_recording(config), 'selected_tasks': tasks,
                  'tasks_sha256': digest(tasks), 'corpus_sha256': digest(corpus), 'data_report': data_report,
                  'games_sha256': environment.task_fingerprints(Path(config['data']['root']), tasks),
                  'runtime_versions': {name: version(name) for name in environment.RUNTIME_PACKAGES},
                  'system_prompt': environment.SYSTEM_PROMPT,
                  'algorithm_image': os.environ.get('ALGORITHM_IMAGE_ID', 'local-service-unverified')})
    print(f'Results: {directory}', flush=True)
    started = time.monotonic()
    await asyncio.to_thread(algorithm.prepare, corpus)
    recorder.save('preparation.json', {'seconds': time.monotonic() - started})
    for task_id in tasks:
        for attempt in range(3):
            spec_name = '.episode-input.json'
            result_name = '.episode-result.json'
            result_path = directory / result_name
            result_path.unlink(missing_ok=True)
            recorder.save(spec_name, {'config': config, 'method': method, 'task_id': task_id,
                                      'directory': str(directory), 'result_name': result_name,
                                      'private_rows': [r for r in private_rows or [] if r['task_id'] == task_id]})
            recorder.trace(task_id, 'episode_start', attempt=attempt + 1)
            wall_start = time.monotonic()
            try:
                code, expired = await supervise_process(
                    [sys.executable, str(ROOT / 'run.py'), '--episode-worker', str(directory / spec_name)],
                    timeout=config['limits']['episode_timeout_seconds'],
                    log_path=directory / f'worker_{len(recorder.episodes):03d}_{attempt}.log')
            finally:
                (directory / spec_name).unlink(missing_ok=True)
            if expired:
                row = {'task_id': task_id, 'domain': environment.DOMAIN, 'status': 'failed', 'score': 0.0,
                       'success': False, 'termination_reason': 'time_limit', 'model_failures': [],
                       'error': 'Episode process exceeded the wall-clock deadline',
                       'elapsed_seconds': time.monotonic() - wall_start}
            elif code != 0 or not result_path.exists():
                row = {'task_id': task_id, 'domain': environment.DOMAIN, 'status': 'error', 'score': 0.0,
                       'success': False, 'termination_reason': 'worker_error', 'model_failures': [],
                       'error': 'Official episode worker exited without a complete result'}
            else:
                row = json.loads(result_path.read_text())
            result_path.unlink(missing_ok=True)
            row['attempt'] = attempt + 1
            recorder.append('attempts.jsonl', row)
            if row.get('retryable_model_failure') and attempt < 2:
                recorder.trace(task_id, 'service_retry', attempt=attempt + 1)
                await asyncio.sleep(2)
                continue
            recorder.episode(row)
            print(f"Task: {task_id}; status={row['status']}; score={row['score']}", flush=True)
            if expired or row.get('termination_reason') == 'algorithm_error' or row.get('algorithm_unavailable'):
                # Do not trust an untrusted service's claim that cancellation worked.
                await restart_algorithm(algorithm)
                if task_id != tasks[-1]:
                    await asyncio.to_thread(algorithm.prepare, corpus)
            break
        if len(recorder.episodes) >= 3 and all(r['status'] == 'error' for r in recorder.episodes[-3:]):
            break
    result = recorder.summary()
    result['rankable'] = not config['data'].get('smoke', False) and result['complete']
    recorder.save('summary.json', result)
    return {'directory': str(directory), 'summary': result}


def validate_config(config, *, require_model=True):
    if not isinstance(config, dict) or set(config) != {'model', 'algorithm', 'data', 'limits', 'output'}:
        raise ValueError('Configuration requires model, algorithm, data, limits and output')
    model = config['model']
    if not isinstance(model, dict):
        raise ValueError('model must be a mapping')
    required = {'provider', 'base_url', 'name', 'temperature', 'max_tokens',
                'request_timeout_seconds'}
    if not required.issubset(model):
        raise ValueError('Incomplete model configuration')
    if model.get('reasoning', 'auto') not in ('disabled', 'enabled', 'auto'):
        raise ValueError('Invalid reasoning setting')
    if type(model['temperature']) not in (int, float) or not 0 <= model['temperature'] <= 2:
        raise ValueError('Temperature must be between 0 and 2')
    providers = model.get('provider_order', [])
    if not isinstance(providers, list) or not all(isinstance(p,str) and p for p in providers):
        raise ValueError('provider_order must be a list of provider names')
    if type(model.get('allow_fallbacks', False)) is not bool:
        raise ValueError('allow_fallbacks must be a boolean')
    if any(k in model for k in ('reasoning', 'provider_order', 'allow_fallbacks')) and str(model.get('provider', '')).lower() != 'openrouter':
        raise ValueError('Legacy routing/reasoning fields are OpenRouter-only; use extra_body for your endpoint')
    from urllib.parse import urlsplit
    import re
    if model.get('api_key'):
        raise ValueError('Keep API keys outside config; use api_key_env or its _FILE companion')
    key_env = model.get('api_key_env', 'MODEL_API_KEY')
    if not isinstance(key_env, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key_env) or key_env.endswith('_FILE'):
        raise ValueError('api_key_env must name an environment variable, without the _FILE suffix')
    extra = model.get('extra_body', {})
    if not isinstance(extra, dict) or not all(isinstance(k, str) for k in extra):
        raise ValueError('extra_body must be a mapping with string keys')
    if set(extra) & {'model', 'messages', 'tools', 'tool_choice', 'parallel_tool_calls',
                     'stream', 'temperature', 'max_tokens', 'api_key', 'headers', 'authorization'}:
        raise ValueError('extra_body cannot override core request fields or contain credentials')
    try:
        json.dumps(extra, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError('extra_body must contain finite JSON values') from None
    for section, keys in {'model': ('max_tokens', 'request_timeout_seconds'),
                          'limits': ('max_actions', 'max_model_calls', 'episode_timeout_seconds')}.items():
        for key in keys:
            if type(config[section].get(key)) is not int or config[section][key] < 1:
                raise ValueError(f'{section}.{key} must be a positive integer')
    if type(config['data'].get('seed')) is not int:
        raise ValueError('data.seed must be an integer')
    if not isinstance(config['algorithm'].get('options', {}), dict) or 'top_k' in config['algorithm']:
        raise ValueError('Use algorithm.options; there is no global top_k setting')
    for key in ('provider', 'base_url', 'name'):
        if not isinstance(model[key], str) or ((require_model or key == 'provider') and not model[key].strip()):
            raise ValueError(f'model.{key} must be a nonempty string')
    url = urlsplit(model['base_url'])
    if model['base_url'] and (url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment):
        raise ValueError('base_url must be an HTTP(S) API base address without credentials, query or fragment')
    if not isinstance(config['output'].get('directory'), str) or not config['output']['directory']:
        raise ValueError('output.directory must be a nonempty string')


def prepare_inputs(config, algorithm):
    domain = config['data']['dataset']
    environment = get_environment(domain)
    root = ROOT / 'data' / domain
    path = root / 'tasks.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [r.get('task_id') for r in rows]
    if not ids or any(not isinstance(t, str) or not t for t in ids) or len(ids) != len(set(ids)):
        raise ValueError('Task list must contain unique nonempty task IDs')
    if domain == 'alfworld':
        os.environ['ALFWORLD_DATA'] = str(environment.raw_data_root(root))
        for task_id in ids:
            environment.validate_task(root, task_id)
    else:
        from environments import task_rows
        known = task_rows(environment.raw_data_root(root))
        for row in rows:
            if row['task_id'] not in known or row['goal'] != known[row['task_id']]['goal']:
                raise ValueError('Task instruction does not match installed runtime resources')
    if hasattr(environment, 'preflight'):
        environment.preflight(root)
    # Check overlap even when the method chooses not to load the corpus.
    library_ids = {json.loads(l)['task_id'] for l in (root / 'trajectories.jsonl').read_text().splitlines() if l.strip()}
    if library_ids.intersection(ids):
        raise ValueError('Development tasks overlap the trajectory library')
    corpus, report = environment.load_expert_corpus(root) if algorithm.requires_corpus else ([], None)
    return ids, corpus, {'task_list_sha256': digest(rows), 'expert_corpus': report,
                         'task_splits': {r['task_id']: r.get('split', 'validation') for r in rows}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    parser.add_argument('--method', required=True, type=Path, help='Python method file')
    parser.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    parser.add_argument('--check', action='store_true', help='Check files and resources without model calls')
    parser.add_argument('--smoke', type=int, default=0, help='Run the first N tasks for integration only; not rankable')
    args = parser.parse_args(argv)
    if args.smoke < 0:
        parser.error('--smoke must be nonnegative')
    config = yaml.safe_load(args.config.read_text())
    validate_config(config, require_model=not args.check)
    config['data'].update(dataset=args.dataset, root=str(ROOT / 'data' / args.dataset), evaluation_split='validation')
    config['algorithm'].update(name=args.method.stem, kind='remote', method=str(args.method))
    algorithm = remote_algorithm(config, args.method)
    tasks, corpus, report = prepare_inputs(config, algorithm)
    if args.smoke:
        tasks = tasks[:args.smoke]
        config['data']['smoke'] = True
    if args.check:
        print(f'{args.dataset}: {len(tasks)} tasks; {len(corpus)} loaded references; checks passed. No model calls.')
        return 0
    result = asyncio.run(experiment(config, tasks, corpus, report, algorithm, ROOT))
    return 0 if result['summary']['complete'] else 1


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == '--episode-worker':
        asyncio.run(_execute_worker(json.loads(Path(sys.argv[2]).read_text())))
        raise SystemExit(0)
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from None
