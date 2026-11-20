"""Host-side CPU container launcher. Python standard library only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def configuration_key_env(run, compose, *, require_model=True):
    """Parse YAML in the official image; the host needs only standard Python."""
    probe = """import json, yaml
from run import validate_config
with open('/app/config.yaml') as stream:
    config = yaml.safe_load(stream)
try:
    validate_config(config, require_model=REQUIRE_MODEL)
except ValueError as exc:
    raise SystemExit(str(exc)) from None
print(json.dumps(config['model'].get('api_key_env', 'MODEL_API_KEY')))
"""
    probe = probe.replace('REQUIRE_MODEL', repr(require_model))
    try:
        checked = run(compose + ['run', '--rm', '--no-deps', '--entrypoint', 'python',
                                 'official', '-c', probe], capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print('Configuration check failed: ' + (exc.stderr or '').strip(), file=sys.stderr)
        raise
    name = json.loads(checked.stdout)
    # Only a variable name crosses back to the launcher; never YAML or secrets.
    import re
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name) or name.endswith('_FILE'):
        raise ValueError('Invalid configured API key environment variable')
    return name


def read_model_key(key_env, environment):
    key_file = environment.get(key_env + '_FILE')
    key = Path(key_file).read_text().strip() if key_file else environment.get(key_env, '').strip()
    if not key:
        raise ValueError(f'Set {key_env} or {key_env}_FILE before model runs')
    return key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'check', 'run', 'verify'))
    parser.add_argument('--dataset', choices=('all', 'alfworld', 'webshop', 'hotpotqa'), default='all')
    parser.add_argument('--method', default='algorithms/baselines/bm25.py')
    parser.add_argument('--algorithm-image', default='agentic-algorithm:cpu')
    parser.add_argument('--official-image', default='agentic-official:cpu')
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--resources', type=Path, default=ROOT / 'resources')
    parser.add_argument('--config', type=Path, default=ROOT / 'config.yaml',
                        help='Local runtime configuration; mounted read-only in the official container')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    parser.add_argument('--smoke', type=int, default=0)
    parser.add_argument('--build', action='store_true', help='Build both images first')
    parser.add_argument('--cpus', type=float, default=4, help='Algorithm provisioning, not a finalized contest quota')
    parser.add_argument('--memory', default='8g', help='Algorithm provisioning, e.g. 8g; same for all compared runs')
    args = parser.parse_args()
    method = Path(args.method)
    if method.is_absolute() or '..' in method.parts or not method.parts or method.parts[0] != 'algorithms' or method.suffix != '.py':
        parser.error('Use a relative method path inside algorithms/')
    if args.smoke < 0 or args.cpus <= 0:
        parser.error('Invalid smoke count or CPU allocation')
    is_model_run = args.command == 'run'
    if not args.config.resolve().is_file():
        parser.error(f'Missing configuration: {args.config}')
    for name, path in [('data', args.data), ('resources', args.resources)]:
        if args.command not in ('build', 'verify') and not path.resolve().is_dir():
            parser.error(f'Missing {name}: {path}')
    # Dedicated per-launch directory: no chmod of existing results or data.
    output = args.output.resolve() / ('containers_' + uuid4().hex[:10])
    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o777)  # Official non-root UID writes here; not mounted to algorithm.
    (output / '.control').mkdir()
    (output / '.control').chmod(0o777)
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith('COMPOSE_'):
            environment.pop(key)
    environment.update(ALGORITHM_IMAGE=args.algorithm_image, OFFICIAL_IMAGE=args.official_image,
                       PUBLIC_DATA=str(args.data.resolve()), RUNTIME_RESOURCES=str(args.resources.resolve()),
                       RUN_OUTPUT=str(output),
                       ALGORITHM_CPUS=str(args.cpus), ALGORITHM_MEMORY=args.memory)
    project = 'agentic-' + uuid4().hex[:10]
    compose = ['docker', 'compose', '--project-name', project, '--file', str(ROOT / 'docker/compose.yaml')]

    def run(arguments, **kwargs):
        return subprocess.run(arguments, cwd=ROOT, env=environment, check=True, **kwargs)

    monitor_stop = threading.Event()
    monitor = None

    def monitor_restarts():
        control = output / '.control'
        handled = None
        while not monitor_stop.wait(.2):
            try:
                request = json.loads((control / 'request.json').read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            request_id = request.get('id')
            if request_id == handled or not isinstance(request_id, str):
                continue
            handled = request_id
            ok = False
            try:
                if request.get('operation') != 'restart_algorithm':
                    raise ValueError('Unknown control request')
                run(compose + ['stop', '--timeout', '0', 'algorithm'], timeout=15)
                run(compose + ['up', '-d', '--force-recreate', '--wait', '--wait-timeout', '60', 'algorithm'], timeout=70)
                ok = True
            except (subprocess.SubprocessError, ValueError):
                pass
            reply = control / 'reply.tmp'
            reply.write_text(json.dumps({'id': request_id, 'ok': ok}) + '\n')
            reply.replace(control / 'reply.json')

    with tempfile.TemporaryDirectory(prefix='agentic-secret-') as temporary:
        secret = Path(temporary) / 'model_key'
        # Freeze one configuration for all domains in this launch. Edits take
        # effect next time, not halfway through a run.
        runtime_config = Path(temporary) / 'config.yaml'
        runtime_config.write_bytes(args.config.resolve().read_bytes())
        runtime_config.chmod(0o444)
        environment['RUNTIME_CONFIG'] = str(runtime_config)
        secret.write_text('')
        secret.chmod(0o444)  # Parent temp directory is private; container reads this file only.
        environment['MODEL_KEY_FILE'] = str(secret)
        try:
            info = json.loads(run(['docker', 'info', '--format', '{{json .}}'],
                                  capture_output=True, text=True).stdout)
            capabilities = {name: info.get(name) for name in
                            ('ServerVersion', 'CgroupVersion', 'MemoryLimit', 'SwapLimit',
                             'CpuCfsQuota', 'PidsLimit')}
            (output / 'host_capabilities.json').write_text(json.dumps(capabilities, indent=2) + '\n')
            if capabilities.get('SwapLimit') is not True:
                print('Warning: host swap limiting is unavailable or unverified; '
                      'do not claim a RAM-plus-swap ceiling is enforced.', file=sys.stderr)
            run(compose + ['config', '--quiet'])
            if args.command == 'build' or args.build:
                run(compose + ['build'])
            if args.command == 'build':
                return 0
            images = {}
            for role, image_name in [('algorithm', args.algorithm_image), ('official', args.official_image)]:
                image = json.loads(run(['docker', 'image', 'inspect', image_name], capture_output=True, text=True).stdout)[0]
                images[role] = image['Id']
                if image.get('Config', {}).get('Volumes'):
                    raise ValueError('Image declares automatic volumes; use the provided no-volume template')
            # Pin exact local image IDs for this execution, not mutable tags.
            environment.update(ALGORITHM_IMAGE=images['algorithm'], OFFICIAL_IMAGE=images['official'],
                               ALGORITHM_IMAGE_ID=images['algorithm'], OFFICIAL_IMAGE_ID=images['official'])
            key_env = configuration_key_env(run, compose, require_model=is_model_run)
            if is_model_run:
                key = read_model_key(key_env, os.environ)
                secret.chmod(0o600)
                secret.write_text(key)
                secret.chmod(0o444)
            # Compose receives the secret file path, not the model credential.
            for name in (key_env, key_env + '_FILE', 'OPENROUTER_API_KEY', 'OPENROUTER_API_KEY_FILE'):
                # Preserve launcher-owned values if a chosen variable name
                # happens to overlap one (e.g. MODEL_KEY_FILE).
                if environment.get(name) == os.environ.get(name):
                    environment.pop(name, None)
            (output / 'launch.json').write_text(json.dumps({
                'images': images, 'command': args.command, 'method': args.method,
                'smoke': args.smoke, 'algorithm_cpus': args.cpus, 'algorithm_memory': args.memory,
                'platform': 'linux/amd64',
                'config_file': str(args.config.resolve()),
            }, indent=2) + '\n')
            run(compose + ['up', '-d', '--wait', '--wait-timeout', '120', 'algorithm'])
            monitor = threading.Thread(target=monitor_restarts, daemon=True)
            monitor.start()
            if args.command == 'verify':
                container_id = run(compose + ['ps', '-q', 'algorithm'], capture_output=True, text=True).stdout.strip()
                inspected = json.loads(run(['docker', 'inspect', container_id], capture_output=True, text=True).stdout)[0]
                host = inspected['HostConfig']
                if host.get('Privileged') or not host.get('ReadonlyRootfs') or host.get('PidMode') == 'host':
                    raise ValueError('Unsafe algorithm container configuration')
                if 'ALL' not in host.get('CapDrop', []) or not any(
                    value.startswith('no-new-privileges') for value in host.get('SecurityOpt', [])
                ):
                    raise ValueError('Missing algorithm capability restrictions')
                unexpected_mounts = [m for m in inspected.get('Mounts', []) if
                                     m.get('Type') != 'tmpfs' or m.get('Destination') not in ('/tmp', '/cache')]
                if unexpected_mounts or host.get('PortBindings'):
                    raise ValueError('Unexpected algorithm mounts or host ports')
                for network in inspected['NetworkSettings']['Networks']:
                    details = json.loads(run(['docker', 'network', 'inspect', network], capture_output=True, text=True).stdout)[0]
                    if not details.get('Internal'):
                        raise ValueError('Algorithm has an external network')
                (output / 'host_isolation_check.json').write_text(json.dumps({
                    'passed': True, 'container_id': container_id, 'read_only': True,
                    'privileged': False, 'host_ports': False, 'external_network': False,
                }, indent=2) + '\n')
                # Check the actual algorithm namespace, not just the YAML.
                probe = """import os, pathlib, socket
for p in ['/app/test', '/app/resources', '/app/environments', '/app/results', '/app/config.yaml', '/run/secrets/model_key', '/var/run/docker.sock']:
 assert not pathlib.Path(p).exists(), p
assert not os.environ.get('OPENROUTER_API_KEY')
assert not os.environ.get('MODEL_API_KEY_FILE')
assert os.getuid() != 0
assert 'CapEff:\\t0000000000000000' in pathlib.Path('/proc/self/status').read_text()
try:
 s=socket.create_connection(('1.1.1.1',443),timeout=3)
except OSError:
 pass
else:
 s.close(); raise AssertionError('Unexpected external connectivity')
print('Algorithm filesystem, identity, capability and egress probes passed.')
"""
                run(compose + ['exec', '-T', 'algorithm', 'python', '-c', probe])
                handshake = """from environments.algorithm_client import RemoteAlgorithm
from algorithms.base import PlanningInput
corpus=[{'task_id':'public-'+str(i),'goal':'find object '+str(i),
 'initial_observation':'room '+str(i), 'actions':['look'],
 'observations':['room '+str(i),'an object'],
 'steps':[{'action':'look','observation':'an object'}]} for i in range(4)]
state=PlanningInput('new-task','find object 2','room 2',('look',),(),'alfworld')
for name in ('no_retrieval','random','bm25','dense_static'):
 client=RemoteAlgorithm('http://algorithm:8765','algorithms/baselines/'+name+'.py',domain='alfworld')
 client.prepare(corpus); client.reset(state,corpus)
 result=client.get_guidance(state)
 assert isinstance(result.text,str) and result.text
 print('PASS two-container protocol:',name,flush=True)
import asyncio
from run import restart_algorithm
asyncio.run(restart_algorithm(client))
client.prepare(corpus); client.reset(state,corpus)
assert client.get_guidance(state).text
print('PASS trusted-host algorithm-container replacement and recovery',flush=True)
client.cancel()
"""
                run(compose + ['run', '--rm', '--no-deps', '--entrypoint', 'python',
                               'official', '-c', handshake])
                print('This probe is not a container-escape security audit.', flush=True)
                return 0
            common = ['--method', args.method]
            if args.smoke:
                common += ['--smoke', str(args.smoke)]
            base = compose + ['run', '--rm', '--no-deps', '--entrypoint', 'python', 'official']
            domains = ('alfworld', 'webshop', 'hotpotqa') if args.dataset == 'all' else (args.dataset,)
            for domain in domains:
                command = ['run.py', '--dataset', domain] + common
                if args.command == 'check':
                    command += ['--check']
                run(base + command)
            print(f'Host results: {output}', flush=True)
            return 0
        except (subprocess.CalledProcessError, KeyboardInterrupt, ValueError, OSError) as exc:
            (output / 'launch_error.json').write_text(json.dumps({'complete': False, 'error_type': type(exc).__name__}) + '\n')
            print(f'Run incomplete; inspect {output}. No final score should be inferred.', file=sys.stderr)
            if isinstance(exc, ValueError):
                print(str(exc), file=sys.stderr)
            return 1
        finally:
            monitor_stop.set()
            if monitor:
                monitor.join(timeout=90)
            try:
                subprocess.run(compose + ['down', '--timeout', '5', '--remove-orphans'], cwd=ROOT,
                               env=environment, stdout=subprocess.DEVNULL, timeout=30)
            except subprocess.TimeoutExpired:
                print('Docker cleanup did not finish; inspect project ' + project, file=sys.stderr)


if __name__ == '__main__':
    raise SystemExit(main())
