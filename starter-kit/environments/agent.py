"""Use openJiuwen's loop; only adapt prompts, tools and recording."""
import asyncio
import json
import time
from uuid import uuid4

import hashlib

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

from algorithms.base import Guidance
from environments.algorithm_client import AlgorithmError, AlgorithmUnavailable

from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig
from openjiuwen.core.foundation.tool import LocalFunction, ToolCard
from openjiuwen.core.single_agent import AgentCard, ReActAgent, ReActAgentConfig, Session
from openjiuwen.core.single_agent.rail import AgentRail


class InvalidModelAction(ValueError):
    """A received model response violates the one-action contract; do not retry for a better answer."""


def model_failure(exc):
    """Classify typed transport errors at the model boundary, never by task score."""
    import httpx
    import openai
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, InvalidModelAction):
            return {'kind': 'output', 'exception': type(exc).__name__}
        status = getattr(exc, 'status_code', None)
        if status is None:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
        if isinstance(status, int):
            return {'kind': 'service' if status in (408, 429) or status >= 500 else 'configuration',
                    'http_status': status, 'exception': type(exc).__name__}
        if isinstance(exc, (openai.APIConnectionError, httpx.TransportError, TimeoutError)):
            return {'kind': 'service', 'exception': type(exc).__name__}
        exc = exc.__cause__ or exc.__context__
    return None


def model_settings(config, credentials):
    """Use the configured Chat Completions endpoint and explicit extensions."""
    from copy import deepcopy
    body = deepcopy(config.get('extra_body', {}))
    # Retain support for older organizer configs without applying gateway
    # extensions to other providers. New configs use extra_body directly.
    if credentials['provider'].lower() == 'openrouter':
        if 'provider_order' in config or 'allow_fallbacks' in config:
            routing = {'require_parameters': True,
                       'allow_fallbacks': config.get('allow_fallbacks', False)}
            providers = config.get('provider_order', [])
            if providers:
                routing.update(only=providers, order=providers)
            body.setdefault('provider', routing)
        if config.get('reasoning', 'auto') != 'auto':
            body.setdefault('reasoning', {'enabled': config['reasoning'] == 'enabled'})
    client = ModelClientConfig(
        client_provider=credentials['provider'], api_key=credentials['key'],
        api_base=credentials['base'], api_mode='chat_completions', verify_ssl=True,
        timeout=config['request_timeout_seconds'], max_retries=0,
        openrouter_enable_explicit_prompt_caching=False)
    # Provider-specific controls are explicit; never guess a thinking switch.
    model = ModelRequestConfig(model=credentials['model'], temperature=config['temperature'],
                               max_tokens=config['max_tokens'],
                               extra_body=body)
    return client, model


class RecordedModel(Model):
    """Use the SDK's final-request trace hook; no global monkeypatch or new loop."""
    def __init__(self, client, model, recorder, task_id, calls, max_calls=50):
        super().__init__(model_client_config=client, model_config=model)
        self.recorder, self.task_id, self.calls = recorder, task_id, calls
        self.max_calls, self.issued_calls = max_calls, 0

    async def invoke(self, *args, **kwargs):
        previous = kwargs.pop('tracer_record_data', None)
        captured = {}

        async def trace(**data):
            if 'llm_params' in data:
                params = data['llm_params']
                # Whitelist excludes headers, credentials and SDK objects.
                safe = {k: params[k] for k in ('model', 'messages', 'tools', 'tool_choice',
                        'temperature', 'top_p', 'max_tokens', 'parallel_tool_calls', 'stream',
                        'extra_body') if k in params}
                expected = self.model_config.extra_body
                if (safe.get('extra_body') or {}) != (expected or {}):
                    raise ValueError('SDK changed configured model request extensions')
                captured.update(safe)
                self.recorder.trace(self.task_id, 'api_request', request=safe,
                                    request_sha256=digest(safe))
            if previous:
                await previous(**data)

        started = time.perf_counter()
        response = None
        failure = None
        try:
            if self.issued_calls >= self.max_calls:
                raise InvalidModelAction('Model call budget exhausted; no request sent')
            self.issued_calls += 1
            response = await super().invoke(*args, tracer_record_data=trace, **kwargs)
            if len(getattr(response, 'tool_calls', None) or []) > 1:
                raise InvalidModelAction('Expected one action per model call; no actions executed')
            return response
        except Exception as exc:
            failure = model_failure(exc)
            raise
        finally:
            call = {'task_id': self.task_id, 'request_id': getattr(response, 'response_id', None),
                    'response_model': getattr(response, 'response_model', None),
                    'elapsed_seconds': time.perf_counter() - started,
                    'request_controls': {k: v for k, v in captured.items() if k not in ('messages', 'tools')},
                    'usage': getattr(response, 'usage_metadata', None), 'failure': failure}
            self.calls.append(call)
            self.recorder.trace(self.task_id, 'api_response', **{k:v for k,v in call.items() if k != 'task_id'})




class PlanningRail(AgentRail):
    def __init__(self, adapter, algorithm, recorder, stats):
        self.adapter = adapter
        self.algorithm = algorithm
        self.recorder = recorder
        self.stats = stats

    async def before_model_call(self, ctx):
        if self.adapter.done:
            ctx.request_force_finish({"output": self.adapter.snapshot(), "result_type": "answer"})
            return
        started = time.perf_counter()
        try:
            guidance = self.algorithm.get_guidance(self.adapter.planning_input())
        except AlgorithmError as exc:
            self.stats['algorithm_unavailable' if isinstance(exc, AlgorithmUnavailable) else 'algorithm_error'] = True
            raise
        if not isinstance(guidance, Guidance) or not isinstance(guidance.text, str):
            raise ValueError("get_guidance must return Guidance with string text")
        self.stats["guidance_calls"] += 1
        self.stats["guidance_seconds"] += time.perf_counter() - started
        ctx.agent.add_prompt_builder_section("planning_guidance", guidance.text, priority=20)
        self.stats["llm_requests"] += 1
        self.recorder.trace(self.adapter.task_id, "planning", guidance=guidance.text,
                            retrieved_ids=guidance.retrieved_ids)

    async def after_model_call(self, ctx):
        response = ctx.inputs.response
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            self.stats["usage_missing_calls"] += 1
        else:
            for key in ("input_tokens", "output_tokens", "reasoning_tokens"):
                self.stats[key] += getattr(usage, key, 0) or 0
        self.recorder.trace(self.adapter.task_id, "model", messages=ctx.inputs.messages,
                            tools=ctx.inputs.tools, response=response)

    async def on_model_exception(self, ctx):
        self.stats["usage_missing_calls"] += 1
        self.recorder.trace(self.adapter.task_id, "model_error", error=str(ctx.exception))

    async def before_tool_call(self, ctx):
        if self.adapter.done:
            ctx.request_force_finish({"output": self.adapter.snapshot(), "result_type": "answer"})

    async def after_tool_call(self, ctx):
        self.recorder.trace(self.adapter.task_id, "tool", name=ctx.inputs.tool_name,
                            arguments=ctx.inputs.tool_args, result=ctx.inputs.tool_result)
        if self.adapter.done:
            ctx.request_force_finish({"output": self.adapter.snapshot(), "result_type": "answer"})


async def execute_episode(adapter, algorithm, corpus, config, credentials, recorder, stats, api_calls=None):
    initial = adapter.reset()
    recorder.trace(adapter.task_id, "reset", goal=adapter.goal, **initial)
    started = time.perf_counter()
    try:
        algorithm.reset(adapter.planning_input(), corpus)
    except AlgorithmError as exc:
        stats['algorithm_unavailable' if isinstance(exc, AlgorithmUnavailable) else 'algorithm_error'] = True
        raise
    stats["retrieval_initialization_seconds"] = time.perf_counter() - started
    client, model = model_settings(config, credentials)
    card = AgentCard(id=adapter.domain + "-" + uuid4().hex, name=adapter.domain + " ReAct",
                     description="Fixed ReAct runtime for planning comparisons")
    agent = ReActAgent(card=card)
    agent.configure(ReActAgentConfig(
        model_name=credentials["model"], model_provider=credentials["provider"],
        model_client_config=client, model_config_obj=model,
        max_iterations=config["max_model_calls"], parallel_tool_calls=False,
        prompt_template=[{"role": "system", "content": adapter.system_prompt}],
    ))
    agent.set_llm(RecordedModel(client, model, recorder, adapter.task_id,
                               api_calls if api_calls is not None else [], config['max_model_calls']))

    def act(action: str):
        result = adapter.step(action)
        recorder.trace(adapter.task_id, "action", action=action, **result)
        return json.dumps(result, ensure_ascii=False)

    action_schema = {"type": "string"}
    if getattr(adapter, "tool_action_description", None):
        action_schema["description"] = adapter.tool_action_description
    tool = LocalFunction(ToolCard(
        id=adapter.tool_name, name=adapter.tool_name, description=adapter.tool_description,
        input_params={"type": "object", "properties": {"action": action_schema},
                      "required": ["action"], "additionalProperties": False},
        parallel_safe=False, stateless=False, idempotent=False,
    ), act)
    agent.ability_manager.add_ability(tool.card, tool)
    rail = PlanningRail(adapter, algorithm, recorder, stats)
    await agent.register_rail(rail)
    session = Session(session_id=uuid4().hex, card=card)
    try:
        return await agent.invoke({"query": json.dumps({"goal": adapter.goal, **initial},
                                                      ensure_ascii=False)}, session=session)
    finally:
        await agent.unregister_rail(rail)
        agent.ability_manager.teardown_tools()

