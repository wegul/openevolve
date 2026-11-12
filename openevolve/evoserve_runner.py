"""Expose OpenEvolve's native search steps to the EvoServe timing loop."""
import argparse
import asyncio
from concurrent.futures import BrokenExecutor
import os
import uuid
from pathlib import Path

import yaml

from openai import DefaultHttpxClient
from openevolve import OpenEvolve
from openevolve.config import Config
from openevolve.llm.openai import OpenAILLM
from openevolve.utils.metrics_utils import safe_numeric_average

from evoserve import EventLog, TaskSpec, run, tags
from evoserve.state import ArchiveState, Candidate, Program, Result


def score_of(metrics):
    return metrics.get('combined_score', safe_numeric_average(metrics)) if metrics else None


def build_config(spec):
    data = yaml.safe_load((spec.workload / 'config.yaml').read_text())
    data.setdefault('llm', {})['primary_model'] = spec.served_model_name
    if 'num_islands' in spec.algorithm:
        data.setdefault('database', {})['num_islands'] = spec.algorithm['num_islands']
    cfg = Config.from_dict(data)
    if cfg.prompt.template_dir:
        cfg.prompt.template_dir = str((spec.workload / Path(cfg.prompt.template_dir)).resolve())
    cfg.llm.update_model_params({'system_message': cfg.prompt.system_message})
    cfg.max_iterations = spec.budget
    cfg.checkpoint_interval = min(10, spec.budget)
    cfg.log_level = 'INFO'
    cfg.early_stopping_patience = None
    cfg.evolution_trace.enabled = False
    cfg.database.log_prompts = False
    cfg.random_seed = cfg.database.random_seed = spec.seed
    if os.environ.get('EVOSERVE_EVALUATION_POOL'):
        cfg.evaluator.timeout = max(cfg.evaluator.timeout,
                                    int(os.environ['EVOSERVE_EVALUATION_WAIT_SECONDS']))
    options = spec.algorithm
    if 'num_islands' in options:
        cfg.database.num_islands = options['num_islands']
    cfg.llm.api_base = spec.endpoint
    cfg.llm.primary_model = spec.served_model_name
    # Workload defaults and the campaign's generation controls apply to every model.
    for model in [cfg.llm, *cfg.llm.models, *cfg.llm.evaluator_models]:
        model.api_base, model.api_key, model.random_seed = spec.endpoint, 'local', spec.seed
        model.name = spec.served_model_name
        model.temperature = options.get('temperature', 0.0)
        for key in ('max_tokens', 'reasoning_effort', 'retries', 'retry_delay'):
            if key in options:
                setattr(model, key, options[key])
        if 'request_timeout' in options:
            model.timeout = options['request_timeout']
        model.init_client = EvoServeLLM
    return cfg


class PoolErrors:
    """Surface executor failures that native execution reports as candidate errors."""
    def __init__(self, executor):
        self.executor, self.error = executor, None

    def __getattr__(self, name):
        return getattr(self.executor, name)

    def submit(self, *args, **kwargs):
        try:
            future = self.executor.submit(*args, **kwargs)
        except BrokenExecutor as error:
            self.error = error
            raise

        def completed(done):
            if not done.cancelled() and isinstance(done.exception(), BrokenExecutor):
                self.error = done.exception()

        future.add_done_callback(completed)
        return future


class OpenEvolveSteps:
    mode = 'async'

    def __init__(self, evolution):
        self.evolution = evolution

    async def start(self, capacity):
        self.evolution.config.evaluator.parallel_evaluations = capacity
        await self.evolution.__aenter__()
        self.controller = self.evolution.parallel_controller
        self.controller.executor = PoolErrors(self.controller.executor)
        self.pending = [0] * self.controller.num_islands
        self.next_island = 0
        self.first_iteration = max(1, self.evolution.database.last_iteration + 1)

    async def stop(self):
        await self.evolution.__aexit__(None, None, None)

    def archive(self):
        db = self.evolution.database
        best = db.get_best_program()
        live = set().union(*db.islands) & db.programs.keys()
        return ArchiveState(tuple(Program(p.id, score_of(p.metrics), p.metadata.get('island'), p.iteration_found, p.id in live)
                                  for p in db.programs.values()), score_of(best.metrics) if best else None)

    def recombine(self, attempt):
        minimum = min(self.pending)
        island = next((self.next_island + offset) % len(self.pending) for offset in range(len(self.pending))
                      if self.pending[(self.next_island + offset) % len(self.pending)] == minimum)
        self.next_island = (island + 1) % len(self.pending)
        request = self.controller._prepare_request(self.first_iteration + attempt - 1, island)
        self.pending[island] += 1
        return Candidate(uuid.uuid4().hex, request.generation_number, island, request.parent_id, request)

    async def execute(self, candidate, values):
        try:
            result = await self.controller._execute_request(candidate.payload, self.evolution._save_checkpoint, tags=values)
            if self.controller.executor.error:
                raise self.controller.executor.error
            return Result(candidate.id, score_of(result.metrics), result.admitted, result.error,
                          result.generation_wall_time_seconds, result.evaluation_time_seconds,
                          result.metrics, result.child.id if result.child else None, result.admitted_at, result.artifacts)
        finally:
            self.pending[candidate.island] -= 1


def request_context(request):
    return dict(candidate_id=request.headers['x-evoserve-candidate'],
                api_call_index=int(request.headers['x-evoserve-call']),
                transport_retry_index=int(request.headers.get('x-stainless-retry-count', '0')),
                request_id=request.headers['x-request-id'])


def on_request(request):
    EventLog.current().write('request_started', **request_context(request), endpoint=str(request.url))


def on_response(response):
    response.read()  # Nonstreaming: the timestamp includes the complete response body.
    try:
        completion_id = response.json().get('id')
    except (ValueError, AttributeError):
        completion_id = None
    EventLog.current().write('request_finished', **request_context(response.request),
                             completion_id=completion_id, http_status=response.status_code)


class EvoServeLLM(OpenAILLM):
    def __init__(self, config):
        super().__init__(config)
        previous = self.client
        self.client = previous.with_options(max_retries=0, http_client=DefaultHttpxClient(
            event_hooks={'request': [on_request], 'response': [on_response]}))
        previous.close()
        self.candidate_id, self.calls = None, 0

    async def _call_api(self, params):
        if tags.current is None:
            raise RuntimeError('Inference requires a candidate context')
        candidate_id = tags.current['evoserve_candidate']
        if self.candidate_id != candidate_id:
            self.candidate_id, self.calls = candidate_id, 0
        context = dict(candidate_id=candidate_id, api_call_index=self.calls)
        self.calls += 1
        request = tags.to_request_kwargs(tags.current)
        headers = dict(params.get('extra_headers', {}), **request['extra_headers'],
                       **{'X-EvoServe-Candidate': candidate_id, 'X-EvoServe-Call': str(context['api_call_index'])})
        body = dict(params.get('extra_body', {}), **request['extra_body'], return_token_ids=True)
        budget = os.environ.get('EVOSERVE_THINKING_TOKEN_BUDGET')
        if budget is not None:
            body['thinking_token_budget'] = int(budget)
        try:
            content = await super()._call_api(dict(params, extra_headers=headers, extra_body=body))
            EventLog.current().write('api_call_succeeded', **context, request_id=headers['X-Request-Id'])
            return content
        except BaseException as error:
            EventLog.current().write('api_call_error', **context, request_id=headers['X-Request-Id'],
                                     error=f'{type(error).__name__}: {error}'[:2000])
            raise


def main(task_dir):
    spec = TaskSpec.load(task_dir)
    if 'thinking_token_budget' in spec.algorithm:
        os.environ['EVOSERVE_THINKING_TOKEN_BUDGET'] = str(spec.algorithm['thinking_token_budget'])
    initial = 'initial_program.rs' if (spec.workload / 'initial_program.rs').is_file() else 'initial_program.py'
    evolution = OpenEvolve(str(spec.workload / initial), str(spec.workload / 'evaluator.py'),
                           build_config(spec), str(spec.task_dir / 'output'))
    return asyncio.run(run(OpenEvolveSteps(evolution), spec))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task_dir')
    main(parser.parse_args().task_dir)
