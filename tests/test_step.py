"""Explicit selections share worker execution and database updates with normal runs."""
import asyncio
from contextlib import nullcontext
from concurrent.futures import Future
import logging
import random
import re
import signal
import tempfile
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, Mock, patch

from openevolve import Config, OpenEvolve, StepResult
from openevolve.config import LLMModelConfig
from openevolve.database import Program, ProgramDatabase
from openevolve.population import PopulationStrategy
from openevolve.process_parallel import ProcessParallelController, _run_iteration_worker, default_prompt_context
import openevolve.process_parallel as parallel


class Executor:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def submit(self, function, *args):
        self.calls.append((function, args))
        future = Future()
        if self.result is not None:
            future.set_result(self.result)
        else:
            number, snapshot, parent, inspirations = args
            child = Program(id=f'candidate-{number}', parent_id=parent,
                            code=f'def solve(): return {number}',
                            metrics={'combined_score': number/100, 'validity': 1.0},
                            metadata={'island': 0})
            future.set_result(StepResult(child_program_dict=child.to_dict(), parent_id=parent,
                                         target_island=snapshot['sampling_island'], iteration=number,
                                         artifacts={'stdout': 'evaluated'}, prompt={'system': 's', 'user': 'u'}))
        return future


class LocalModel:
    """Picklable provider for a real process-pool test with no network calls."""
    def __init__(self, config):
        self.model = config.name
        self.last_usage = {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}

    async def generate_with_context(self, system_message, messages, **kwargs):
        number = int(re.findall(r'def solve\(\): return (\d+)', messages[-1]['content'])[-1]) + 1
        return f'```python\n# EVOLVE-BLOCK-START\ndef solve(): return {number}\n# EVOLVE-BLOCK-END\n```'


class ProcessPoolStepTests(unittest.IsolatedAsyncioTestCase):
    async def test_step_runs_generation_and_evaluation_in_real_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            initial = Path(tmp)/'initial.py'
            initial.write_text('# EVOLVE-BLOCK-START\ndef solve(): return 0\n# EVOLVE-BLOCK-END\n')
            evaluator = Path(tmp)/'evaluator.py'
            evaluator.write_text(
                'import re\nfrom pathlib import Path\n'
                'def evaluate(path):\n'
                '    n = float(re.search(r"return (\\d+)", Path(path).read_text())[1])\n'
                '    return {"combined_score": n, "validity": 1.0, "complexity": n, "diversity": n/10}\n')
            config = Config()
            config.random_seed = 42
            config.diff_based_evolution = False
            config.evaluator.cascade_evaluation = False
            config.evaluator.use_llm_feedback = False
            config.evaluator.parallel_evaluations = 1
            config.database.num_islands = 2
            config.llm.models = [LLMModelConfig(name='local', init_client=LocalModel)]
            config.llm.evaluator_models = []
            evolution = OpenEvolve(str(initial), str(evaluator), config, str(Path(tmp)/'output'))
            async with evolution:
                first = await asyncio.wait_for(evolution.step(
                    parent_id='initial', previous_programs=[], top_programs=[], inspirations=[],
                    target_island=1, generation_number=2), timeout=20)
                second = await asyncio.wait_for(evolution.step(
                    parent_id=first.child.id, previous_programs=[], top_programs=[], inspirations=[],
                    target_island=1, generation_number=6), timeout=20)
                self.assertTrue(first.admitted and second.admitted)
                self.assertEqual((first.metrics['combined_score'], second.metrics['combined_score']), (1.0, 2.0))
                self.assertEqual(second.child.parent_id, 'candidate-2')
                self.assertEqual(second.child.metadata['island'], 1)
                self.assertNotIn(first.child.id, evolution.database.programs)
                third = await asyncio.wait_for(evolution.step(
                    parent_id=first.child.id, previous_programs=[first.child.id],
                    top_programs=[second.child.id], inspirations=[first.child.id],
                    target_island=1, generation_number=18), timeout=20)
                self.assertTrue(third.admitted, third.error)
                self.assertEqual(third.child.parent_id, first.child.id)
                self.assertNotIn(first.child.id, evolution.database.programs)
            self.assertTrue((Path(tmp)/'output/best/best_program.py').is_file())


class StepTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = Config()
        self.config.database.num_islands = 2
        self.config.database.in_memory = True
        self.config.database.feature_dimensions = ['complexity', 'diversity']
        self.config.database.feature_bins = 20
        self.config.evaluator.parallel_evaluations = 1
        self.config.random_seed = 42
        self.db = ProgramDatabase(self.config.database)
        self.db.add(Program(id='initial', code='def solve(): return 0', metrics={'combined_score': 0.0}))
        self.controller = ProcessParallelController(self.config, 'unused.py', self.db)
        self.executor = Executor()
        self.controller.executor = self.executor

    def evolution(self, directory):
        initial = Path(directory)/'initial.py'
        initial.write_text('def solve(): return 0')
        evaluator = Path(directory)/'evaluator.py'
        evaluator.write_text('def evaluate(path): return {"combined_score": 0.0}')
        self.config.llm.models = [LLMModelConfig(name='local', init_client=LocalModel)]
        self.config.llm.evaluator_models = []
        return OpenEvolve(str(initial), str(evaluator), self.config, str(Path(directory)/'output'))

    async def test_shutdown_and_cancellation_release_a_running_step(self):
        for interruption in ('before_submit', 'shutdown', 'cancel'):
            with self.subTest(interruption=interruption):
                future = Future()
                future.set_running_or_notify_cancel()
                controller = ProcessParallelController(self.config, 'unused.py', self.db)
                controller.executor = Mock(submit=Mock(return_value=future))
                if interruption == 'before_submit':
                    controller.request_shutdown()
                task = asyncio.create_task(controller.step(
                    parent_id='initial', previous_programs=[], top_programs=[], inspirations=[],
                    target_island=1, generation_number=2))
                await asyncio.sleep(0)
                if interruption == 'shutdown':
                    controller.request_shutdown()
                else:
                    task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=1)
                future.set_result(StepResult(error='interrupted', iteration=2))
                self.assertEqual(set(self.db.programs), {'initial'})
                self.assertEqual(self.db.last_iteration, 0)
                if interruption == 'before_submit':
                    controller.executor.submit.assert_not_called()

    async def test_result_is_committed_before_returning_and_labels_stay_sparse(self):
        result = await self.controller.step(parent_id='initial', previous_programs=[], top_programs=[],
                                            inspirations=[], target_island=1, generation_number=2)
        self.assertTrue(result.admitted)
        self.assertEqual(result.child.id, 'candidate-2')
        self.assertEqual(result.metrics['combined_score'], 0.02)
        self.assertIn(result.child.id, self.db.islands[1])
        self.assertEqual(self.db.island_generations[1], 1)
        self.assertEqual(self.db.get_artifacts(result.child.id), {'stdout': 'evaluated'})
        await self.controller.step(parent_id='candidate-2', previous_programs=['candidate-2', 'initial'],
                                   top_programs=[], inspirations=[], target_island=1, generation_number=6)
        args = self.executor.calls[-1][1]
        self.assertEqual(args[0], 6)
        self.assertEqual(args[1]['prompt_selection']['previous_programs'], ['candidate-2', 'initial'])
        self.assertEqual(args[1]['prompt_selection']['top_programs'], [])
        self.assertEqual(args[3], [])

    async def test_unavailable_selection_and_bad_island_do_not_submit(self):
        for updates in ({'parent_id': 'missing'}, {'inspirations': ['missing']}, {'target_island': 2},
                        {'generation_number': 0}, {'top_programs': {'initial'}}):
            inputs = dict(parent_id='initial', previous_programs=[], top_programs=[], inspirations=[],
                          target_island=1, generation_number=2)
            inputs.update(updates)
            with self.assertRaises(ValueError):
                await self.controller.step(**inputs)
        self.assertFalse(self.executor.calls)

    async def test_replaced_program_can_be_selected_without_restoring_population(self):
        self.config.database.feature_dimensions = ['slot']
        self.config.database.feature_bins = 1
        first = Program(id='candidate-2', parent_id='initial', code='def solve(): return 2',
                        metrics={'combined_score': 0.2, 'validity': 0.0, 'slot': 0.0})
        self.executor.result = StepResult(child_program_dict=first.to_dict(), parent_id='initial',
                                         target_island=1, iteration=2,
                                         artifacts={'error_message': 'invalid result', 'blob': b'feedback'})
        await self.controller.step(parent_id='initial', previous_programs=[], top_programs=[],
                                   inspirations=[], target_island=1, generation_number=2)
        feedback = self.db.get_artifacts(first.id)
        second = Program(id='candidate-12', parent_id=first.id, code='def solve(): return 12',
                         metrics={'combined_score': 0.9, 'validity': 1.0, 'slot': 0.0})
        self.executor.result = StepResult(child_program_dict=second.to_dict(), parent_id=first.id,
                                         target_island=1, iteration=12)
        await self.controller.step(parent_id=first.id, previous_programs=[], top_programs=[],
                                   inspirations=[], target_island=1, generation_number=12)
        self.assertNotIn(first.id, self.db.programs)
        self.assertNotIn(first.id, self.db.islands[1])
        self.assertNotIn(first.id, self.db.archive)
        active = set(self.db.programs)
        self.executor.result = StepResult(error='generation failed', iteration=16)
        await self.controller.step(parent_id=first.id, previous_programs=[first.id, 'initial'],
                                   top_programs=[second.id, first.id], inspirations=[first.id],
                                   target_island=1, generation_number=16)
        snapshot = self.executor.calls[-1][1][1]
        self.assertEqual(snapshot['programs'][first.id]['code'], first.code)
        self.assertEqual(snapshot['programs'][first.id]['metrics'], first.metrics)
        self.assertEqual(snapshot['artifacts'][first.id], feedback)
        self.assertEqual(snapshot['prompt_selection']['previous_programs'], [first.id, 'initial'])
        self.assertEqual(snapshot['prompt_selection']['top_programs'], [second.id, first.id])
        self.assertEqual(set(self.db.programs), active)
        self.assertNotIn(first.id, snapshot['islands'][1])
        context = default_prompt_context(snapshot, second.id, self.config.prompt)
        self.assertNotIn(first.id, context['previous_programs'] + context['top_programs'])

    async def test_worker_failure_and_admission_rejection_are_returned(self):
        self.executor.result = StepResult(error='generation failed', iteration=2)
        result = await self.controller.step(parent_id='initial', previous_programs=[], top_programs=[],
                                            inspirations=[], target_island=1, generation_number=2)
        self.assertEqual(result.error, 'generation failed')
        self.assertIsNone(result.child)
        self.assertEqual(self.db.last_iteration, 2)
        self.executor.result = None
        self.db.population_strategy = PopulationStrategy(admit=lambda *args: False)
        result = await self.controller.step(parent_id='initial', previous_programs=[], top_programs=[],
                                            inspirations=[], target_island=1, generation_number=6)
        self.assertIn('candidate-6', result.error)
        self.assertFalse(result.admitted)
        self.assertNotIn('candidate-6', self.db.programs)

    async def test_processing_error_preserves_committed_child_metadata(self):
        with patch.object(self.db, "store_artifacts", side_effect=OSError("storage failed")):
            result = await self.controller.step(parent_id="initial", previous_programs=[], top_programs=[],
                                                inspirations=[], target_island=1, generation_number=2)
        self.assertIn("storage failed", result.error)
        self.assertTrue(result.admitted)
        self.assertEqual(result.child.metadata["island"], 1)
        self.assertIn(result.child.id, self.db.islands[1])

    async def test_normal_submission_and_step_prepare_identical_worker_inputs(self):
        parent = self.db.programs['initial']
        snapshot = self.controller._create_database_snapshot()
        context = default_prompt_context(snapshot, parent.id, self.config.prompt)
        with patch.object(self.db, 'sample_from_island', return_value=(parent, [])):
            self.controller._submit_iteration(2, 1)
        ordinary = self.executor.calls[-1]
        await self.controller.step(parent_id=parent.id, **context, inspirations=[],
                                   target_island=1, generation_number=2)
        self.assertEqual(ordinary, self.executor.calls[-1])

    async def test_run_and_failed_contexts_restore_signals_and_close_logs(self):
        for mode in ('steps', 'run', 'enter_failure', 'run_failure', 'save_failure', 'shutdown'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                evolution = self.evolution(tmp)
                handlers = evolution._logging_handlers
                before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
                evaluate = AsyncMock(return_value={'combined_score': 0.0})
                if mode in ('enter_failure', 'run_failure'):
                    evaluate.side_effect = RuntimeError('initial evaluation failed')
                save = evolution._save_best_program if mode != 'save_failure' else Mock(
                    side_effect=RuntimeError('save failed'))
                def start(controller):
                    controller.executor = self.executor
                with patch.object(evolution.evaluator, 'evaluate_program', evaluate), \
                        patch.object(evolution, '_run_evolution_with_checkpoints', AsyncMock()), \
                        patch.object(evolution, '_save_best_program', save), \
                        patch.object(ProcessParallelController, 'start', start), \
                        patch.object(ProcessParallelController, 'stop') as stop:
                    with nullcontext() if mode in ('steps', 'run') else self.assertRaises(
                            asyncio.CancelledError if mode == 'shutdown' else RuntimeError):
                        if mode.startswith('run'):
                            await evolution.run(iterations=1)
                        else:
                            async with evolution:
                                if mode == 'shutdown':
                                    evolution.parallel_controller.request_shutdown()
                                if mode == 'steps':
                                    for number in (2, 6):
                                        await evolution.step(
                                            parent_id='initial', previous_programs=[], top_programs=[],
                                            inspirations=[], target_island=1, generation_number=number)
                    self.assertEqual(stop.call_count, int(mode in ('steps', 'run', 'save_failure', 'shutdown')))
                self.assertEqual(evaluate.await_count, 1)
                if mode == 'steps':
                    self.assertEqual(set(evolution.generated_programs), {'initial', 'candidate-2', 'candidate-6'})
                    self.assertTrue((Path(tmp)/'output/checkpoints/checkpoint_6/metadata.json').is_file())
                if mode in ('steps', 'run'):
                    self.assertTrue((Path(tmp)/'output/best/best_program.py').is_file())
                self.assertIsNone(evolution.parallel_controller)
                self.assertEqual(before, {sig: signal.getsignal(sig) for sig in before})
                self.assertFalse(set(handlers) & set(logging.getLogger().handlers))
                self.assertIsNone(handlers[0].stream)
                with self.assertRaises(RuntimeError):
                    await evolution.step(parent_id='initial', previous_programs=[], top_programs=[],
                                         inspirations=[], target_island=1, generation_number=18)


class WorkerSelectionTests(unittest.TestCase):
    def test_worker_obeys_lists_and_preserves_failure_prompt_and_seed(self):
        config = Config()
        config.random_seed = 42
        config.diff_based_evolution = False
        config.prompt.programs_as_changes_description = False
        sampler = Mock()
        sampler.build_prompt.return_value = {'system': 'system', 'user': 'user'}
        ensemble = Mock(random_state=random.Random())
        ensemble.last_usage = {'prompt_tokens': 10}
        ensemble.generate_with_context = AsyncMock(return_value=None)
        programs = {pid: Program(id=pid, code=f'def solve(): return "{pid}"',
                                metrics={'combined_score': 0.1}, metadata={'island': 0}).to_dict()
                    for pid in ('parent', 'first', 'second', 'hidden')}
        snapshot = {'programs': programs, 'islands': [list(programs), []], 'current_island': 0,
                    'sampling_island': 1, 'artifacts': {},
                    'prompt_selection': {'previous_programs': ['second', 'first'], 'top_programs': []}}
        with patch.object(parallel, '_lazy_init_worker_components'), \
                patch.object(parallel, '_worker_config', config, create=True), \
                patch.object(parallel, '_worker_prompt_sampler', sampler, create=True), \
                patch.object(parallel, '_worker_llm_ensemble', ensemble, create=True):
            result = _run_iteration_worker(18, snapshot, 'parent', [])
        kwargs = sampler.build_prompt.call_args.kwargs
        self.assertEqual([p['id'] for p in kwargs['previous_programs']], ['second', 'first'])
        self.assertEqual(kwargs['top_programs'], [])
        self.assertEqual(kwargs['inspirations'], [])
        self.assertEqual(kwargs['evolution_round'], 18)
        self.assertEqual(kwargs['island_id'], 1)
        self.assertEqual(ensemble.random_state.getstate(), random.Random(60).getstate())
        self.assertEqual(result.prompt, {'system': 'system', 'user': 'user'})
        self.assertEqual(result.token_usage, {'prompt_tokens': 10})
        self.assertIn('None', result.error)


if __name__ == '__main__':
    unittest.main()
