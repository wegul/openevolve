"""Runtime lifecycle and detached inputs without preliminary control APIs."""
import asyncio
from contextlib import nullcontext
import inspect
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from openevolve import OpenEvolve
from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.process_parallel import ProcessParallelController


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_and_run_restore_signals_and_close_owned_logs(self):
        for mode in ('context', 'run', 'enter_failure', 'run_failure', 'save_failure', 'shutdown'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                initial = root/'initial.py'
                evaluator = root/'evaluator.py'
                initial.write_text('def solve(): return 0\n')
                evaluator.write_text('def evaluate(path): return {"combined_score": 0.0}\n')
                evolution = OpenEvolve(str(initial), str(evaluator), Config(), str(root/'output'))
                handlers = evolution._logging_handlers
                before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
                evaluate = AsyncMock(return_value={'combined_score': 0.0})
                if mode in ('enter_failure', 'run_failure'):
                    evaluate.side_effect = RuntimeError('initial evaluation failed')
                save = Mock(side_effect=RuntimeError('save failed') if mode == 'save_failure' else None)
                with patch.object(evolution.evaluator, 'evaluate_program', evaluate), \
                        patch.object(evolution, '_save_checkpoint'), \
                        patch.object(evolution, '_save_best_program', save), \
                        patch.object(evolution, '_run_evolution_with_checkpoints', AsyncMock()), \
                        patch.object(ProcessParallelController, 'start'), \
                        patch.object(ProcessParallelController, 'stop') as stop:
                    expected = (nullcontext() if mode in ('context', 'run') else self.assertRaises(
                        asyncio.CancelledError if mode == 'shutdown' else RuntimeError))
                    with expected:
                        if mode.startswith('run'):
                            await evolution.run(iterations=1)
                        else:
                            async with evolution:
                                if mode == 'shutdown':
                                    evolution.parallel_controller.request_shutdown()
                    if mode not in ('enter_failure', 'run_failure'):
                        stop.assert_called_once()
                self.assertIsNone(evolution.parallel_controller)
                for sig, previous in before.items():
                    self.assertIs(signal.getsignal(sig), previous)
                for handler in handlers:
                    self.assertTrue(handler._closed)

    def test_prepared_inputs_are_detached_from_database_and_each_read(self):
        config = Config()
        database = ProgramDatabase(config.database)
        database.add(Program(id='initial', code='def solve(): return 0', metrics={'combined_score': 0.0}))
        controller = ProcessParallelController(config, 'unused', database)
        request = controller._prepare_request(1, 0)
        database.programs[request.parent_id].code = 'changed'
        inputs = request.worker_inputs()
        self.assertEqual(inputs[1]['programs'][request.parent_id]['code'], 'def solve(): return 0')
        inputs[1]['programs'][request.parent_id]['code'] = 'also changed'
        self.assertEqual(request.worker_inputs()[1]['programs'][request.parent_id]['code'], 'def solve(): return 0')

    def test_runtime_has_no_explicit_selection_or_request_dispatch_policy(self):
        for cls in (OpenEvolve, ProcessParallelController):
            self.assertFalse(hasattr(cls, 'step'))
            self.assertNotIn('request_dispatcher', inspect.signature(cls).parameters)
