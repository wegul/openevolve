"""Repeat identity is independent of hash layout, worker history and timings."""
import copy
import os
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openevolve.config import Config, DatabaseConfig
from openevolve.database import Program, ProgramDatabase
from openevolve import process_parallel as parallel
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.metrics_utils import get_fitness_score


class RandomLLM:
    def __init__(self):
        self.random_state = random.Random()

    async def generate_with_context(self, **kwargs):
        value = random.random() + np.random.random() + self.random_state.random()
        return f'```python\ndef solution():\n    return {value!r}\n```'


class Evaluator:
    async def evaluate_program(self, code, program_id):
        return {'combined_score': 0.7}

    def get_pending_artifacts(self, program_id):
        return {}


class ReproducibilityTests(unittest.TestCase):
    def test_sampling_is_independent_of_python_hash_seed(self):
        source = '''
import json
from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase
db = ProgramDatabase(DatabaseConfig(num_islands=1, random_seed=42))
for i in range(12):
    p = Program(id=f'candidate-{i}', code=str(i), iteration_found=i)
    db.programs[p.id] = p
    db.islands[0].add(p.id)
print(json.dumps([db._sample_from_island_random(0).id for _ in range(20)]))
'''
        outputs = [subprocess.check_output(
            [sys.executable, '-c', source],
            env={**os.environ, 'PYTHONHASHSEED': seed}, text=True,
        ) for seed in ('0', '42', '123')]
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], outputs[2])

    def test_candidate_identity_is_independent_of_worker_history(self):
        config = Config(random_seed=42, diff_based_evolution=False, language='python')
        parent = Program(id='initial', code='def solution(): return 0',
                         metrics={'combined_score': 0.0})
        snapshot = {'programs': {'initial': parent.to_dict()}, 'islands': [['initial']],
                    'current_island': 0, 'sampling_island': 0, 'artifacts': {},
                    'feature_dimensions': []}
        initialized = False

        def initialize():
            nonlocal initialized
            if not initialized:
                random.random()
                np.random.random()
                initialized = True

        with patch.object(parallel, '_worker_config', config, create=True), \
                patch.object(parallel, '_worker_llm_ensemble', RandomLLM(), create=True), \
                patch.object(parallel, '_worker_prompt_sampler', PromptSampler(config.prompt), create=True), \
                patch.object(parallel, '_worker_evaluator', Evaluator(), create=True), \
                patch.object(parallel, '_lazy_init_worker_components', side_effect=initialize):
            first = parallel._run_iteration_worker(17, snapshot, 'initial', [])
            parallel._run_iteration_worker(2, snapshot, 'initial', [])
            second = parallel._run_iteration_worker(17, snapshot, 'initial', [])
        self.assertIsNone(first.error)
        self.assertIsNone(second.error)
        for key in ('id', 'code', 'metrics', 'parent_id', 'iteration_found'):
            self.assertEqual(first.child_program_dict[key], second.child_program_dict[key])
        self.assertEqual(first.child_program_dict['id'], 'candidate-17')

    def test_eval_time_is_retained_but_does_not_change_prompts_or_fitness(self):
        sampler = PromptSampler(Config().prompt)
        def prompt(duration):
            metrics = {'combined_score': 0.7, 'eval_time': duration}
            program = {'id': 'candidate-1', 'code': 'pass', 'metrics': metrics,
                       'metadata': {'parent_metrics': metrics}}
            before = copy.deepcopy(program)
            result = sampler.build_prompt(current_program='pass', program_metrics=metrics,
                previous_programs=[program], top_programs=[program], inspirations=[program],
                program_artifacts={'eval_time': duration, 'diagnostic': 'stable'})
            self.assertEqual(program, before)
            return result
        self.assertEqual(prompt(0.001), prompt(999.0))
        self.assertEqual(get_fitness_score({'score': 0.7, 'eval_time': 999}), 0.7)

    def test_copy_ids_remain_unique_after_checkpoint_resume(self):
        db = ProgramDatabase(DatabaseConfig(num_islands=1))
        self.assertEqual(db._copy_id(), 'copy-1')
        with tempfile.TemporaryDirectory() as temp:
            db.save(temp)
            restored = ProgramDatabase(DatabaseConfig(num_islands=1))
            restored.load(temp)
        self.assertEqual(restored._copy_id(), 'copy-2')
