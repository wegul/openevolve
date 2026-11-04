"""
Process-based parallel controller for true parallelism
"""

import asyncio
import logging
import multiprocessing as mp
import pickle
import random
import signal
import time
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.selection import IslandSelectionContext, IslandSelector, IslandState
from openevolve.utils.metrics_utils import safe_numeric_average

logger = logging.getLogger(__name__)


class _CandidateRejected(Exception):
    """Stop processing an iteration after admission rejects its child."""


@dataclass(frozen=True)
class PreparedRequest:
    """Detached worker inputs; creation number fixes candidate identity and RNG seed."""

    generation_number: int
    target_island: int
    parent_id: str
    _payload: bytes

    @property
    def request_id(self) -> int:
        return self.generation_number

    def worker_inputs(self):
        """Return independent copies, including records no longer in the database."""
        snapshot, inspirations = pickle.loads(self._payload)
        return self.generation_number, snapshot, self.parent_id, inspirations


@dataclass
class SerializableResult:
    """Result that can be pickled and sent between processes"""

    child_program_dict: Optional[Dict[str, Any]] = None
    parent_id: Optional[str] = None
    iteration_time: float = 0.0
    prompt: Optional[Dict[str, str]] = None
    llm_response: Optional[str] = None
    artifacts: Optional[Dict[str, Any]] = None
    iteration: int = 0
    error: Optional[str] = None
    target_island: Optional[int] = None  # Island where child should be placed
    token_usage: Optional[Dict[str, Any]] = None
    admitted: bool = False
    admitted_at: Optional[float] = None
    evaluation_time_seconds: Optional[float] = None
    generation_wall_time_seconds: Optional[float] = None

    @property
    def child(self) -> Optional[Program]:
        return Program(**self.child_program_dict) if self.child_program_dict else None

    @property
    def metrics(self) -> Dict[str, Any]:
        return self.child_program_dict.get("metrics", {}) if self.child_program_dict else {}


def _worker_init(config_dict: dict, evaluation_file: str, parent_env: dict = None) -> None:
    """Initialize worker process with necessary components"""
    import os

    # Set environment from parent process
    if parent_env:
        os.environ.update(parent_env)

    global _worker_config
    global _worker_evaluation_file
    global _worker_evaluator
    global _worker_llm_ensemble
    global _worker_prompt_sampler

    # Store config for later use
    # Reconstruct Config object from nested dictionaries
    from openevolve.config import (
        Config,
        DatabaseConfig,
        EvaluatorConfig,
        LLMConfig,
        LLMModelConfig,
        PromptConfig,
    )

    # Reconstruct model objects
    models = [LLMModelConfig(**m) for m in config_dict["llm"]["models"]]
    evaluator_models = [LLMModelConfig(**m) for m in config_dict["llm"]["evaluator_models"]]

    # Create LLM config with models
    llm_dict = config_dict["llm"].copy()
    llm_dict["models"] = models
    llm_dict["evaluator_models"] = evaluator_models
    llm_config = LLMConfig(**llm_dict)

    # Create other configs
    prompt_config = PromptConfig(**config_dict["prompt"])
    database_config = DatabaseConfig(**config_dict["database"])
    evaluator_config = EvaluatorConfig(**config_dict["evaluator"])

    _worker_config = Config(
        llm=llm_config,
        prompt=prompt_config,
        database=database_config,
        evaluator=evaluator_config,
        **{
            k: v
            for k, v in config_dict.items()
            if k not in ["llm", "prompt", "database", "evaluator"]
        },
    )
    _worker_evaluation_file = evaluation_file

    # These will be lazily initialized on first use
    _worker_evaluator = None
    _worker_llm_ensemble = None
    _worker_prompt_sampler = None


def _lazy_init_worker_components():
    """Lazily initialize expensive components on first use"""
    global _worker_evaluator
    global _worker_llm_ensemble
    global _worker_prompt_sampler

    if _worker_llm_ensemble is None:
        from openevolve.llm.ensemble import LLMEnsemble

        _worker_llm_ensemble = LLMEnsemble(_worker_config.llm.models)

    if _worker_prompt_sampler is None:
        from openevolve.prompt.sampler import PromptSampler

        _worker_prompt_sampler = PromptSampler(_worker_config.prompt)

    if _worker_evaluator is None:
        from openevolve.evaluator import Evaluator
        from openevolve.llm.ensemble import LLMEnsemble
        from openevolve.prompt.sampler import PromptSampler

        # Create evaluator-specific components
        evaluator_llm = LLMEnsemble(_worker_config.llm.evaluator_models)
        evaluator_prompt = PromptSampler(_worker_config.prompt)
        evaluator_prompt.set_templates("evaluator_system_message")

        _worker_evaluator = Evaluator(
            _worker_config.evaluator,
            _worker_evaluation_file,
            evaluator_llm,
            evaluator_prompt,
            database=None,  # No shared database in worker
            suffix=getattr(_worker_config, "file_suffix", ".py"),
        )


def default_prompt_context(snapshot: Dict[str, Any], parent_id: str, prompt_config) -> Dict[str, List[str]]:
    """Apply the ordinary island ranking rules before executing a generation."""
    programs = snapshot["programs"]
    parent = programs[parent_id]
    island = parent.get("metadata", {}).get("island", snapshot["current_island"])
    ranked = [pid for pid in snapshot["islands"][island] if pid in programs]
    ranked.sort(
        key=lambda pid: programs[pid]["metrics"].get(
            "combined_score", safe_numeric_average(programs[pid]["metrics"])
        ),
        reverse=True,
    )
    return {
        "previous_programs": ranked[:prompt_config.num_top_programs],
        "top_programs": ranked[:prompt_config.num_top_programs + prompt_config.num_diverse_programs],
    }


def _run_iteration_worker(
    iteration: int, db_snapshot: Dict[str, Any], parent_id: str, inspiration_ids: List[str]
) -> SerializableResult:
    """Run a single iteration in a worker process"""
    prompt = None
    llm_response = None
    token_usage = None
    evaluation_time_seconds = 0.0
    generation_wall_time_seconds = 0.0
    worker_started = time.time()

    def make_result(**fields):
        defaults = dict(iteration=iteration, parent_id=parent_id, prompt=prompt,
                        llm_response=llm_response, token_usage=token_usage,
                        target_island=db_snapshot.get("sampling_island"),
                        iteration_time=time.time() - worker_started,
                        evaluation_time_seconds=evaluation_time_seconds,
                        generation_wall_time_seconds=generation_wall_time_seconds)
        defaults.update(fields)
        return SerializableResult(**defaults)

    try:
        # Lazy initialization
        _lazy_init_worker_components()

        # A task's random choices must not depend on its worker's earlier tasks.
        if _worker_config.random_seed is not None:
            seed = _worker_config.random_seed + iteration
            random.seed(seed)
            np.random.seed(seed % (2**32))
            _worker_llm_ensemble.random_state.seed(seed)

        # Reconstruct programs from snapshot
        programs = {pid: Program(**prog_dict) for pid, prog_dict in db_snapshot["programs"].items()}

        parent = programs[parent_id]
        inspirations = [programs[pid] for pid in inspiration_ids if pid in programs]

        # Get parent artifacts if available
        parent_artifacts = db_snapshot["artifacts"].get(parent_id)

        parent_island = parent.metadata.get("island", db_snapshot["current_island"])
        selection = db_snapshot.get("prompt_selection")
        if selection is None:
            selection = default_prompt_context(db_snapshot, parent_id, _worker_config.prompt)
        best_programs_only = [programs[pid] for pid in selection["previous_programs"]]
        programs_for_prompt = [programs[pid] for pid in selection["top_programs"]]

        # Build prompt
        if _worker_config.prompt.programs_as_changes_description:
            parent_changes_desc = (
                parent.changes_description or _worker_config.prompt.initial_changes_description
            )
            child_changes_desc = parent_changes_desc
        else:
            parent_changes_desc = None
            child_changes_desc = None

        prompt = _worker_prompt_sampler.build_prompt(
            current_program=parent.code,
            parent_program=parent.code,
            program_metrics=parent.metrics,
            previous_programs=[p.to_dict() for p in best_programs_only],
            top_programs=[p.to_dict() for p in programs_for_prompt],
            inspirations=[p.to_dict() for p in inspirations],
            language=_worker_config.language,
            evolution_round=iteration,
            diff_based_evolution=_worker_config.diff_based_evolution,
            program_artifacts=parent_artifacts,
            feature_dimensions=db_snapshot.get("feature_dimensions", []),
            current_changes_description=parent_changes_desc,
            island_id=db_snapshot.get("sampling_island"),
        )

        iteration_start = time.time()

        # Measure the full model call, including retries and failed calls.
        generation_started = time.perf_counter()
        try:
            llm_response = asyncio.run(
                _worker_llm_ensemble.generate_with_context(
                    system_message=prompt["system"],
                    messages=[{"role": "user", "content": prompt["user"]}],
                )
            )
            token_usage = getattr(_worker_llm_ensemble, "last_usage", None)
            generation_wall_time_seconds = time.perf_counter() - generation_started
        except Exception as e:
            generation_wall_time_seconds = time.perf_counter() - generation_started
            logger.error(f"LLM generation failed: {e}")
            return make_result(error=f"LLM generation failed: {str(e)}")

        # Check for None response
        if llm_response is None:
            return make_result(error="LLM returned None response")

        # Parse response based on evolution mode
        if _worker_config.diff_based_evolution:
            from openevolve.utils.code_utils import (
                apply_diff_blocks,
                extract_diffs,
                format_diff_summary,
                split_diffs_by_target,
            )

            try:
                diff_blocks = extract_diffs(llm_response, _worker_config.diff_pattern)
            except ValueError as exc:
                return make_result(error=str(exc))
            if not diff_blocks:
                return make_result(error="No valid diffs found in response")

            if _worker_config.prompt.programs_as_changes_description:
                try:
                    code_blocks, desc_blocks, _unmatched = split_diffs_by_target(
                        diff_blocks,
                        code_text=parent.code,
                        changes_description_text=parent_changes_desc,
                    )
                except Exception as e:
                    return make_result(error=str(e))

                child_code, _ = apply_diff_blocks(parent.code, code_blocks)
                child_changes_desc, desc_applied = apply_diff_blocks(
                    parent_changes_desc, desc_blocks
                )

                # Must update the previous changes description
                if (
                    desc_applied == 0
                    or not child_changes_desc.strip()
                    or child_changes_desc.strip() == parent_changes_desc.strip()
                ):
                    return make_result(
                        error="changes_description was not updated or empty, program is discarded"
                    )

                changes_summary = format_diff_summary(
                    code_blocks,
                    max_line_len=_worker_config.prompt.diff_summary_max_line_len,
                    max_lines=_worker_config.prompt.diff_summary_max_lines,
                )
            else:
                # All diffs applied only to code
                child_code, applied = apply_diff_blocks(parent.code, diff_blocks)
                if applied == 0:
                    return make_result(
                        error=(
                            f"None of the {len(diff_blocks)} SEARCH block(s) matched the "
                            "parent program"
                        ),
                    )
                if applied < len(diff_blocks):
                    logger.warning(
                        f"Iteration {iteration}: only {applied} of {len(diff_blocks)} "
                        "SEARCH block(s) matched the parent program"
                    )
                if child_code == parent.code:
                    return make_result(error="Diff did not change the parent program")
                changes_summary = format_diff_summary(
                    diff_blocks,
                    max_line_len=_worker_config.prompt.diff_summary_max_line_len,
                    max_lines=_worker_config.prompt.diff_summary_max_lines,
                )
        else:
            from openevolve.utils.code_utils import parse_full_rewrite

            new_code = parse_full_rewrite(llm_response, _worker_config.language)
            if not new_code:
                return make_result(error=f"No valid code found in response")

            if new_code == parent.code:
                return make_result(error="Rewrite is identical to the parent program")

            child_code = new_code
            changes_summary = "Full rewrite"

        # Revert edits made outside the EVOLVE-BLOCK regions if configured
        if _worker_config.enforce_evolve_blocks:
            from openevolve.utils.code_utils import enforce_evolve_blocks

            try:
                enforced_code = enforce_evolve_blocks(parent.code, child_code)
            except ValueError as exc:
                return make_result(error=str(exc))
            if enforced_code != child_code:
                logger.info(
                    f"Iteration {iteration}: reverted edits outside the EVOLVE-BLOCK regions"
                )
                child_code = enforced_code
                if child_code == parent.code:
                    return make_result(error="All edits were outside the EVOLVE-BLOCK regions")

        # Check code length
        if len(child_code) > _worker_config.max_code_length:
            return make_result(
                error=f"Generated code exceeds maximum length ({len(child_code)} > {_worker_config.max_code_length})",
            )

        # Evaluate the child program
        child_id = f"candidate-{iteration}"
        # Measure all cascade stages and retries, including failed evaluation calls.
        evaluation_started = time.perf_counter()
        try:
            child_metrics = asyncio.run(_worker_evaluator.evaluate_program(child_code, child_id))
        finally:
            evaluation_time_seconds = time.perf_counter() - evaluation_started

        # Get artifacts
        artifacts = _worker_evaluator.get_pending_artifacts(child_id)

        # Create child program
        child_program = Program(
            id=child_id,
            code=child_code,
            changes_description=child_changes_desc,
            language=_worker_config.language,
            parent_id=parent.id,
            generation=parent.generation + 1,
            metrics=child_metrics,
            iteration_found=iteration,
            metadata={
                "changes": changes_summary,
                "parent_metrics": parent.metrics,
                "island": parent_island,
                "token_usage": token_usage,
            },
        )

        iteration_time = time.time() - iteration_start

        return make_result(
            child_program_dict=child_program.to_dict(),
            iteration_time=iteration_time,
            artifacts=artifacts,
        )

    except Exception as e:
        logger.exception(f"Error in worker iteration {iteration}")
        return make_result(error=str(e))


def _wait_for_processes(processes: tuple[mp.Process, ...], timeout: float) -> list[mp.Process]:
    """Wait for process handles to observe worker exits without blocking indefinitely."""
    deadline = time.monotonic() + timeout
    alive = list(processes)
    while alive:
        next_alive = []
        for process in alive:
            try:
                process.join(timeout=0)
                if process.is_alive():
                    next_alive.append(process)
            except (AssertionError, ValueError):
                continue
        alive = next_alive
        remaining = deadline - time.monotonic()
        if not alive or remaining <= 0:
            break
        time.sleep(min(0.001, remaining))
    return alive


def _terminate_process_pool(executor: ProcessPoolExecutor) -> None:
    """Cancel queued work and ensure all process-pool workers have exited."""
    # Python < 3.14 has no public force-shutdown API. Capture only this
    # executor's workers before shutdown clears its private process mapping.
    process_map = getattr(executor, "_processes", None) or {}
    processes = tuple(process_map.copy().values())
    terminate_workers = getattr(executor, "terminate_workers", None)

    if callable(terminate_workers):
        terminate_workers()
    else:
        executor.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            try:
                if process.is_alive():
                    process.terminate()
            except (ProcessLookupError, ValueError):
                continue

    surviving_processes = _wait_for_processes(processes, timeout=1.0)
    for process in surviving_processes:
        try:
            process.kill()
        except (ProcessLookupError, ValueError):
            continue

    surviving_processes = _wait_for_processes(tuple(surviving_processes), timeout=1.0)
    if surviving_processes:
        logger.warning(
            "Process-pool workers did not exit: %s",
            [process.pid for process in surviving_processes],
        )


class ProcessParallelController:
    """Controller for process-based parallel evolution"""

    def __init__(
        self,
        config: Config,
        evaluation_file: str,
        database: ProgramDatabase,
        evolution_tracer=None,
        file_suffix: str = ".py",
        island_selector: Optional[IslandSelector] = None,
    ):
        self.config = config
        self.evaluation_file = evaluation_file
        self.database = database
        self.evolution_tracer = evolution_tracer
        self.file_suffix = file_suffix
        self.island_selector = island_selector
        self.request_observer = None
        self.executor: Optional[ProcessPoolExecutor] = None
        self.shutdown_event = mp.Event()
        self.early_stopping_triggered = False

        # Number of worker processes
        self.num_workers = config.evaluator.parallel_evaluations
        self.num_islands = config.database.num_islands

        logger.info(f"Initialized process parallel controller with {self.num_workers} workers")

    def _serialize_config(self, config: Config) -> dict:
        """Serialize config object to a dictionary that can be pickled"""
        # Manual serialization to handle nested objects properly

        # The asdict() call itself triggers the deepcopy which tries to serialize novelty_llm. Remove it first.
        config.database.novelty_llm = None

        return {
            "llm": {
                "models": [asdict(m) for m in config.llm.models],
                "evaluator_models": [asdict(m) for m in config.llm.evaluator_models],
                "api_base": config.llm.api_base,
                "api_key": config.llm.api_key,
                "temperature": config.llm.temperature,
                "top_p": config.llm.top_p,
                "max_tokens": config.llm.max_tokens,
                "timeout": config.llm.timeout,
                "retries": config.llm.retries,
                "retry_delay": config.llm.retry_delay,
            },
            "prompt": asdict(config.prompt),
            "database": asdict(config.database),
            "evaluator": asdict(config.evaluator),
            "max_iterations": config.max_iterations,
            "checkpoint_interval": config.checkpoint_interval,
            "log_level": config.log_level,
            "log_dir": config.log_dir,
            "random_seed": config.random_seed,
            "diff_based_evolution": config.diff_based_evolution,
            "max_code_length": config.max_code_length,
            "language": config.language,
            "file_suffix": self.file_suffix,
        }

    def start(self) -> None:
        """Start the process pool"""
        # Convert config to dict for pickling
        # We need to be careful with nested dataclasses
        config_dict = self._serialize_config(self.config)

        # Pass current environment to worker processes
        import os
        import sys

        current_env = dict(os.environ)

        executor_kwargs = {
            "max_workers": self.num_workers,
            "initializer": _worker_init,
            "initargs": (config_dict, self.evaluation_file, current_env),
        }
        if sys.version_info >= (3, 11):
            logger.info(f"Set max {self.config.max_tasks_per_child} tasks per child")
            executor_kwargs["max_tasks_per_child"] = self.config.max_tasks_per_child
        elif self.config.max_tasks_per_child is not None:
            logger.warn(
                "max_tasks_per_child is only supported in Python 3.11+. "
                "Ignoring max_tasks_per_child and using spawn start method."
            )
            executor_kwargs["mp_context"] = mp.get_context("spawn")

        # Create process pool with initializer
        self.executor = ProcessPoolExecutor(**executor_kwargs)
        logger.info(f"Started process pool with {self.num_workers} processes")

    def stop(self) -> None:
        """Stop the process pool"""
        self.shutdown_event.set()

        executor = self.executor
        self.executor = None
        if executor:
            _terminate_process_pool(executor)

        logger.info("Stopped process pool")

    def request_shutdown(self) -> None:
        """Request graceful shutdown"""
        logger.info("Graceful shutdown requested...")
        self.shutdown_event.set()

    def _create_database_snapshot(self) -> Dict[str, Any]:
        """Create a serializable snapshot of the database state"""
        # Only include necessary data for workers
        snapshot = {
            "programs": {pid: prog.to_dict() for pid, prog in self.database.programs.items()},
            "islands": [self.database._ordered_ids(island) for island in self.database.islands],
            "current_island": self.database.current_island,
            "feature_dimensions": self.database.config.feature_dimensions,
            "artifacts": {},  # Will be populated selectively
        }

        # Include artifacts for programs that might be selected
        # This limits artifacts (execution outputs/errors) to avoid large snapshot sizes.
        # This does NOT affect program code - all programs are fully serialized above.
        # With max_artifact_bytes=20KB and population_size=1000, artifacts could be 20MB total,
        # which would significantly slow worker process initialization. The default limit of 100
        # keeps artifact data under 2MB while still providing execution context for recent programs.
        # Workers can still evolve properly as they have access to ALL program code.
        # Configure via database.max_snapshot_artifacts (None for unlimited).
        max_artifacts = self.database.config.max_snapshot_artifacts
        program_ids = list(self.database.programs.keys())
        if max_artifacts is not None:
            program_ids = program_ids[:max_artifacts]
        for pid in program_ids:
            artifacts = self.database.get_artifacts(pid)
            if artifacts:
                snapshot["artifacts"][pid] = artifacts

        return snapshot

    def _select_island(self, iteration: int, island_pending: Dict[int, List[int]]) -> int:
        """Ask a custom policy for an island using a read-only state snapshot."""
        stats = self.database.get_island_stats()
        context = IslandSelectionContext(
            iteration=iteration,
            pending_counts=tuple(len(island_pending[i]) for i in range(self.num_islands)),
            islands=tuple(
                IslandState(
                    population_size=stat["population_size"],
                    best_score=stat["best_score"],
                    average_score=stat["average_score"],
                    diversity=stat["diversity"],
                    generation=stat["generation"],
                )
                for stat in stats
            ),
        )
        island_id = self.island_selector(context)
        if (
            isinstance(island_id, bool)
            or not isinstance(island_id, Integral)
            or not 0 <= island_id < self.num_islands
        ):
            raise ValueError(
                f"island_selector must return an integer island ID in [0, {self.num_islands - 1}], "
                f"got {island_id!r}"
            )
        return int(island_id)

    def _checkpoint_if_due(self, iteration: int, callback=None) -> None:
        if iteration > 0 and iteration % self.config.checkpoint_interval == 0:
            logger.info("Checkpoint interval reached at iteration %d", iteration)
            self.database.log_island_status()
            if callback:
                callback(iteration)

    def process_result(self, result: SerializableResult, iteration: int, checkpoint_callback=None) -> Program:
        """Commit a worker result using native database admission and checkpoint rules."""
        token_str = ""
        if result.token_usage:
            pt = result.token_usage.get("prompt_tokens", 0)
            ct = result.token_usage.get("completion_tokens", 0)
            tt = result.token_usage.get("total_tokens", 0)
            token_str = f" | tokens: {tt} (prompt: {pt}, completion: {ct})"
        # Reconstruct program from dict
        child_program = Program(**result.child_program_dict)

        # Add to database with explicit target_island to ensure proper island placement
        # This fixes issue #391: children should go to the target island, not inherit
        # from the parent (which may be from a different island due to fallback sampling)
        admitted_id = self.database.add(
            child_program,
            iteration=iteration,
            target_island=result.target_island,
        )
        if admitted_id is None:
            raise _CandidateRejected(child_program.id)

        result.admitted = True
        result.admitted_at = time.time()
        result.child_program_dict = child_program.to_dict()

        # Store artifacts
        if result.artifacts:
            self.database.store_artifacts(child_program.id, result.artifacts)

        # Log evolution trace
        if self.evolution_tracer:
            # Retrieve parent program for trace logging
            parent_program = (
                self.database.get(result.parent_id) if result.parent_id else None
            )
            if parent_program:
                # Determine island ID
                island_id = child_program.metadata.get(
                    "island", self.database.current_island
                )

                self.evolution_tracer.log_trace(
                    iteration=iteration,
                    parent_program=parent_program,
                    child_program=child_program,
                    prompt=result.prompt,
                    llm_response=result.llm_response,
                    artifacts=result.artifacts,
                    island_id=island_id,
                    metadata={
                        "iteration_time": result.iteration_time,
                        "changes": child_program.metadata.get("changes", ""),
                        "token_usage": result.token_usage,
                    },
                )

        # Log prompts
        if result.prompt:
            self.database.log_prompt(
                template_key=(
                    "full_rewrite_user"
                    if not self.config.diff_based_evolution
                    else "diff_user"
                ),
                program_id=child_program.id,
                prompt=result.prompt,
                responses=[result.llm_response] if result.llm_response else [],
                token_usage=result.token_usage,
            )

        # Island management
        # get current program island id
        island_id = child_program.metadata.get("island", self.database.current_island)
        # use this to increment island generation
        self.database.increment_island_generation(island_idx=island_id)

        # Check migration
        if self.database.should_migrate():
            logger.info(f"Performing migration at iteration {iteration}")
            self.database.migrate_programs()
            self.database.log_island_status()

        # Log progress
        logger.info(
            f"Iteration {iteration}: "
            f"Program {child_program.id} "
            f"(parent: {result.parent_id}) "
            f"completed in {result.iteration_time:.2f}s{token_str}"
        )

        if child_program.metrics:
            metrics_str = ", ".join(
                [
                    f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                    for k, v in child_program.metrics.items()
                ]
            )
            logger.info(f"Metrics: {metrics_str}")

            # Check if this is the first program without combined_score
            if not hasattr(self, "_warned_about_combined_score"):
                self._warned_about_combined_score = False

            if (
                "combined_score" not in child_program.metrics
                and not self._warned_about_combined_score
            ):
                avg_score = safe_numeric_average(child_program.metrics)
                logger.warning(
                    f"⚠️  No 'combined_score' metric found in evaluation results. "
                    f"Using average of all numeric metrics ({avg_score:.4f}) for evolution guidance. "
                    f"For better evolution results, please modify your evaluator to return a 'combined_score' "
                    f"metric that properly weights different aspects of program performance."
                )
                self._warned_about_combined_score = True

        # Check for new best
        if self.database.best_program_id == child_program.id:
            logger.info(
                f"🌟 New best solution found at iteration {iteration}: "
                f"{child_program.id}"
            )

        result.child_program_dict = child_program.to_dict()
        self._checkpoint_if_due(iteration, checkpoint_callback)
        return child_program

    def _prepare_request(self, generation_number: int, target_island: int) -> PreparedRequest:
        parent, inspirations = self.database.sample_from_island(
            island_id=target_island,
            num_inspirations=self.config.prompt.num_diverse_programs,
        )
        snapshot = self._create_database_snapshot()
        snapshot["sampling_island"] = target_island
        snapshot["prompt_selection"] = default_prompt_context(snapshot, parent.id, self.config.prompt)
        return PreparedRequest(generation_number, target_island, parent.id,
                               pickle.dumps((snapshot, [p.id for p in inspirations])))

    async def _execute_request(self, request: PreparedRequest, checkpoint_callback=None) -> SerializableResult:
        """Execute a prepared request and apply normal database processing."""
        if self.shutdown_event.is_set():
            raise asyncio.CancelledError("Evolution shutdown requested")
        if self.executor is None:
            raise RuntimeError("The evolution worker pool is not running")
        generation_number, snapshot, parent_id, inspirations = request.worker_inputs()
        target_island = request.target_island
        result = SerializableResult(iteration=generation_number, parent_id=parent_id,
                                    target_island=target_island)
        try:
            future = self.executor.submit(
                _run_iteration_worker, generation_number, snapshot, parent_id, list(inspirations)
            )
            pending = asyncio.wrap_future(future)
            try:
                while not pending.done() and not self.shutdown_event.is_set():
                    await asyncio.wait([pending], timeout=0.1)
                if self.shutdown_event.is_set():
                    raise asyncio.CancelledError("Evolution shutdown requested")
                result = await pending
            finally:
                pending.cancel()
            if not result.error and result.child_program_dict:
                self.process_result(result, generation_number, checkpoint_callback)
        except Exception as exc:
            if getattr(exc, "_openevolve_strategy_error", False):
                raise
            result.error = f"{type(exc).__name__}: {exc}"
        self.database.last_iteration = max(self.database.last_iteration, generation_number)
        return result

    async def run_evolution(
        self,
        start_iteration: int,
        max_iterations: int,
        target_score: Optional[float] = None,
        checkpoint_callback=None,
    ):
        """Run evolution with process-based parallelism"""
        if not self.executor:
            raise RuntimeError("Process pool not started")
        total_iterations = start_iteration + max_iterations

        logger.info(
            f"Starting process-based evolution from iteration {start_iteration} "
            f"for {max_iterations} iterations (total: {total_iterations})"
        )

        # Track pending futures by island to maintain distribution
        pending_futures: Dict[int, Future] = {}
        island_pending: Dict[int, List[int]] = {i: [] for i in range(self.num_islands)}
        batch_size = min(self.num_workers * 2, max_iterations)

        # Submit initial batch - distribute across islands
        batch_per_island = max(1, batch_size // self.num_islands) if batch_size > 0 else 0
        current_iteration = start_iteration

        if self.island_selector is None:
            # Preserve the original round-robin distribution by default.
            for island_id in range(self.num_islands):
                for _ in range(batch_per_island):
                    if current_iteration < total_iterations:
                        future = self._submit_iteration(current_iteration, island_id)
                        if future:
                            pending_futures[current_iteration] = future
                            island_pending[island_id].append(current_iteration)
                        current_iteration += 1
        else:
            # Keep the original initial batch size, but let the policy place each task.
            initial_slots = min(max_iterations, batch_per_island * self.num_islands)
            for _ in range(initial_slots):
                island_id = self._select_island(current_iteration, island_pending)
                future = self._submit_iteration(current_iteration, island_id)
                if future:
                    pending_futures[current_iteration] = future
                    island_pending[island_id].append(current_iteration)
                current_iteration += 1

        next_iteration = current_iteration
        completed_iterations = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_llm_calls = 0

        # Early stopping tracking
        early_stopping_enabled = self.config.early_stopping_patience is not None
        if early_stopping_enabled:
            best_score = float("-inf")
            iterations_without_improvement = 0
            if self.config.early_stopping_patience < 0:
                logger.info(
                    f"Early stopping patience is set to a negative value, running event-based early-stopping, "
                    f"Early stop when metric '{self.config.early_stopping_metric}' reaches {self.config.convergence_threshold}"
                )
            else:
                logger.info(
                    f"Early stopping enabled: patience={self.config.early_stopping_patience}, "
                    f"threshold={self.config.convergence_threshold}, "
                    f"metric={self.config.early_stopping_metric}"
                )
        else:
            logger.info("Early stopping disabled")

        # Process results as they complete
        while (
            pending_futures
            and completed_iterations < max_iterations
            and not self.shutdown_event.is_set()
        ):
            completed_iteration = None
            for iteration, future in list(pending_futures.items()):
                if future.done():
                    completed_iteration = iteration
                    break
            if completed_iteration is None:
                await asyncio.sleep(0.01)
                continue

            # Process completed result
            future = pending_futures.pop(completed_iteration)
            result = None

            try:
                # Use evaluator timeout + buffer to gracefully handle stuck processes
                timeout_seconds = self.config.evaluator.timeout + 30
                result = future.result(timeout=timeout_seconds)

                token_str = ""
                if result.token_usage:
                    pt = result.token_usage.get("prompt_tokens", 0)
                    ct = result.token_usage.get("completion_tokens", 0)
                    tt = result.token_usage.get("total_tokens", 0)
                    total_prompt_tokens += pt
                    total_completion_tokens += ct
                    total_tokens += tt
                    total_llm_calls += 1
                    token_str = f" | tokens: {tt} (prompt: {pt}, completion: {ct})"

                if result.error:
                    logger.warning(
                        f"Iteration {completed_iteration} error: {result.error}{token_str}"
                    )
                elif result.child_program_dict:
                    child_program = self.process_result(result, completed_iteration, checkpoint_callback)

                    # Check target score
                    if target_score is not None and child_program.metrics:
                        if (
                            "combined_score" in child_program.metrics
                            and child_program.metrics["combined_score"] >= target_score
                        ):
                            logger.info(
                                f"Target score {target_score} reached at iteration {completed_iteration}"
                            )
                            break

                    # Check early stopping
                    if early_stopping_enabled and child_program.metrics:
                        # Get the metric to track for early stopping
                        current_score = None
                        if self.config.early_stopping_metric in child_program.metrics:
                            current_score = child_program.metrics[self.config.early_stopping_metric]
                        elif self.config.early_stopping_metric == "combined_score":
                            # Default metric not found, use safe average (standard pattern)
                            current_score = safe_numeric_average(child_program.metrics)
                        else:
                            # User specified a custom metric that doesn't exist
                            logger.warning(
                                f"Early stopping metric '{self.config.early_stopping_metric}' not found, using safe numeric average"
                            )
                            current_score = safe_numeric_average(child_program.metrics)

                        if current_score is not None and isinstance(current_score, (int, float)):
                            # Check for improvement
                            if self.config.early_stopping_patience > 0:
                                improvement = current_score - best_score
                                if improvement >= self.config.convergence_threshold:
                                    best_score = current_score
                                    iterations_without_improvement = 0
                                    logger.debug(
                                        f"New best score: {best_score:.4f} (improvement: {improvement:+.4f})"
                                    )
                                else:
                                    iterations_without_improvement += 1
                                    logger.debug(
                                        f"No improvement: {iterations_without_improvement}/{self.config.early_stopping_patience}"
                                    )

                                # Check if we should stop
                                if (
                                    iterations_without_improvement
                                    >= self.config.early_stopping_patience
                                ):
                                    self.early_stopping_triggered = True
                                    logger.info(
                                        f"🛑 Early stopping triggered at iteration {completed_iteration}: "
                                        f"No improvement for {iterations_without_improvement} iterations "
                                        f"(best score: {best_score:.4f})"
                                    )
                                    break

                            else:
                                # Event-based early stopping
                                if current_score == self.config.convergence_threshold:
                                    best_score = current_score
                                    logger.info(
                                        f"🛑 Early stopping (event-based) triggered at iteration {completed_iteration}: "
                                        f"Task successfully solved with score {best_score:.4f}."
                                    )
                                    self.early_stopping_triggered = True
                                    break

            except _CandidateRejected as rejected:
                logger.info("Admission strategy rejected program %s", rejected)
                try:
                    self._checkpoint_if_due(completed_iteration, checkpoint_callback)
                except Exception as e:
                    if getattr(e, "_openevolve_strategy_error", False):
                        raise
                    logger.error(
                        f"Error processing result from iteration {completed_iteration}: {e}"
                    )
            except FutureTimeoutError as e:
                if getattr(e, "_openevolve_strategy_error", False):
                    raise
                logger.error(
                    f"⏰ Iteration {completed_iteration} timed out after {timeout_seconds}s "
                    f"(evaluator timeout: {self.config.evaluator.timeout}s + 30s buffer). "
                    f"Canceling future and continuing with next iteration."
                )
                # Cancel the future to clean up the process
                future.cancel()
            except Exception as e:
                if getattr(e, "_openevolve_strategy_error", False):
                    raise
                logger.error(f"Error processing result from iteration {completed_iteration}: {e}")

            finally:
                self._request_event(
                    'processed', creation_id=completed_iteration,
                    admitted=bool(result and result.admitted),
                    admitted_at=result.admitted_at if result else None,
                    metrics=result.metrics if result else {},
                    error=result.error if result else 'Worker result unavailable',
                    generation_wall_time_seconds=result.generation_wall_time_seconds if result else None,
                    evaluation_time_seconds=result.evaluation_time_seconds if result else None)

            completed_iterations += 1

            # Remove completed iteration from island tracking
            for island_id, iteration_list in island_pending.items():
                if completed_iteration in iteration_list:
                    iteration_list.remove(completed_iteration)
                    break

            # Submit one replacement task per completion. Custom policies control
            # placement; the default path retains the original balancing logic.
            if self.island_selector is not None:
                if next_iteration < total_iterations and not self.shutdown_event.is_set():
                    island_id = self._select_island(next_iteration, island_pending)
                    future = self._submit_iteration(next_iteration, island_id)
                    if future:
                        pending_futures[next_iteration] = future
                        island_pending[island_id].append(next_iteration)
                        next_iteration += 1
            else:
                for island_id in range(self.num_islands):
                    if (
                        len(island_pending[island_id]) < batch_per_island
                        and next_iteration < total_iterations
                        and not self.shutdown_event.is_set()
                    ):
                        future = self._submit_iteration(next_iteration, island_id)
                        if future:
                            pending_futures[next_iteration] = future
                            island_pending[island_id].append(next_iteration)
                            next_iteration += 1
                            break  # Only submit one iteration per completion to maintain balance

        # Handle shutdown
        if self.shutdown_event.is_set():
            logger.info("Shutdown requested, canceling remaining evaluations...")
            for future in pending_futures.values():
                future.cancel()

        # Log completion reason
        if self.early_stopping_triggered:
            logger.info("✅ Evolution completed - Early stopping triggered due to convergence")
        elif self.shutdown_event.is_set():
            logger.info("✅ Evolution completed - Shutdown requested")
        else:
            logger.info("✅ Evolution completed - Maximum iterations reached")

        if total_llm_calls > 0:
            logger.info(
                f"📊 Total LLM Token Usage ({total_llm_calls} calls): "
                f"total_tokens={total_tokens} "
                f"(prompt_tokens={total_prompt_tokens}, completion_tokens={total_completion_tokens})"
            )

        return self.database.get_best_program()

    def _request_event(self, event, **fields):
        if self.request_observer is not None:
            self.request_observer(event, **fields)

    def _submit_iteration(
        self, iteration: int, island_id: Optional[int] = None
    ) -> Optional[Future]:
        """Submit an iteration to the process pool, optionally pinned to a specific island"""
        try:
            target_island = island_id if island_id is not None else self.database.current_island
            request = self._prepare_request(iteration, target_island)
            self._request_event('prepared', creation_id=iteration, target_island=target_island,
                                parent_id=request.parent_id)
            self._request_event('submitted', creation_id=iteration)
            return self.executor.submit(_run_iteration_worker, *request.worker_inputs())

        except Exception as e:
            logger.error(f"Error submitting iteration {iteration}: {e}")
            return None
