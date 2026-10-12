"""Choose prepared requests before they enter the worker pool."""
from concurrent.futures import Future
from collections import deque
from threading import RLock
import time
from typing import Protocol, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from openevolve.process_parallel import PreparedRequest


class RequestDispatcher(Protocol):
    def choose(self, waiting_requests: Sequence['PreparedRequest'], *,
               turn: int, free_workers: int) -> int | None: ...


class DispatchError(RuntimeError):
    """A policy requested an invalid or impossible assignment."""
    _openevolve_dispatch_error = True


class _RequestQueue:
    """Own waiting requests and their results; completion releases worker capacity."""
    def __init__(self, executor, workers, policy, worker, shutdown, observe):
        self.executor, self.workers, self.policy = executor, workers, policy
        self.worker, self.shutdown, self.observe = worker, shutdown, observe
        self.waiting, self.running = {}, {}
        self.completed = deque()
        self.turn = 1
        self.lock = RLock()
        self.closed = self.filling = False
        self.error = None
        self.deferred_since = None

    def add(self, request):
        with self.lock:
            self.check()
            if self.closed or self.shutdown.is_set():
                raise DispatchError('Cannot queue a request after shutdown')
            identity = request.request_id
            if identity in self.waiting or identity in self.running:
                raise DispatchError(f'Duplicate waiting request: {identity}')
            result = Future()
            self.waiting[identity] = (request, result)
            result.add_done_callback(lambda future: self._cancel(identity) if future.cancelled() else None)
            self._fill()
            self.check()
            return result

    def _cancel(self, identity):
        with self.lock:
            if self.waiting.pop(identity, None) is not None:
                self.completed.append(identity)
                self._fill()
            entry = self.running.get(identity)
            if entry and entry[0] is not None:
                entry[0].cancel()

    def _fill(self):
        if self.filling or self.closed or self.error or self.shutdown.is_set():
            return
        self.filling = True
        try:
            while self.waiting and len(self.running) < self.workers:
                requests = tuple(entry[0] for entry in self.waiting.values())
                free = self.workers - len(self.running)
                identity = self.policy.choose(requests, turn=self.turn, free_workers=free)
                if identity is None:
                    if self.deferred_since is None:
                        self.deferred_since = time.perf_counter()
                    self.observe('deferred', turn=self.turn, queue=list(self.waiting), free_workers=free)
                    return
                if type(identity) is not int or identity not in self.waiting:
                    raise DispatchError(f'Turn {self.turn}: policy chose unavailable request {identity!r}')
                queue = list(self.waiting)
                request, result = self.waiting.pop(identity)
                delay = (time.perf_counter() - self.deferred_since
                         if self.deferred_since is not None else 0.0)
                self.deferred_since = None
                assigned_at = time.time()
                self.running[identity] = (None, result)
                try:
                    execution = self.executor.submit(self.worker, *request.worker_inputs())
                except Exception as exc:
                    self.running.pop(identity)
                    self.completed.append(identity)
                    result.set_exception(exc)
                    raise DispatchError(f'Could not submit selected request {identity}: {exc}') from exc
                self.running[identity] = (execution, result)
                self.observe('assigned', time=assigned_at, creation_id=identity, dispatch_number=self.turn,
                             queue=queue, queue_position=queue.index(identity)+1,
                             free_workers=free, scheduling_delay_seconds=delay)
                self.turn += 1
                execution.add_done_callback(lambda future, rid=identity: self._finished(rid, future))
        except Exception as exc:
            self.error = exc if isinstance(exc, DispatchError) else DispatchError(f'Ordering policy failed: {exc}')
        finally:
            self.filling = False

    def _finished(self, identity, execution):
        with self.lock:
            entry = self.running.pop(identity, None)
            if entry is None:
                return
            result = entry[1]
            self.completed.append(identity)
            # Ordinary requests start before the parent processes this result.
            self._fill()
            if not result.done():
                if execution.cancelled():
                    result.cancel()
                else:
                    try:
                        result.set_result(execution.result())
                    except BaseException as exc:
                        result.set_exception(exc)

    def pop_completed(self):
        with self.lock:
            self.check()
            return self.completed.popleft() if self.completed else None

    def check(self):
        if self.error is not None:
            raise self.error

    def ensure_progress(self):
        with self.lock:
            self.check()
            if self.waiting and not self.running and not self.completed and not self.shutdown.is_set():
                raise DispatchError(
                    f'Turn {self.turn}: requested position cannot become available with '
                    f'the remaining budget and normal refill; waiting requests: {list(self.waiting)}')

    def close(self):
        with self.lock:
            self.closed = True
            for request, result in list(self.waiting.values()):
                result.cancel()
            for execution, result in list(self.running.values()):
                result.cancel()
                if execution is not None:
                    execution.cancel()
            self.waiting.clear()
