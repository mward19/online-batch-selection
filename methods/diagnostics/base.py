"""Diagnostics framework.

A `Diagnostic` computes one thing about the current training state. Diagnostics
form a dependency DAG: a diagnostic's `_run` calls `dep.run()` on each of its
dependencies, and `run` caches by `TrainState` so shared work (a forward
pass, a kernel) is computed once per state and reused. A `DiagnosticsManager`
holds the top-level (logged) diagnostics for one training phase, drives them,
and owns the deduplication table so shared dependency instances are built once.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional


class Phase(Enum):
    """Where in the training iteration a state was captured. Part of the cache
    key so states at the same global step but different points (the weights
    change between them) don't collide."""
    POST_BATCH = "post_batch"
    EPOCH_END = "epoch_end"
    TRAIN_END = "train_end"


class LogType(Enum):
    """How a diagnostic's payload is routed to W&B."""
    RUN = "run"        # wandb.log(payload, step=...)
    SUMMARY = "summary"  # wandb.run.summary.update(payload)
    FILEONLY = "file only" # wandb does not log at all


@dataclass(frozen=True)
class TrainState:
    """When/where we are. Equality (the diagnostic cache key) is `(total_steps,
    phase)` only. Model/optimizer state changes exactly once per optimizer
    step, so equal `total_steps` implies equal weights; `phase` separates points
    within a step. The remaining fields are kept for readable logs but excluded
    from equality."""
    total_steps: int
    phase: Phase
    epoch: int = field(default=0, compare=False)
    batch_idx: int = field(default=0, compare=False)
    total_epochs: Optional[int] = field(default=None, compare=False)
    total_batches: Optional[int] = field(default=None, compare=False)


@dataclass
class DiagnosticInfo:
    """The output of a diagnostic. `info` is the value (scalar, dict, tensor,
    …); `metadata` optionally hints at routing (units, scalar vs histogram)."""
    name: str
    info: Any
    metadata: Optional[dict] = None
    log_type: LogType = LogType.RUN


class Diagnostic:
    """Base class. Subclasses implement `_run` (and `__eq__` if they may be
    deduplicated). A diagnostic is *not* auto-registered with its manager;
    `create_diagnostics` registers only the top-level (logged) ones, so
    dependency-only diagnostics simply hold a manager ref for state/context."""

    # Read-only properties. Diagnostics must not modify the selection method.
    # Technically it is still possible to modify self._method, but not
    # self.method
    @property
    def method(self):
        return self._method

    def __init__(self, manager, log_path: Optional[str] = None,
                 should_run: Optional[Callable[[TrainState], bool]] = None):
        self.manager = manager
        self._method = manager.method
        self.project_root = manager.project_root
        self.log_path = log_path
        self.should_run = should_run if should_run is not None else (lambda state: True)
        self.last_run_state: Optional[TrainState] = None
        self.last_run_diagnostic: Optional[DiagnosticInfo] = None

    def get_state(self) -> TrainState:
        return self.manager.current_state

    def _run(self) -> DiagnosticInfo:
        raise NotImplementedError

    def run(self) -> DiagnosticInfo:
        """Compute, or return the cached result if the state is unchanged."""
        state = self.get_state()
        if state is not None and state == self.last_run_state:
            return self.last_run_diagnostic
        self.last_run_diagnostic = self._run()
        self.last_run_state = state
        return self.last_run_diagnostic

    def conditional_run(self) -> bool:
        """Run iff `should_run` accepts the current state. Returns whether it
        ran (so the manager only logs diagnostics that fired this state)."""
        if self.should_run(self.get_state()):
            self.run()
            return True
        return False

    def _log_payload(self):
        """Map `(last_run_diagnostic, last_run_state)` to a flat dict of
        scalars for logging. Override for non-scalar outputs."""
        info = self.last_run_diagnostic.info
        if isinstance(info, dict):
            return dict(info)
        return {self.last_run_diagnostic.name: info}

    def wandb_log(self, infos: List[DiagnosticInfo]):
        import wandb
        payload = self._log_payload()
        if self.last_run_diagnostic.log_type == LogType.SUMMARY:
            wandb.run.summary.update(payload)
        elif self.last_run_diagnostic.log_type == LogType.RUN:
            wandb.log(payload, step=int(self.last_run_state.total_steps))
        else:
            pass # Do not log to W&B

    def file_log(self, infos: List[DiagnosticInfo]):
        if self.log_path is None:
            return
        payload = self._log_payload()
        with open(self.log_path, "a") as f:
            f.write(f"step={self.last_run_state.total_steps} {payload}\n")

    def log(self):
        self.wandb_log([self.last_run_diagnostic])
        self.file_log([self.last_run_diagnostic])

    def __eq__(self, other):
        raise NotImplementedError(
            f"{type(self).__name__} does not implement __eq__; override it if this "
            "diagnostic may be deduplicated by DiagnosticsManager.build."
        )

    __hash__ = None


class DiagnosticsManager:
    """Drives the top-level diagnostics for one training phase and owns the
    deduplication table so shared dependency instances are built once."""

    def __init__(self, method=None, project_root=None, should_run: bool = True):
        self.diagnostics: List[Diagnostic] = []
        self.current_state: Optional[TrainState] = None
        self.should_run = should_run
        self.method = method
        self.project_root = project_root
        self._all_diagnostics: dict = defaultdict(list)

    def build(self, diagnostic_class, *args, **kwargs) -> "Diagnostic":
        """Return a shared instance if an equal diagnostic already exists,
        otherwise construct, cache, and return a new one."""
        new_diagnostic = diagnostic_class(*args, **kwargs)
        matches = [x for x in self._all_diagnostics[diagnostic_class] if x == new_diagnostic]
        if len(matches) > 1:
            raise ValueError(f"Multiple identical diagnostics of type {diagnostic_class.__name__}")
        if matches:
            return matches[0]
        self._all_diagnostics[diagnostic_class].append(new_diagnostic)
        return new_diagnostic

    def register(self, diagnostic: Diagnostic):
        self.diagnostics.append(diagnostic)
        return diagnostic

    def _update_state(self, state: TrainState):
        self.current_state = state

    def run_diagnostics(self, state: TrainState):
        if not self.should_run:
            return
        self._update_state(state)
        for diagnostic in self.diagnostics:
            diagnostic.conditional_run()
        self._log_diagnostics()

    def _log_diagnostics(self):
        for diagnostic in self.diagnostics:
            if diagnostic.last_run_state == self.current_state:
                diagnostic.log()
