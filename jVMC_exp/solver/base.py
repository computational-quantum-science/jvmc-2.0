from abc import ABC, abstractmethod
from chex import dataclass
from typing import Callable

from jVMC_exp.stats import SampledObs

@dataclass
class SolverState():
    covar_grad_o_loc: Callable[[], SampledObs] | None = None
    rhs_trans_fn: Callable | None = None
    exact_sampler: bool = False

class AbstractSolver(ABC):
    @property
    @abstractmethod
    def _needs_dense_matrix(self) -> bool:
        pass

    @abstractmethod
    def __call__(self, A, b, solver_state: SolverState):
        pass