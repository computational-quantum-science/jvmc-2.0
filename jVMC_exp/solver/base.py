from abc import ABC, abstractmethod
from chex import dataclass
from typing import Callable

from jVMC_exp.stats import SampledObs

class AbstractSolver(ABC):
    @property
    @abstractmethod
    def _needs_dense_matrix(self) -> bool:
        pass

    @abstractmethod
    def __call__(self, A, b, **kwargs):
        pass