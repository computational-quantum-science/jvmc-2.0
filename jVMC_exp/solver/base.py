from abc import ABC, abstractmethod

class AbstractSolver(ABC):
    @property
    @abstractmethod
    def _needs_dense_matrix(self) -> bool:
        pass

    @abstractmethod
    def __call__(self, A, b, **kwargs):
        pass