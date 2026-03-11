from abc import ABC, abstractmethod
from jVMC_exp.vqs import NQS

class AbstractOperator(ABC):
    @abstractmethod
    def get_O_loc(self, s, psi: NQS, *, logPsiS=None, **kwargs):
        pass