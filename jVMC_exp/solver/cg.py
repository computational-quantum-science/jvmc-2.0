from .base import AbstractSolver, SolverState
from jax.scipy.sparse.linalg import cg

class CG(AbstractSolver):
    def __init__(self, tol: float = 1e-5, maxiter: int | None = None):
        self._tol = tol
        self._maxiter = maxiter

    @property
    def tol(self):
        return self._tol
    
    @property
    def maxiter(self):
        return self._maxiter

    def __call__(self, A, b, solver_state: SolverState):
        # The current jax implementaiton of cg does not return any info
        x, info = cg(A, b, tol=self.tol, maxiter=self.maxiter) 

        return x, {}

    @property
    def _needs_dense_matrix(self):
        return False

