import jax.numpy as jnp
import warnings
import numpy as np
import jax

from jVMC_exp.solver.base import SolverState, AbstractSolver

def _eigh_numpy(S):
    e, V = np.linalg.eigh(np.array(S))

    return jnp.array(e), jnp.array(V)

@jax.jit(static_argnums=(2,))
def get_snr(VtF, rho_var, num_samples):
    return jnp.sqrt(jnp.abs(num_samples * (jnp.conj(VtF) * VtF) / (rho_var + 1e-10))).ravel()
    
class PinvSNR(AbstractSolver):
    def __init__(self, snr_tol=2, pinv_tol=1e-14, pinv_cutoff=1e-8, diagonalize_on_device=True):
        self._snr_tol = snr_tol
        self._pinv_tol = pinv_tol
        self.pinv_cutoff = pinv_cutoff
        self._diagonalize_on_device = diagonalize_on_device

        self._ev = None
        self._V = None
    
    @property
    def snr_tol(self):
        return self._snr_tol
    
    @property
    def pinv_tol(self):
        return self._pinv_tol
    
    @property
    def last_eigenvalues(self):
        return self._ev
    
    @property
    def last_eigenvectors(self):
        return self._V
    
    @property
    def _needs_dense_matrix(self) -> bool:
        return True
    
    def __call__(self, S, F, solver_state: SolverState):
        # Transform equation to eigenbasis and compute Signal to Noise Ratio
        self._transform_to_eigenbasis(S, F)
        rho = solver_state.covar_grad_o_loc().transform(solver_state.rhs_trans_fn, jnp.transpose(self.last_eigenvectors))
        snr = get_snr(self._VtF, rho.var.ravel(), rho._num_samples)

        # Discard eigenvalues below numerical precision
        invEv = jnp.where(jnp.abs(self.last_eigenvalues / self.last_eigenvalues[-1]) > 1e-14, 1. / self.last_eigenvalues, 0.)

        residual = 1.0
        cutoff = 1e-2
        F_norm = jnp.linalg.norm(F)
        while residual > self.pinv_tol and cutoff > self.pinv_cutoff:
            residual, cutoff, pinvEv = self._regularizer_step(
                cutoff, snr, self.last_eigenvalues, invEv, self._VtF, F_norm, solver_state.exact_sampler
            )

        update = jnp.real(jnp.dot(self.last_eigenvectors, (pinvEv * self._VtF)))
        info = dict(
            residual=residual.item(),
            pinv_cutoff=max(cutoff, self.pinv_cutoff),
            snr=snr,
            # spectrum=self.last_eigenvalues
        )

        return update, info
    
    @jax.jit(static_argnums=(0, 7))
    def _regularizer_step(self, cutoff, snr, eigenvalues, invEv, VtF, F_norm, exact_sampler):
        cutoff = 0.8 * cutoff
        effective_cutoff = jnp.max(jnp.array([cutoff, self.pinv_cutoff]))

        # Set regularizer for singular value cutoff
        regularizer = 1. / (1. + (effective_cutoff / jnp.abs(eigenvalues / eigenvalues[-1]))**6)

        if not exact_sampler:
            # Construct a soft cutoff based on the SNR
            regularizer *= 1. / (1. + (self.snr_tol / snr)**6)

        pinvEv = invEv * regularizer
        residual = jnp.linalg.norm((pinvEv * eigenvalues - jnp.ones_like(pinvEv)) * VtF) / F_norm

        return residual, cutoff, pinvEv
    
    def _transform_to_eigenbasis(self, S, F):
        if self._diagonalize_on_device:
            try:
                self._ev, self._V = jnp.linalg.eigh(S)
            except ValueError:
                warnings.warn(
                    "jax.numpy.linalg.eigh raised an exception. Falling back to " 
                    "numpy.linalg.eigh for diagonalization.", RuntimeWarning
                )
            
                self._ev, self._V = _eigh_numpy(S)
        else:
            self._ev, self._V = _eigh_numpy(S)

        self._VtF = jnp.dot(jnp.transpose(jnp.conj(self._V)), F)