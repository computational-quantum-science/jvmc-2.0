import jax.numpy as jnp
import warnings
import numpy as np
import jax

from jVMC_exp.solver.base import AbstractSolver

def _eigh_numpy(S):
    e, V = np.linalg.eigh(np.array(S))

    return jnp.array(e), jnp.array(V)

def smooth_cutoff_fn(x, c, exp=6):
    return 1 / (1 + (c / x)**exp)

@jax.jit(static_argnums=(2,))
def get_snr(Vtb, Vtb_var, num_samples):
    return jnp.sqrt(jnp.abs(num_samples * (jnp.conj(Vtb) * Vtb) / (Vtb_var + 1e-14))).ravel()
    
class PinvSNR(AbstractSolver):
    """
    Pseudo-inverse solver based on an eigenvalue decomposition of the covariance
    matrix.

    The inverse is regularized using two smooth filters:

    - an eigenvalue cutoff controlled by ``pinv_cutoff``, which suppresses
      ill-conditioned directions of the covariance matrix;
    - a signal-to-noise ratio (SNR) cutoff controlled by ``snr_tol``, which
      suppresses statistically unresolved update directions.

    The effective eigenvalue cutoff is chosen adaptively such that the residual
    force discarded by the regularization is below ``pinv_tol`` whenever
    possible.

    Parameters
    ----------
    snr_tol : float, default=2
        Minimum signal-to-noise ratio of an eigenmode before it contributes
        significantly to the update.

    pinv_tol : float, default=1e-14
        Target residual. The solver decreases the eigenvalue cutoff until the
        fraction of discarded force falls below this threshold or the minimum
        cutoff ``pinv_cutoff`` is reached.

    pinv_cutoff : float, default=1e-8
        Minimum allowed relative eigenvalue cutoff.

    diagonalize_on_device : bool, default=True
        If True, diagonalize the covariance matrix using JAX. Otherwise,
        fall back to NumPy.
    """
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

    def __call__(
            self, A, b, b_var=None, *, 
            effective_num_samples, exact_sampler, holomorphic, **kwargs
        ):
        # Transform equation to eigenbasis and compute Signal to Noise Ratio
        self._transform_to_eigenbasis(A, b)
        b_norm = jnp.linalg.norm(b)

        if jnp.abs(self.last_eigenvalues[-1]) < 1e-14:
                    raise RuntimeError(
                        f"Largest eigenvalue of the QGT is {self.last_eigenvalues[-1]}. "
                        "QGT is most likely highly ill-conditioned/zero "
                    )

        snr = None
        if not exact_sampler:
            if b_var is not None:
                snr = get_snr(
                    self._Vtb, 
                    jnp.dot(jnp.abs(jnp.transpose(jnp.conj(self._V)))**2, b_var),
                    effective_num_samples
                )
            elif self.snr_tol != 0 and jax.process_index() == 0:
                warnings.warn(
                    f"PinvSNR has snr_tol={self.snr_tol}, but was called with b_var=None, "
                    "so no SNR-based regularization can be applied. "
                    "Pass b_var to enable the SNR cutoff, or set snr_tol=0 to silence " 
                    "this warning if that's intended.",
                    UserWarning,
                )

        # Discard eigenvalues below numerical precision
        invEv = jnp.where(
            jnp.abs(self.last_eigenvalues / self.last_eigenvalues[-1]) > 1e-14
            , 1. / self.last_eigenvalues,
            0.
        )
        
        residual = 1.0
        cutoff = 1e-2
        first = True
        while (residual > self.pinv_tol and cutoff > self.pinv_cutoff) or first:
            residual, cutoff, pinvEv, effective_rank = self._regularizer_step(
                cutoff, snr, self.last_eigenvalues, invEv, self._Vtb, b_norm, exact_sampler
            )

            first = False

        update = jnp.dot(self.last_eigenvectors, (pinvEv * self._Vtb))
        update = update if holomorphic else jnp.real(update)
        info = dict(
            residual=residual.item(),
            pinv_cutoff=cutoff.item(),
            snr=snr,
            condition_number=(self.last_eigenvalues[-1] / jnp.min(jnp.abs(self.last_eigenvalues))).item(),
            spectrum=self.last_eigenvalues,
            effective_rank=effective_rank.item()
        )

        return update, info
    
    @jax.jit(static_argnums=(0, 7))
    def _regularizer_step(self, cutoff, snr, eigenvalues, invEv, Vtb, b_norm, exact_sampler):
        # Set regularizer for singular value cutoff
        cutoff = jnp.max(jnp.array([0.8 * cutoff, self.pinv_cutoff]))
        regularizer = smooth_cutoff_fn(jnp.abs(eigenvalues / eigenvalues[-1]), cutoff)

        # Construct a soft cutoff based on the SNR
        if not exact_sampler and snr is not None:
            regularizer *= smooth_cutoff_fn(snr, self.snr_tol)

        pinvEv = invEv * regularizer
        residual = jnp.linalg.norm((pinvEv * eigenvalues - 1) * Vtb) / b_norm
        effective_rank = jnp.mean(regularizer)

        return residual, cutoff, pinvEv, effective_rank
    
    def _transform_to_eigenbasis(self, A, b):
        if self._diagonalize_on_device:
            try:
                self._ev, self._V = jnp.linalg.eigh(A)
            except ValueError:
                warnings.warn(
                    "jax.numpy.linalg.eigh raised an exception. Falling back to " 
                    "numpy.linalg.eigh for diagonalization.", RuntimeWarning
                )
            
                self._ev, self._V = _eigh_numpy(A)
        else:
            self._ev, self._V = _eigh_numpy(A)

        self._Vtb = jnp.dot(jnp.transpose(jnp.conj(self._V)), b)
