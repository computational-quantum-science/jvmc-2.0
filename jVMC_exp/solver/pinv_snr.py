import jax.numpy as jnp
import warnings
import jax
from typing import Literal

from jVMC_exp.solver.base import AbstractSolver
from jVMC_exp.solver.util import diagonalize, smooth_cutoff_fn

@jax.jit
def get_snr(Vtb, Vtb_var, num_samples):
    return jnp.sqrt(
        jnp.abs(num_samples * (jnp.conj(Vtb) * Vtb) / (Vtb_var + 1e-14))
    ).ravel()
    
class PinvSNR(AbstractSolver):
    """
    Pseudo-inverse solver based on an eigenvalue decomposition of the covariance
    matrix.

    The inverse is regularized using two smooth filters:

    - an eigenvalue cutoff controlled by ``pinv_cutoff``, which suppresses
      ill-conditioned directions of the covariance matrix;
    - a signal-to-noise ratio (SNR) cutoff controlled by ``snr_tol``, which
      suppresses statistically unresolved update directions. This regularization
      is skipped by default. Change snr_tol to activate it.

    The effective eigenvalue cutoff is chosen adaptively such that the residual
    force discarded by the regularization is below ``pinv_tol`` whenever
    possible.

    Parameters
    ----------
    snr_tol : float, default=0
        Minimum signal-to-noise ratio of an eigenmode before it contributes
        significantly to the update.

    pinv_tol : float, default=1e-14
        Target residual. The solver decreases the eigenvalue cutoff until the
        fraction of discarded force falls below this threshold or the minimum
        cutoff ``pinv_cutoff`` is reached.

    pinv_cutoff : float, default=1e-8
        Minimum allowed relative eigenvalue cutoff.

    diagonalization_mode : {"device", "distributed", "host"}, default="device"
        Backend used to diagonalize the covariance matrix; forwarded to
        :func:`jVMC_exp.solver.util.diagonalize`. ``"device"`` uses
        ``jax.numpy.linalg.eigh``; ``"distributed"`` uses a distributed
        eigensolver across the device mesh, falling back to ``"device"``
        if the prerequisites for distributed execution aren't met;
        ``"host"`` uses ``numpy.linalg.eigh``.

    T_A : int, optional
        Tile size forwarded to :func:`jVMC_exp.solver.util.diagonalize`
        for ``diagonalization_mode="distributed"``. If not given, a tile
        size is chosen automatically.
    """
    def __init__(
            self, snr_tol=0, pinv_tol=1e-14, pinv_cutoff=1e-8, 
            diagonalization_mode: Literal["device", "distributed", "host"] = "device",
            T_A: int | None = None
        ):
        self._snr_tol = snr_tol
        self._pinv_tol = pinv_tol
        self._pinv_cutoff = pinv_cutoff
        self._diagonalization_mode = diagonalization_mode
        self._T_A = T_A
    
    @property
    def snr_tol(self):
        return self._snr_tol
    
    @property
    def pinv_tol(self):
        return self._pinv_tol

    @property
    def pinv_cutoff(self):
        return self._pinv_cutoff
    
    @property
    def _needs_dense_matrix(self) -> bool:
        return True

    def __call__(
            self, A, b, b_var=None, *,
            pad_size, effective_num_samples, exact_sampler, holomorphic,
            **kwargs
        ):
        """
        Solve ``A @ x = b`` for ``x`` via a regularized pseudo-inverse of `A`.

        Parameters
        ----------
        A : array_like
        b : array_like
        b_var : array_like, optional
            Per-component variance of `b`, used to compute the SNR-based
            regularization. If None, only the eigenvalue cutoff is applied.
        pad_size : int
            Number of trailing rows/columns `A` needs to be padded with so
            that its size is compatible with the diagonalization backend;
            see :func:`jVMC_exp.solver.util.diagonalize`.
        effective_num_samples : int
            Effective number of samples used to compute the SNR of each
            eigenmode.
        exact_sampler : bool
            If True, disables the SNR-based regularization.
        holomorphic : bool
            If False, only the real part of the update is returned.
        **kwargs
            Ignored; accepted for interface compatibility with other
            solvers.

        Returns
        -------
        x : jax.Array
            The regularized pseudo-inverse update ``A^+ @ b``.
        info : dict
            Diagnostic information about the solve: ``residual``,
            ``pinv_cutoff``, ``snr``, ``condition_number``, ``spectrum``
            (eigenvalues of `A`), and ``effective_rank``.
        """
        def unpad(x):
            return x[:-pad_size] if pad_size else x

        # Keep V padded so that it doesn't get replicated on devices
        ev, V = diagonalize(
            A, pad_size, mode=self._diagonalization_mode, T_A=self._T_A
        )
        ev = unpad(ev)
        if jnp.max(jnp.abs(ev)) < 1e-14:
            raise RuntimeError(
                f"Largest eigenvalue of the QGT is {jnp.max(jnp.abs(ev))}. "
                "QGT is most likely highly ill-conditioned/zero "
            )

        Vtb = unpad(jnp.dot(
            jnp.transpose(jnp.conj(V)), jnp.pad(b, (0, pad_size))
        ))

        snr = None
        if not exact_sampler:
            if b_var is not None and self.snr_tol != 0:
                snr = get_snr(
                    Vtb, 
                    unpad(jnp.dot(
                        jnp.abs(jnp.transpose(jnp.conj(V)))**2, jnp.pad(b_var, (0, pad_size))
                    )),
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
        invEv = jnp.where(jnp.abs(ev / ev[-1]) > 1e-14, 1. / ev, 0.)

        b_norm = jnp.linalg.norm(b) 
        residual = 1.0
        cutoff = 1e-2
        first = True
        while (residual > self.pinv_tol and cutoff > self.pinv_cutoff) or first:
            residual, cutoff, pinvEv, effective_rank = self._regularizer_step(
                cutoff, snr, ev, invEv, Vtb, b_norm, exact_sampler
            )

            first = False

        x = unpad(jnp.dot(
            V,
            jnp.pad((pinvEv * Vtb), (0, pad_size))
        ))
        x = x if holomorphic else jnp.real(x)

        info = dict(
            residual=residual.item(),
            pinv_cutoff=cutoff.item(),
            snr=snr,
            condition_number=(ev[-1] / jnp.min(jnp.abs(ev))).item(),
            spectrum=ev,
            effective_rank=effective_rank.item()
        )

        return x, info
    
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