"""
Randomized eigendecomposition of the quantum geometric tensor and the pseudo-inverse
solver built on it.

The ``k`` dominant eigenpairs of the left hand side :math:`A = q[S_0] + \\dots` are
obtained by randomized subspace iteration (Halko, Martinsson, Tropp, SIAM Rev. 53,
217 (2011)) without ever forming :math:`S_0 = J^\\dagger J`: the operator is only
applied to blocks of ``k + n_oversamples`` vectors, which costs
:math:`O(\\text{num\\_samples} \\cdot \\text{num\\_params} \\cdot r)` per application
instead of the :math:`O(\\text{num\\_params}^3)` of a dense diagonalization. The
Rayleigh-Ritz matrix :math:`B = Q^\\dagger A Q` of size ``r x r`` is the only matrix
that is diagonalized, with any of the backends of
:func:`jVMC_exp.solver.util.diagonalize`, including the distributed one.
"""
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as random

from jVMC_exp.sharding_config import MESH, REPLICATED_SHARDING, broadcast_split_key
from jVMC_exp.solver.pinv import PinvSNR
from jVMC_exp.solver.qgt_operator import QGTOperator, _qgt_product, transformed_product
from jVMC_exp.solver.util import diagonalize

WHICH = ("LM", "LA")


def _pin(x):
    return jax.lax.with_sharding_constraint(x, REPLICATED_SHARDING)


@partial(jax.jit, static_argnames=("total_rank", "n_iter", "apply_mode", "has_diag"))
def _subspace_iteration(jac, key, diag, diag_shift, *, total_rank, n_iter, apply_mode, has_diag):
    """
    Orthonormal basis ``Q`` of the dominant subspace of the operator and its
    Rayleigh-Ritz matrix ``B = Q^H A Q``.

    ``apply_mode`` selects how the operator is applied to a block:

    * ``"identity"``: :math:`S_0 Q` (complex probe),
    * ``"real"``: :math:`\\text{Re}(S_0) Q` (real probe),
    * ``"imag_real"``: :math:`\\text{Im}(S_0) Q` for a real probe. The operator is
      :math:`i\\,\\text{Im}(S_0)`; the global factor ``i`` is stripped so that the
      iterate stays real, and restored once in ``B``. Only valid without
      diagonal terms, which do not share the factor ``i``.
    * ``"imag_cpx"``: :math:`i\\,\\text{Im}(S_0) Q` (complex probe), used with
      diagonal terms.

    The operator is applied ``2 * n_iter + 2`` times. ``Q`` is drawn inside the jit so
    that it is created replicated on every device. The ``(num_params, r)`` blocks are
    replicated because ``jnp.linalg.qr`` has no distributed lowering; the parallelism
    lies in the products with the sample-sharded Jacobian.
    """
    n = jac.shape[1]
    real_dtype = jnp.real(jac).dtype

    def apply(Q):
        if apply_mode == "imag_real":
            SQ = jnp.imag(_qgt_product(jac, Q))
        else:
            SQ = transformed_product(jac, Q, "imag" if apply_mode == "imag_cpx" else apply_mode)
        if has_diag:
            SQ = SQ + diag[:, None] * Q + diag_shift * Q

        return _pin(SQ)

    if apply_mode in ("real", "imag_real"):
        Omega = random.normal(key, (n, total_rank), dtype=real_dtype)
    else:
        key_re, key_im = random.split(key)
        Omega = (
            random.normal(key_re, (n, total_rank), dtype=real_dtype)
            + 1j * random.normal(key_im, (n, total_rank), dtype=real_dtype)
        )
    Omega = _pin(Omega)

    Y = apply(Omega)
    for _ in range(n_iter):
        Q, _ = jnp.linalg.qr(Y, mode="reduced")
        Y = apply(Q)
        Q, _ = jnp.linalg.qr(Y, mode="reduced")
        Y = apply(Q)
    Q, _ = jnp.linalg.qr(Y, mode="reduced")

    B = jnp.conj(Q).T @ apply(Q)
    if apply_mode == "imag_real":
        B = 1j * B
    # Hermitian by construction, symmetrized against round-off
    B = 0.5 * (B + jnp.conj(B).T)

    return Q, B


@partial(jax.jit, static_argnames=("k", "which"))
def _select(ev, V, *, k, which):
    """
    The ``k`` Ritz pairs with the largest magnitude (``"LM"``) or the largest algebraic
    value (``"LA"``), re-sorted into ASCENDING order of the eigenvalue, as
    ``jnp.linalg.eigh`` returns them: the pseudo-inverse solvers read ``ev[-1]`` as the
    largest eigenvalue.
    """
    if which == "LM":
        idx = jnp.argsort(jnp.abs(ev))[::-1][:k]
    else:
        idx = jnp.argsort(ev)[::-1][:k]
    idx = idx[jnp.argsort(ev[idx])]

    return ev[idx], V[:, idx]


def _apply_mode(operator: QGTOperator):
    if operator.mode == "imag":
        return "imag_cpx" if operator.has_diagonal_terms else "imag_real"

    return operator.mode


def randomized_eigh(
        operator: QGTOperator, k: int, key, n_oversamples: int = 0, n_iter: int = 4,
        which: Literal["LM", "LA"] = "LM",
        diagonalization_mode: Literal["device", "distributed", "host"] = "device",
        T_A: int | None = None
    ):
    """
    The ``k`` dominant eigenpairs of a :class:`QGTOperator` by randomized subspace
    iteration.

    Parameters
    ----------
    operator : QGTOperator
        The Hermitian operator, built from the Jacobian.
    k : int
        Number of eigenpairs, ``0 < k <= num_params``.
    key : jax.random.PRNGKey
        Key for the random probe block. In a multi-process run every process has
        to pass the same key.
    n_oversamples : int, default=0
        Extra probe vectors beyond ``k``. The cheap accuracy lever: each costs one
        column in the products and buys more than ``n_iter`` when the spectrum is flat.
        The total ``k + n_oversamples`` is capped at ``num_params``.
    n_iter : int, default=4
        Subspace (power) iterations. The operator is applied ``2 * n_iter + 2`` times.
    which : {"LM", "LA"}, default="LM"
        Select the eigenvalues of largest magnitude or largest algebraic value.
        Subspace iteration captures the largest-magnitude subspace, so ``"LA"`` is
        only meaningful for a positive semi-definite operator (``mode="real"`` or
        ``"identity"``), where the two coincide.
    diagonalization_mode : {"device", "distributed", "host"}, default="device"
        Backend for the diagonalization of the ``r x r`` Rayleigh-Ritz matrix,
        forwarded to :func:`jVMC_exp.solver.util.diagonalize`. For
        ``"distributed"`` the matrix is padded to a multiple of the number of
        devices (the padded eigenvalues are dropped again).
    T_A : int, optional
        Tile size for ``diagonalization_mode="distributed"``.

    Returns
    -------
    ev : jax.Array
        The ``k`` eigenvalues in ascending order.
    V : jax.Array
        The eigenvectors as columns, shape ``(num_params, k)``, replicated.
    """
    n = operator.shape[0]
    k = int(k)
    if not 0 < k <= n:
        raise ValueError(f"k has to satisfy 0 < k <= num_params = {n}, got {k}.")
    if which not in WHICH:
        raise ValueError(f"which has to be one of {WHICH}, got {which!r}.")
    if int(n_oversamples) < 0 or int(n_iter) < 0:
        raise ValueError("n_oversamples and n_iter have to be non-negative.")

    total_rank = min(k + int(n_oversamples), n)
    Q, B = _subspace_iteration(
        operator.jacobian, key, operator._diag, operator._shift,
        total_rank=total_rank, n_iter=int(n_iter),
        apply_mode=_apply_mode(operator), has_diag=operator.has_diagonal_terms
    )

    # The distributed backend needs a size divisible by the number of devices; the
    # padded modes get a huge eigenvalue and end up last in the ascending spectrum.
    pad = (-total_rank) % MESH.shape["devices"] if diagonalization_mode == "distributed" else 0
    if pad:
        B = jnp.pad(B, ((0, pad), (0, pad)))
        Q = jnp.pad(Q, ((0, 0), (0, pad)))
    ev, V_small = diagonalize(B, pad_size=pad, mode=diagonalization_mode, T_A=T_A)
    V = Q @ V_small
    if pad:
        ev, V = ev[:total_rank], V[:, :total_rank]

    return _select(ev, jax.device_put(V, REPLICATED_SHARDING), k=k, which=which)


class RandomizedPinvSNR(PinvSNR):
    """
    :class:`PinvSNR` with a randomized eigendecomposition of the quantum geometric
    tensor as backend.

    Instead of forming and diagonalizing the dense ``num_params x num_params`` left
    hand side, the ``n_components`` dominant eigenpairs are computed matrix-free from
    the Jacobian by :func:`randomized_eigh`, and the regularized pseudo-inverse is
    built in that truncated eigenbasis exactly as in :class:`PinvSNR` (eigenvalue
    cutoff, optional SNR cutoff, adaptive cutoff until ``pinv_tol`` is met).

    The solver is matrix-free (``_needs_dense_matrix`` is ``False``): the optimizer
    hands it a :class:`jVMC_exp.solver.qgt_operator.QGTOperator` instead of a matrix,
    and the parameter padding of the dense path does not apply. The Jacobian has to
    be materialized (``batched_jacobian=False``).

    Only the top ``n_components`` eigenvalues are resolved. Choose ``n_components``
    such that the smallest retained eigenvalue already lies below ``pinv_cutoff``
    relative to the largest one, otherwise physical directions are truncated;
    ``info["true_residual"]`` (:math:`\\|A x - b\\| / \\|b\\|`, one extra operator
    application) measures what the truncation discards, while ``info["residual"]``
    only sees the force inside the retained subspace.

    Parameters
    ----------
    n_components : int
        Number of eigenpairs ``k`` to retain.
    n_oversamples : int, default=0
        Extra probe vectors, see :func:`randomized_eigh`.
    n_iter : int, default=4
        Subspace iterations, see :func:`randomized_eigh`.
    seed : int or jax.random.PRNGKey, default=4321
        Seed of the random probes. The key is split (and broadcast from process 0)
        before every solve.
    which : {"LM", "LA"}, default="LM"
        Eigenvalue selection, see :func:`randomized_eigh`.
    snr_tol, pinv_tol, pinv_cutoff, snr_mode
        As in :class:`PinvSNR`.
    diagonalization_mode : {"device", "distributed", "host"}, default="device"
        Backend for the ``r x r`` Rayleigh-Ritz matrix, with
        ``r = n_components + n_oversamples``; see :func:`randomized_eigh`.
    T_A : int, optional
        Tile size for ``diagonalization_mode="distributed"``.
    compute_true_residual : bool, default=True
        Report ``true_residual`` in the info dictionary, at the cost of one more
        operator application per solve.
    """
    def __init__(
            self, n_components: int, n_oversamples: int = 0, n_iter: int = 4,
            seed=4321, which: Literal["LM", "LA"] = "LM",
            snr_tol=0, pinv_tol=1e-14, pinv_cutoff=1e-8,
            snr_mode: Literal["empiric", "gaussian"] = "empiric",
            diagonalization_mode: Literal["device", "distributed", "host"] = "device",
            T_A: int | None = None, compute_true_residual: bool = True
        ):
        super().__init__(
            snr_tol=snr_tol, pinv_tol=pinv_tol, pinv_cutoff=pinv_cutoff, snr_mode=snr_mode,
            diagonalization_mode=diagonalization_mode, T_A=T_A
        )
        if int(n_components) <= 0:
            raise ValueError(f"n_components has to be positive, got {n_components}.")
        if which not in WHICH:
            raise ValueError(f"which has to be one of {WHICH}, got {which!r}.")

        self._n_components = int(n_components)
        self._n_oversamples = int(n_oversamples)
        self._n_iter = int(n_iter)
        self._which = which
        self._compute_true_residual = bool(compute_true_residual)
        self._key = seed if isinstance(seed, jax.Array) else random.PRNGKey(seed)

    @property
    def n_components(self):
        return self._n_components

    @property
    def n_oversamples(self):
        return self._n_oversamples

    @property
    def n_iter(self):
        return self._n_iter

    @property
    def which(self):
        return self._which

    @property
    def _needs_dense_matrix(self) -> bool:
        return False

    def _next_key(self):
        self._key, subkey = broadcast_split_key(self._key, 2)

        return subkey

    def _diagonalize(self, A: QGTOperator, b, pad_size):
        if not isinstance(A, QGTOperator):
            raise TypeError(
                f"{type(self).__name__} is matrix-free and expects the left hand side as a "
                f"jVMC_exp.solver.qgt_operator.QGTOperator, got {type(A).__name__}. Use "
                "PinvSNR for a dense matrix."
            )

        ev, V = randomized_eigh(
            A, self.n_components, self._next_key(),
            n_oversamples=self.n_oversamples, n_iter=self.n_iter, which=self.which,
            diagonalization_mode=self._diagonalization_mode, T_A=self._T_A
        )
        if jnp.max(jnp.abs(ev)) < 1e-14:
            raise RuntimeError(
                f"Largest eigenvalue of the matrix A to invert is {jnp.max(jnp.abs(ev))}. "
                "A is most likely highly ill-conditioned/zero "
            )
        Vtb = jnp.dot(jnp.transpose(jnp.conj(V)), b)

        return ev, V, Vtb

    def __call__(self, A: QGTOperator, b, *, pad_size=0, **kwargs):
        """
        Solve ``A @ x = b`` in the truncated eigenbasis of ``A``. ``pad_size`` is
        accepted for interface compatibility and ignored: the operator acts on the
        unpadded parameter space.
        """
        x, info = super().__call__(A, b, pad_size=0, **kwargs)

        info["n_components"] = self.n_components
        if self._compute_true_residual:
            info["true_residual"] = (jnp.linalg.norm(A(x) - b) / jnp.linalg.norm(b)).item()

        return x, info
