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

Two layouts of the ``(num_params, r)`` blocks:

* ``"samples"`` (a :class:`QGTOperator`): the blocks are replicated on every device,
  only the Jacobian is sharded (over the samples); orthonormalization by
  ``jnp.linalg.qr``.
* ``"parameters"`` (a :class:`ShardedQGTOperator`): the blocks are sharded over the
  parameter rows of a 2D mesh, orthonormalization by shifted CholeskyQR3
  (:func:`_cholesky_qr`) or jaxmg's distributed QR; every array whose size grows with
  the number of parameters is divided by the number of parameter groups. The
  ``r x r`` Rayleigh-Ritz matrix and its eigenvectors are sharded in blocks as well
  (:func:`_ritz_matrix_blocks`, :func:`_ritz_vectors_blocks`); with the
  ``"distributed"`` diagonalization no ``r x r`` object is replicated.
"""
from functools import partial
from typing import Literal
import math
import os
import time
import warnings

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from jax.sharding import PartitionSpec as P

from jVMC_exp import global_defs
from jVMC_exp.sharding_config import MESH, REPLICATED_SHARDING, REPLICATED_SPEC, broadcast_split_key
from jVMC_exp.solver import util as solver_util
from jVMC_exp.solver.pinv import PinvSNR
from jVMC_exp.solver.qgt_operator import (
    QGTOperator, ShardedQGTOperator, ParameterLayout, parameter_mesh,
    PARAMS_AXIS, SAMPLES_AXIS, JACOBIAN_SPEC, ROWS_SPEC,
    _column_block_product, _qgt_product_im, transformed_product, _apply
)
from jVMC_exp.solver.util import diagonalize

WHICH = ("LM", "LA")
LAYOUTS = ("samples", "parameters")
QR_MODES = ("cholesky", "distributed")


def _pin(x):
    return jax.lax.with_sharding_constraint(x, REPLICATED_SHARDING)


# JVMC_RSVD_DEBUG=1: wait for every stage of randomized_eigh and print the device memory
# of process 0 afterwards, so that an out-of-memory error is reported at the stage that
# caused it instead of at the next host synchronization
_DEBUG = os.environ.get("JVMC_RSVD_DEBUG", "").lower() in ("1", "true", "yes")

# Stage timings of the current randomized_eigh call: switched on by JVMC_RSVD_DEBUG or by
# RandomizedPinvSNR(stage_timings=True); every stage then synchronizes with the devices.
_STAGES = {"enabled": _DEBUG, "clock": None, "timings": {}}

# Stage name prefix -> key of the accumulated timing
_STAGE_KEYS = (
    ("application", "apply"), ("QR", "qr"), ("probe", "probe"),
    ("Rayleigh-Ritz matrix", "ritz_matrix"), ("Rayleigh-Ritz diagonalization", "eigh"),
    ("Ritz vectors", "ritz_vectors"), ("selection", "select"),
)


def _begin_stages(enabled):
    _STAGES["enabled"] = bool(enabled) or _DEBUG
    _STAGES["clock"] = time.perf_counter()
    _STAGES["timings"] = {}


def _stage_timings():
    """The timings accumulated since `_begin_stages`, by stage kind, in seconds."""
    return dict(_STAGES["timings"])


def _debug_stage(name, *arrays):
    """
    End of a stage of `randomized_eigh`: with stage timings enabled, wait for ``arrays``
    and add the time since the previous stage to its kind; with JVMC_RSVD_DEBUG also
    print it together with the device memory of process 0.
    """
    if not _STAGES["enabled"]:
        return
    jax.block_until_ready(arrays)
    now = time.perf_counter()
    elapsed = 0.0 if _STAGES["clock"] is None else now - _STAGES["clock"]
    _STAGES["clock"] = now
    for prefix, key in _STAGE_KEYS:
        if name.startswith(prefix):
            _STAGES["timings"][key] = _STAGES["timings"].get(key, 0.0) + elapsed
            break
    if _DEBUG and jax.process_index() == 0:
        stats = jax.local_devices()[0].memory_stats() or {}
        in_use = stats.get("bytes_in_use", 0) / 2 ** 30
        peak = stats.get("peak_bytes_in_use", 0) / 2 ** 30
        print(f"[rsvd] {name}: {elapsed:.1f}s, {in_use:.1f} GiB in use, peak {peak:.1f} GiB", flush=True)


@jax.jit
def _ritz_vectors(Q, V_small):
    return _pin(Q @ V_small)


@partial(jax.jit, static_argnames=("apply_mode", "has_diag"))
def _apply_block(jac, Q, diag, diag_shift, *, apply_mode, has_diag):
    """
    The operator applied to a block ``Q``. ``apply_mode`` selects how:

    * ``"identity"``: :math:`S_0 Q` (complex probe),
    * ``"real"``: :math:`\\text{Re}(S_0) Q` (real probe),
    * ``"imag_real"``: :math:`\\text{Im}(S_0) Q` for a real probe. The operator is
      :math:`i\\,\\text{Im}(S_0)`; the global factor ``i`` is stripped so that the
      iterate stays real, and restored once in the Rayleigh-Ritz matrix. Only valid
      without diagonal terms, which do not share the factor ``i``.
    * ``"imag_cpx"``: :math:`i\\,\\text{Im}(S_0) Q` (complex probe), used with
      diagonal terms.
    """
    if apply_mode == "imag_real":
        SQ = _qgt_product_im(jac, Q)
    else:
        SQ = transformed_product(jac, Q, "imag" if apply_mode == "imag_cpx" else apply_mode)
    if has_diag:
        SQ = SQ + diag[:, None] * Q + diag_shift * Q

    return _pin(SQ)


@partial(jax.jit, static_argnames=("width",))
def _alloc_output(piece, *, width):
    return _pin(jnp.zeros((piece.shape[0], width), piece.dtype))


@partial(jax.jit, static_argnames=("start",), donate_argnums=0)
def _put_block(out, piece, *, start):
    return jax.lax.dynamic_update_slice_in_dim(out, piece, start, axis=1)


@partial(jax.jit, static_argnames=("start", "stop"))
def _take_block(Q, *, start, stop):
    return jax.lax.slice_in_dim(Q, start, stop, axis=1)


def _apply_operator(jac, Q, diag, diag_shift, *, apply_mode, has_diag):
    """
    The operator applied to the ``(num_params, r)`` block ``Q``, one column block of at
    most ``global_defs.MATMUL_CHUNK_BYTES`` of output per dispatch, written into a
    preallocated result.

    Done eagerly rather than inside one jitted kernel on purpose: this bounds what
    is alive at once to the input, the output and the temporaries of one block
    (XLA's buffer assignment for the whole iteration in one executable kept several
    full ``(num_params, r)`` temporaries, 47 GiB for 94640 x 16000 doubles).
    """
    n, r = Q.shape
    out_dtype = jnp.result_type(jac.dtype, Q.dtype) if apply_mode in ("identity", "imag_cpx") else Q.dtype
    budget = global_defs.MATMUL_CHUNK_BYTES
    block = r if not budget else max(1, min(r, budget // (n * jnp.dtype(out_dtype).itemsize)))
    if block >= r:
        return _apply_block(jac, Q, diag, diag_shift, apply_mode=apply_mode, has_diag=has_diag)

    out = None
    for start in range(0, r, block):
        stop = min(start + block, r)
        piece = _apply_block(
            jac, _take_block(Q, start=start, stop=stop), diag, diag_shift,
            apply_mode=apply_mode, has_diag=has_diag
        )
        if out is None:
            out = _alloc_output(piece, width=r)
        out = _put_block(out, piece, start=start)
        del piece

    return out


@partial(jax.jit, static_argnames=("real_dtype", "n", "total_rank", "complex_probe"))
def _probe(key, *, real_dtype, n, total_rank, complex_probe):
    """The random probe block, drawn inside the jit so that it is created replicated."""
    if complex_probe:
        key_re, key_im = random.split(key)
        Omega = (
            random.normal(key_re, (n, total_rank), dtype=real_dtype)
            + 1j * random.normal(key_im, (n, total_rank), dtype=real_dtype)
        )
    else:
        Omega = random.normal(key, (n, total_rank), dtype=real_dtype)

    return _pin(Omega)


@partial(jax.jit, donate_argnums=0)
def _orthonormalize(Y):
    """``Q`` of the reduced QR of ``Y``; ``Y`` is donated so that the factorization can reuse its buffer."""
    Q, _ = jnp.linalg.qr(Y, mode="reduced")

    return _pin(Q)


@partial(jax.jit, static_argnames=("restore_i",))
def _rayleigh_ritz(Q, SQ, *, restore_i):
    B = jnp.conj(Q).T @ SQ
    if restore_i:
        B = 1j * B
    # Hermitian by construction, symmetrized against round-off
    return 0.5 * (B + jnp.conj(B).T)


def _subspace_iteration(jac, key, diag, diag_shift, *, total_rank, n_iter, apply_mode, has_diag):
    """
    Orthonormal basis ``Q`` of the dominant subspace of the operator and its
    Rayleigh-Ritz matrix ``B = Q^H A Q``, see :func:`_apply_block` for ``apply_mode``.

    The operator is applied ``2 * n_iter + 2`` times. The ``(num_params, r)`` blocks
    are replicated because ``jnp.linalg.qr`` has no distributed lowering; the
    parallelism lies in the products with the sample-sharded Jacobian. The stages
    are separate dispatches (see :func:`_apply_operator`), so that at most two of
    these blocks plus the temporaries of one column block (applications) or the QR
    workspace (orthonormalization) are alive at once: for 94640 x 16000 doubles
    about 25 GB plus the Jacobian shard.
    """
    n = jac.shape[1]
    real_dtype = jnp.real(jac).dtype
    kwargs = dict(apply_mode=apply_mode, has_diag=has_diag)

    _debug_stage("start of the subspace iteration")
    Omega = _probe(key, real_dtype=real_dtype, n=n, total_rank=total_rank, complex_probe=apply_mode not in ("real", "imag_real"))
    _debug_stage("probe", Omega)
    Y = _apply_operator(jac, Omega, diag, diag_shift, **kwargs)
    del Omega
    _debug_stage("application 1", Y)
    for it in range(n_iter):
        Q = _orthonormalize(Y)
        _debug_stage(f"QR {2 * it + 1}", Q)
        Y = _apply_operator(jac, Q, diag, diag_shift, **kwargs)
        del Q
        _debug_stage(f"application {2 * it + 2}", Y)
        Q = _orthonormalize(Y)
        _debug_stage(f"QR {2 * it + 2}", Q)
        Y = _apply_operator(jac, Q, diag, diag_shift, **kwargs)
        del Q
        _debug_stage(f"application {2 * it + 3}", Y)
    Q = _orthonormalize(Y)
    del Y
    _debug_stage(f"QR {2 * n_iter + 1}", Q)
    SQ = _apply_operator(jac, Q, diag, diag_shift, **kwargs)
    _debug_stage(f"application {2 * n_iter + 2}", SQ)
    B = _rayleigh_ritz(Q, SQ, restore_i=apply_mode == "imag_real")
    del SQ
    _debug_stage("Rayleigh-Ritz matrix", B)

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
        T_A: int | None = None, qr_mode: Literal["cholesky", "distributed"] = "cholesky"
    ):
    """
    The ``k`` dominant eigenpairs of a :class:`QGTOperator` by randomized subspace
    iteration.

    Parameters
    ----------
    operator : QGTOperator or ShardedQGTOperator
        The Hermitian operator, built from the Jacobian. For a
        :class:`ShardedQGTOperator` the blocks are sharded over the parameter rows
        (see :func:`_subspace_iteration_rows`) and so are the returned eigenvectors.
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
        devices (the padded eigenvalues are dropped again). For a
        :class:`ShardedQGTOperator` the matrix is always formed block-sharded (padded
        the same way); ``"distributed"`` hands the blocks to jaxmg's ``syevd`` and the
        eigenvectors stay block-sharded, the other modes gather it and diagonalize
        it replicated (:func:`_diagonalize_blocks`).
    T_A : int, optional
        Tile size for ``diagonalization_mode="distributed"`` (and for
        ``qr_mode="distributed"``).
    qr_mode : {"cholesky", "distributed"}, default="cholesky"
        Orthonormalization of the blocks of a :class:`ShardedQGTOperator`: shifted
        CholeskyQR3 (:func:`_cholesky_qr`, its ``r x r`` Gram matrix and Cholesky
        factor replicated) or jaxmg's distributed QR (:func:`_distributed_qr`, falls
        back to ``"cholesky"`` with a warning without jaxmg, a GPU backend, one
        process per GPU, or for a mesh with more than one sample group). Ignored for
        a :class:`QGTOperator`.

    Returns
    -------
    ev : jax.Array
        The ``k`` eigenvalues in ascending order.
    V : jax.Array
        The eigenvectors as columns, shape ``(num_params, k)``, replicated. For a
        :class:`ShardedQGTOperator` shape ``(padded_num_params, k)``, sharded over the
        rows like the blocks, the padding rows zero: stripping them would gather the
        array (as for the padded eigenvectors of :func:`jVMC_exp.solver.util.diagonalize`).
    """
    n = operator.shape[0]
    k = int(k)
    if not 0 < k <= n:
        raise ValueError(f"k has to satisfy 0 < k <= num_params = {n}, got {k}.")
    if which not in WHICH:
        raise ValueError(f"which has to be one of {WHICH}, got {which!r}.")
    if int(n_oversamples) < 0 or int(n_iter) < 0:
        raise ValueError("n_oversamples and n_iter have to be non-negative.")
    if qr_mode not in QR_MODES:
        raise ValueError(f"qr_mode has to be one of {QR_MODES}, got {qr_mode!r}.")

    total_rank = min(k + int(n_oversamples), n)
    if isinstance(operator, ShardedQGTOperator):
        return _randomized_eigh_rows(
            operator, k, key, total_rank=total_rank, n_iter=int(n_iter), which=which,
            diagonalization_mode=diagonalization_mode, T_A=T_A, qr_mode=qr_mode
        )

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
    _debug_stage("Rayleigh-Ritz diagonalization", ev, V_small)

    # The small eigenvector matrix is replicated explicitly (r x r) before the product
    # with the replicated Q, so that the partitioner does not build partial
    # (num_params, r) products on every device.
    V_small = jax.device_put(V_small, REPLICATED_SHARDING)
    V = _ritz_vectors(Q, V_small)
    del Q, V_small
    if pad:
        ev, V = ev[:total_rank], V[:, :total_rank]
    _debug_stage("Ritz vectors", V)

    # Selecting and reordering is a copy of the (num_params, k) block; skipped when
    # it would be the identity (k == total_rank with an already ascending spectrum).
    ev_host = np.asarray(ev)
    idx = np.argsort(np.abs(ev_host))[::-1][:k] if which == "LM" else np.argsort(ev_host)[::-1][:k]
    idx = idx[np.argsort(ev_host[idx])]
    if k == total_rank and np.array_equal(idx, np.arange(total_rank)):
        return ev, V

    ev, V = _select(ev, V, k=k, which=which)
    _debug_stage("selection", ev, V)

    return ev, V


# ----------------------------------------------------------------------------------
# Parameter-sharded layout: blocks sharded over the parameter rows (ShardedQGTOperator)
# ----------------------------------------------------------------------------------

_PARTS = {"identity": "cpx", "real": "re", "imag_real": "im"}


def _local_apply_rows(jac, q, diag, diag_shift, *, apply_mode, has_diag):
    """The operator applied to the rows ``q`` of this device, see :func:`_apply_block` for ``apply_mode``."""
    if apply_mode == "imag_cpx":
        r = q.shape[1]
        Sp = _column_block_product(jac, jnp.concatenate([jnp.real(q), jnp.imag(q)], axis=1), "im")
        SQ = 1j * (Sp[:, :r] + 1j * Sp[:, r:])
    else:
        SQ = _column_block_product(jac, q, _PARTS[apply_mode])
    if has_diag:
        SQ = SQ + diag[:, None] * q + diag_shift * q

    return SQ


@partial(jax.jit, static_argnames=("layout", "apply_mode", "has_diag"))
def _apply_rows(jac, Q, diag, diag_shift, *, layout: ParameterLayout, apply_mode, has_diag):
    """The operator applied to the row-sharded block ``Q``; the result is row-sharded."""
    return jax.shard_map(
        partial(_local_apply_rows, apply_mode=apply_mode, has_diag=has_diag),
        mesh=layout.mesh,
        in_specs=(JACOBIAN_SPEC, ROWS_SPEC, ROWS_SPEC, REPLICATED_SPEC),
        out_specs=ROWS_SPEC,
    )(jac, Q, diag, diag_shift)


@partial(jax.jit, static_argnames=("layout", "real_dtype", "total_rank", "complex_probe"))
def _probe_rows(key, *, layout: ParameterLayout, real_dtype, total_rank, complex_probe):
    """
    The random probe block, row-sharded: every parameter group draws its own rows from
    the broadcast ``key`` folded with its index (so the sample groups agree), the
    padding rows are zero and stay zero in every iterate.
    """
    def local(key):
        group = jax.lax.axis_index(PARAMS_AXIS)
        key = random.fold_in(key, group)
        shape = (layout.local, total_rank)
        if complex_probe:
            key_re, key_im = random.split(key)
            Omega = random.normal(key_re, shape, dtype=real_dtype) + 1j * random.normal(key_im, shape, dtype=real_dtype)
        else:
            Omega = random.normal(key, shape, dtype=real_dtype)
        rows = group * layout.local + jnp.arange(layout.local)

        return jnp.where((rows < layout.num_params)[:, None], Omega, jnp.zeros((), Omega.dtype))

    return jax.shard_map(local, mesh=layout.mesh, in_specs=(REPLICATED_SPEC,), out_specs=ROWS_SPEC)(key)


@partial(jax.jit, static_argnames=("layout",))
def _inner_rows(X, Y, *, layout: ParameterLayout):
    """``X^H Y`` for row-sharded ``X`` and ``Y``: local products, ``psum`` over the parameter groups; replicated."""
    return jax.shard_map(
        lambda x, y: jax.lax.psum(jnp.conj(x).T @ y, PARAMS_AXIS),
        mesh=layout.mesh,
        in_specs=(ROWS_SPEC, ROWS_SPEC),
        out_specs=REPLICATED_SPEC,
    )(X, Y)


@jax.jit
def _cholesky(G, rel_shift):
    """
    Lower Cholesky factor ``L`` of ``G + s I`` with ``s = rel_shift * ||G||_F`` (an
    upper bound of ``rel_shift * ||G||_2``), and whether it is finite (a failed
    factorization is NaN).
    """
    shift = rel_shift * jnp.linalg.norm(G)
    idx = jnp.arange(G.shape[0])
    L = jnp.linalg.cholesky(G.at[idx, idx].add(shift.astype(G.dtype)))

    return L, jnp.all(jnp.isfinite(L))


@partial(jax.jit, static_argnames=("layout",), donate_argnums=0)
def _solve_rows(Y, L, *, layout: ParameterLayout):
    """``Y L^{-H}``, i.e. ``Q`` of ``Y = Q R`` with ``R = L^H``, row by row; ``Y`` is donated."""
    return jax.shard_map(
        lambda y, l: jax.lax.linalg.triangular_solve(
            l, y, left_side=False, lower=True, transpose_a=True, conjugate_a=True
        ),
        mesh=layout.mesh,
        in_specs=(ROWS_SPEC, REPLICATED_SPEC),
        out_specs=ROWS_SPEC,
    )(Y, L)


@jax.jit
def _orthogonality_error(G):
    return jnp.linalg.norm(G - jnp.eye(G.shape[0], dtype=G.dtype))


# CholeskyQR passes per orthonormalization: shifted ones are added while the Gram
# matrix is too ill-conditioned for a plain Cholesky factorization
_MAX_QR_PASSES = 6


def _cholesky_qr(Y, layout: ParameterLayout, name="QR"):
    """
    ``Q`` of the reduced QR of the row-sharded ``Y`` (donated) by shifted CholeskyQR3
    (Fukaya, Kannan, Nakatsukasa, Yamamoto, Yanagisawa, SIAM J. Sci. Comput. 42,
    A477 (2020)), without ever gathering ``Y``:

    1. Gram matrix ``G = Y^H Y``: local ``r x r`` products, ``psum`` over the
       parameter groups;
    2. Cholesky ``L L^H = G + s I`` with the shift ``s = 11 (m r + r (r + 1)) u ||Y||_2^2``
       (``m`` rows, ``u`` the unit round-off; ``||G||_F`` stands in for
       ``||Y||_2^2 = ||G||_2``, an upper bound), which makes the factorization succeed
       for any condition number up to ``O(1/u)``;
    3. ``Q = Y L^{-H}``, a local triangular solve;
    4. two plain CholeskyQR passes (CholeskyQR2) on ``Q``, which restore orthogonality
       to ``O(u)``.

    A plain pass whose Cholesky factorization fails (``G`` numerically indefinite: the
    shifted pass left ``Q`` with condition number above ``u^{-1/2}``, e.g. for a
    numerically rank-deficient ``Y``) is redone shifted, which shrinks the condition
    number further, up to ``_MAX_QR_PASSES`` passes in total. Every pass costs two
    ``(m / P_p) x r x r`` products per device and an ``r x r`` Cholesky factorization.
    """
    m, r = Y.shape
    rel_shift = 11 * (m * r + r * (r + 1)) * float(jnp.finfo(Y.dtype).eps) / 2

    L, _ = _cholesky(_inner_rows(Y, Y, layout=layout), rel_shift)
    Q = _solve_rows(Y, L, layout=layout)
    n_passes, n_plain = 1, 0
    while n_plain < 2 and n_passes < _MAX_QR_PASSES:
        G = _inner_rows(Q, Q, layout=layout)
        L, finite = _cholesky(G, 0.0)
        if bool(finite):
            n_plain += 1
        else:
            L, _ = _cholesky(G, rel_shift)
        del G
        Q = _solve_rows(Q, L, layout=layout)
        n_passes += 1
    if n_plain < 2 and jax.process_index() == 0:
        warnings.warn(
            f"CholeskyQR stopped after {n_passes} passes without two plain ones: the block is "
            "numerically rank deficient and its orthonormalization may be inaccurate.",
            RuntimeWarning
        )
    if _DEBUG:
        error = float(_orthogonality_error(_inner_rows(Q, Q, layout=layout)))
        _debug_stage(f"{name} ({n_passes} CholeskyQR passes, ||Q^H Q - I||_F = {error:.2e})", Q)
    else:
        _debug_stage(name, Q)

    return Q


def _jaxmg_fallback_reason():
    """Why jaxmg cannot be used in this run (``None`` if it can), as in :func:`jVMC_exp.solver.util.diagonalize`."""
    if solver_util.jaxmg is None:
        return "the 'jaxmg' package is not installed"
    if jax.default_backend() != "gpu":
        return f"it needs a GPU backend, got {jax.default_backend()!r}"
    if MESH.size == 1:
        return "there is only one device"
    if not global_defs.USE_DISTRIBUTED:
        return "it needs JVMC_USE_DISTRIBUTED=true and one process per GPU"

    return None


def _warn_fallback(what, reason, fallback):
    if jax.process_index() == 0:
        warnings.warn(f"{what} is not available: {reason}. Falling back to {fallback}.")


def _resolve_qr_mode(qr_mode, layout: ParameterLayout):
    if qr_mode != "distributed":
        return qr_mode
    reason = _jaxmg_fallback_reason()
    if reason is None and layout.sample_groups > 1:
        reason = "jaxmg's process grid has to cover the whole mesh, so it needs a (1, P) mesh"
    if reason is not None:
        _warn_fallback('qr_mode="distributed" (jaxmg QR)', reason, 'qr_mode="cholesky"')
        return "cholesky"

    return qr_mode


def _default_tile(*local_sizes):
    """A cuSOLVERMp tile that divides the local sizes, or 256 (jaxmg pads) if their gcd is too small to be efficient."""
    tile = math.gcd(*local_sizes)

    return tile if tile >= 128 else 256


@partial(jax.jit, static_argnames=("layout",))
def _zero_padding_rows(Q, *, layout: ParameterLayout):
    def local(q):
        rows = jax.lax.axis_index(PARAMS_AXIS) * layout.local + jnp.arange(layout.local)
        return jnp.where((rows < layout.num_params)[:, None], q, jnp.zeros((), q.dtype))

    return jax.shard_map(local, mesh=layout.mesh, in_specs=(ROWS_SPEC,), out_specs=ROWS_SPEC)(Q)


def _distributed_qr(Y, layout: ParameterLayout, T_A, name="QR"):
    """
    ``Q`` of the reduced QR of the row-sharded ``Y`` (donated) by jaxmg's distributed
    (cuSOLVERMp) QR, which keeps the row sharding: nothing ``r x r`` is replicated (the
    triangular factor comes back row-sharded and is dropped). Needs a ``(1, P)`` mesh
    (see :func:`_resolve_qr_mode`). Householder QR may put round-off into the padding
    rows of a rank-deficient block; they are zeroed again.
    """
    tile = T_A or _default_tile(layout.local, Y.shape[1])
    Q, R = solver_util.jaxmg.qr(Y, tile, layout.mesh, matrix_specs=P(PARAMS_AXIS, None))
    del R
    if layout.pad:
        Q = _zero_padding_rows(Q, layout=layout)
    if _DEBUG:
        error = float(_orthogonality_error(_inner_rows(Q, Q, layout=layout)))
        _debug_stage(f"{name} (jaxmg QR, T_A = {tile}, ||Q^H Q - I||_F = {error:.2e})", Q)
    else:
        _debug_stage(name, Q)

    return Q


def _orthonormalize_rows(Y, layout: ParameterLayout, *, qr_mode, T_A, name):
    if qr_mode == "distributed":
        return _distributed_qr(Y, layout, T_A, name=name)

    return _cholesky_qr(Y, layout, name=name)


# The r x r objects of the parameters layout (Rayleigh-Ritz matrix B, its eigenvectors
# W) are split in blocks: rows over the parameter groups, columns over the sample
# groups, so that device (s, p) holds the block (p, s) of r / P_p x r / P_s entries.
# This is the layout jaxmg takes (one mesh axis per matrix dimension, covering the
# whole mesh); for a (1, P) mesh it is the plain row sharding over all devices.
BLOCK_SPEC = P(PARAMS_AXIS, SAMPLES_AXIS)


def _block_rank(total_rank, layout: ParameterLayout):
    """The rank padded to a multiple of the number of devices (hence of P_p and P_s) for the block layout."""
    n_devices = layout.mesh.size

    return -(-total_rank // n_devices) * n_devices


@partial(jax.jit, static_argnames=("layout", "width"), donate_argnums=0)
def _pad_columns(Q, *, layout: ParameterLayout, width):
    return jax.shard_map(
        lambda q: jnp.pad(q, ((0, 0), (0, width - q.shape[1]))), mesh=layout.mesh,
        in_specs=(ROWS_SPEC,), out_specs=ROWS_SPEC
    )(Q)


def _ring_permutation(n):
    return [(i, (i + 1) % n) for i in range(n)]


@partial(jax.jit, static_argnames=("layout", "rank", "restore_i"))
def _ritz_matrix_blocks(Q, SQ, *, layout: ParameterLayout, rank, restore_i):
    """
    The Rayleigh-Ritz matrix ``B = Q^H (A Q)`` of the row-sharded ``Q`` and ``A Q``
    (``r_b`` columns, ``r_b`` a multiple of the device count, the columns from ``rank``
    on zero), in the block layout ``BLOCK_SPEC``: device ``(s, p)`` ends up with
    ``B[rows p, columns s]``. A ring reduce-scatter over the parameter groups: in step
    ``k`` every device adds its local contribution ``Q_p[:, t]^H (A Q)_p[:, s]`` to the
    block ``t = p - k - 1`` it received from its neighbour and passes it on, so that
    at most two blocks of ``r_b / P_p x r_b / P_s`` live at once and no ``r x r``
    partial sum is formed. No psum over the sample groups is needed: ``Q`` and ``A Q``
    are the same in all of them.

    ``B`` is Hermitian up to round-off and is not symmetrized here (the eigensolvers
    read one triangle; the gathered path symmetrizes). With ``restore_i`` the factor
    ``i`` of the ``"imag_real"`` iteration is restored. The padding is decoupled as in
    :func:`jVMC_exp.solver.util.diagonalize`: ``2 ||B||_F + 1``, more than the spectral
    radius, on its diagonal, so that its eigenpairs come last in the ascending spectrum.
    """
    r_b = Q.shape[1]
    n_groups = layout.groups
    rows, cols = r_b // n_groups, r_b // layout.sample_groups
    perm = _ring_permutation(n_groups)

    def local(q, sq):
        p = jax.lax.axis_index(PARAMS_AXIS)
        s = jax.lax.axis_index(SAMPLES_AXIS)
        sq_cols = jax.lax.dynamic_slice_in_dim(sq, s * cols, cols, axis=1)

        def contribution(t):
            q_rows = jax.lax.dynamic_slice_in_dim(q, t * rows, rows, axis=1)
            return jnp.conj(q_rows).T @ sq_cols

        def step(k, acc):
            return jax.lax.ppermute(acc, PARAMS_AXIS, perm) + contribution((p - k - 1) % n_groups)

        acc = jax.lax.fori_loop(1, n_groups, step, contribution((p - 1) % n_groups))
        if restore_i:
            acc = 1j * acc
        if r_b > rank:
            norm = jnp.sqrt(jax.lax.psum(jnp.sum(jnp.abs(acc) ** 2), (PARAMS_AXIS, SAMPLES_AXIS)))
            i = p * rows + jnp.arange(rows)[:, None]
            j = s * cols + jnp.arange(cols)[None, :]
            acc = acc + jnp.where((i == j) & (i >= rank), 2 * norm + 1, 0.).astype(acc.dtype)

        return acc

    return jax.shard_map(
        local, mesh=layout.mesh, in_specs=(ROWS_SPEC, ROWS_SPEC), out_specs=BLOCK_SPEC
    )(Q, SQ)


@partial(jax.jit, static_argnames=("layout",))
def _gathered_ritz_matrix(B, *, layout: ParameterLayout):
    """The block-sharded ``B`` replicated and symmetrized, for the replicated eigensolvers."""
    B = jax.lax.with_sharding_constraint(B, layout.sharding(REPLICATED_SPEC))

    return jax.lax.with_sharding_constraint(0.5 * (B + jnp.conj(B).T), layout.sharding(REPLICATED_SPEC))


def _diagonalize_blocks(B, layout: ParameterLayout, *, mode, T_A):
    """
    Eigenvalues (ascending, replicated) and eigenvectors ``W`` (``BLOCK_SPEC``) of the
    block-sharded ``B``, and the backend used. ``"distributed"``: jaxmg's ``syevd``
    directly on the blocks (``B`` is donated), ``W`` comes back in the same blocks,
    nothing ``r x r`` is replicated; falls back to ``"device"`` with a warning where
    jaxmg cannot be used. ``"device"`` / ``"host"``: ``B`` gathered and diagonalized
    replicated by :func:`jVMC_exp.solver.util.diagonalize`, ``W`` re-sharded into the
    blocks (a local slice).
    """
    if mode == "distributed":
        reason = _jaxmg_fallback_reason()
        if reason is not None:
            _warn_fallback('diagonalization_mode="distributed" (jaxmg syevd)', reason, '"device"')
            mode = "device"
    if mode == "distributed":
        r_b = B.shape[0]
        tile = T_A or _default_tile(r_b // layout.groups, r_b // layout.sample_groups)
        ev, W = solver_util.jaxmg.syevd(B, tile, layout.mesh, matrix_specs=BLOCK_SPEC)
        return ev, W, f"distributed, T_A = {tile}"

    B = _gathered_ritz_matrix(B, layout=layout)
    ev, W = diagonalize(B, pad_size=0, mode=mode)
    del B
    W = jax.device_put(W, layout.sharding(BLOCK_SPEC))

    return ev, W, mode


@partial(jax.jit, static_argnames=("layout",))
def _ritz_vectors_blocks(Q, W, *, layout: ParameterLayout):
    """
    The Ritz vectors ``V = Q W`` for the row-sharded ``Q`` and the block-sharded ``W``,
    row-sharded like ``Q``, without replicating ``W``: on device ``(s, p)``,
    ``V_p[:, s] = sum_k Q_p[:, k] W[k, s]`` over the row blocks ``k`` of ``W``, which
    travel around the ring of parameter groups (one ``ppermute`` per step, at most two
    blocks alive). With several sample groups the column blocks of ``V`` are gathered
    at the end.
    """
    r_b = Q.shape[1]
    n_groups = layout.groups
    rows = r_b // n_groups
    perm = _ring_permutation(n_groups)

    def local(q, w):
        p = jax.lax.axis_index(PARAMS_AXIS)

        def contribution(k, w):
            # after k steps this device holds the block of group p - k
            q_cols = jax.lax.dynamic_slice_in_dim(q, ((p - k) % n_groups) * rows, rows, axis=1)
            return q_cols @ w

        def step(k, carry):
            acc, w = carry
            w = jax.lax.ppermute(w, PARAMS_AXIS, perm)
            return acc + contribution(k, w), w

        acc, _ = jax.lax.fori_loop(1, n_groups, step, (contribution(0, w), w))
        return acc

    V = jax.shard_map(
        local, mesh=layout.mesh, in_specs=(ROWS_SPEC, BLOCK_SPEC), out_specs=BLOCK_SPEC
    )(Q, W)

    return jax.lax.with_sharding_constraint(V, layout.sharding(ROWS_SPEC))


@partial(jax.jit, static_argnames=("layout",))
def _select_rows(ev, V, idx, *, layout: ParameterLayout):
    """``ev[idx]`` and the columns ``idx`` of the row-sharded ``V``, locally."""
    V = jax.shard_map(
        lambda v, i: v[:, i], mesh=layout.mesh,
        in_specs=(ROWS_SPEC, REPLICATED_SPEC), out_specs=ROWS_SPEC
    )(V, idx)

    return ev[idx], V


@partial(jax.jit, static_argnames=("layout",))
def _project_rows(V, b, *, layout: ParameterLayout):
    """``V^H b`` for the row-sharded (padded) ``V`` and a replicated ``(num_params,)`` ``b``; replicated."""
    return jax.shard_map(
        lambda v, b: jax.lax.psum(jnp.conj(v).T @ b, PARAMS_AXIS), mesh=layout.mesh,
        in_specs=(ROWS_SPEC, ROWS_SPEC), out_specs=REPLICATED_SPEC
    )(V, jnp.pad(b, (0, layout.pad)))


@partial(jax.jit, static_argnames=("layout",))
def _gather_rows(x, *, layout: ParameterLayout):
    """The row-sharded, padded vector ``x`` replicated (all-gathered) and stripped of the padding."""
    x = jax.lax.with_sharding_constraint(x, layout.sharding(REPLICATED_SPEC))

    return x[:layout.num_params]


def _subspace_iteration_rows(operator: ShardedQGTOperator, key, *, total_rank, n_iter, apply_mode,
                             qr_mode, T_A):
    """
    :func:`_subspace_iteration` with the blocks sharded over the parameter rows of the
    operator's 2D mesh: ``(padded_num_params / P_p, r)`` per device. Probe, iterates
    and ``Q`` never leave this layout; orthonormalization by :func:`_cholesky_qr` or
    :func:`_distributed_qr` (``qr_mode``, already resolved). The final ``Q`` is padded
    with zero columns to a multiple of the device count (:func:`_block_rank`, a few
    columns) and the Rayleigh-Ritz matrix ``B = Q^H A Q`` is formed block-sharded by
    :func:`_ritz_matrix_blocks`.
    """
    layout = operator.layout
    jac, diag, shift = operator.jacobian, operator._diag_rows, operator._shift
    kwargs = dict(layout=layout, apply_mode=apply_mode, has_diag=operator.has_diagonal_terms)
    orthonormalize = partial(_orthonormalize_rows, layout=layout, qr_mode=qr_mode, T_A=T_A)

    _debug_stage("start of the subspace iteration")
    Omega = _probe_rows(
        key, layout=layout, real_dtype=jnp.finfo(jac.dtype).dtype, total_rank=total_rank,
        complex_probe=apply_mode not in ("real", "imag_real")
    )
    _debug_stage("probe", Omega)
    Y = _apply_rows(jac, Omega, diag, shift, **kwargs)
    del Omega
    _debug_stage("application 1", Y)
    for it in range(2 * n_iter):
        Q = orthonormalize(Y, name=f"QR {it + 1}")
        Y = _apply_rows(jac, Q, diag, shift, **kwargs)
        del Q
        _debug_stage(f"application {it + 2}", Y)
    Q = orthonormalize(Y, name=f"QR {2 * n_iter + 1}")
    del Y
    r_b = _block_rank(total_rank, layout)
    if r_b > total_rank:
        Q = _pad_columns(Q, layout=layout, width=r_b)
    SQ = _apply_rows(jac, Q, diag, shift, **kwargs)
    _debug_stage(f"application {2 * n_iter + 2}", SQ)
    B = _ritz_matrix_blocks(Q, SQ, layout=layout, rank=total_rank, restore_i=apply_mode == "imag_real")
    del SQ
    _debug_stage("Rayleigh-Ritz matrix (block-sharded)", B)

    return Q, B


def _randomized_eigh_rows(operator: ShardedQGTOperator, k, key, *, total_rank, n_iter, which,
                          diagonalization_mode, T_A, qr_mode="cholesky"):
    """
    :func:`randomized_eigh` for a :class:`ShardedQGTOperator`. ``V`` is row-sharded and
    padded; the Rayleigh-Ritz matrix and its eigenvectors stay block-sharded
    (:func:`_diagonalize_blocks`, :func:`_ritz_vectors_blocks`).
    """
    layout = operator.layout
    Q, B = _subspace_iteration_rows(
        operator, key, total_rank=total_rank, n_iter=n_iter, apply_mode=_apply_mode(operator),
        qr_mode=_resolve_qr_mode(qr_mode, layout), T_A=T_A
    )

    ev, W, backend = _diagonalize_blocks(B, layout, mode=diagonalization_mode, T_A=T_A)
    del B
    _debug_stage(f"Rayleigh-Ritz diagonalization ({backend})", ev, W)

    V = _ritz_vectors_blocks(Q, W, layout=layout)
    del Q, W
    _debug_stage("Ritz vectors", V)

    # The padded eigenpairs are the last ones of the ascending spectrum
    r_b = ev.shape[0]
    ev_host = np.asarray(ev)[:total_rank]
    idx = np.argsort(np.abs(ev_host))[::-1][:k] if which == "LM" else np.argsort(ev_host)[::-1][:k]
    idx = idx[np.argsort(ev_host[idx])]
    if r_b == total_rank and k == total_rank and np.array_equal(idx, np.arange(total_rank)):
        return ev, V

    ev, V = _select_rows(ev, V, jnp.asarray(idx), layout=layout)
    _debug_stage("selection", ev, V)

    return ev, V


def _pack(x, complex_solution):
    """Real optimization variables for the refinement: ``x`` itself, or ``[Re x, Im x]``."""
    if complex_solution:
        return jnp.concatenate([jnp.real(x), jnp.imag(x)])

    return jnp.real(x)


def _unpack(z, complex_solution):
    if complex_solution:
        n = z.shape[0] // 2
        return z[:n] + 1j * z[n:]

    return z


def _refinement_loss(z, jac, diag, shift, b, b_norm, *, mode, has_diag, complex_solution, layout=None):
    """
    The true residual ``||A x - b|| / ||b||`` as a function of the packed ``x``, for the
    sample-sharded Jacobian (``layout=None``) or the column-sharded one of a
    :class:`ShardedQGTOperator`.
    """
    x = _unpack(z, complex_solution)
    Ax = _apply(jac, x, diag, shift, mode=mode, has_diag=has_diag, layout=layout)

    return jnp.linalg.norm(Ax - b) / b_norm


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

    With ``layout="parameters"`` the operator is turned into a
    :class:`jVMC_exp.solver.qgt_operator.ShardedQGTOperator` at every solve (one
    ``all_to_all`` of the Jacobian into a second, column-sharded copy, released after
    the solve) and the subspace iteration keeps all ``(num_params, r)`` blocks, the
    Ritz vectors included, sharded over the parameter rows of a ``(P_s, P_p)`` mesh,
    so that their memory per device is divided by ``P_p``. ``V^H b`` is a ``psum`` of
    a ``k``-vector and the update ``V (pinvEv * V^H b)`` is all-gathered into the
    replicated vector the optimizer expects. The ``r x r`` Rayleigh-Ritz matrix and
    its eigenvectors are block-sharded; with ``diagonalization_mode="distributed"``
    and ``qr_mode="distributed"`` no ``r x r`` object is replicated. The true residual and the L-BFGS
    refinement use the sharded operator's vector products.

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
    layout : {"samples", "parameters"}, default="samples"
        Layout of the ``(num_params, r)`` blocks: replicated on every device
        (``"samples"``, only the Jacobian is sharded, over the samples) or sharded over
        the parameter rows (``"parameters"``, see above). In the ``"parameters"``
        layout the empirical SNR (``snr_mode="empiric"`` with ``snr_tol != 0``) is not
        implemented (it needs the per-sample projections onto the replicated
        eigenvectors); ``snr_mode="gaussian"`` is.
    mesh_shape : tuple (P_s, P_p), optional
        Mesh of sample groups times parameter groups for ``layout="parameters"``,
        with ``P_s * P_p`` the number of devices; default ``(1, number of devices)``,
        pure parameter sharding. See
        :func:`jVMC_exp.solver.qgt_operator.parameter_mesh`.
    qr_mode : {"cholesky", "distributed"}, default="cholesky"
        Orthonormalization in the ``"parameters"`` layout: shifted CholeskyQR3 (its
        ``r x r`` Gram matrix and Cholesky factor are replicated) or jaxmg's
        distributed QR (nothing ``r x r`` replicated; needs jaxmg, a GPU backend, one
        process per GPU and a ``(1, P)`` mesh, otherwise it falls back to
        ``"cholesky"`` with a warning). ``T_A`` is its tile size. See
        :func:`randomized_eigh`.
    compute_true_residual : bool, default=True
        Report ``true_residual`` in the info dictionary, at the cost of one more
        operator application per solve. Always on when ``lbfgs`` is set.
    lbfgs : bool, default=False
        Refine the truncated solution by minimizing the true residual
        ``||A x - b|| / ||b||`` with L-BFGS (``jaxopt.LBFGS``), starting from the
        randomized pseudo-inverse update, whenever that residual exceeds
        ``lbfgs_tol``. Recovers the force that the truncation to ``n_components``
        discarded, at the cost of two operator applications per L-BFGS iteration.
        For a non-holomorphic ansatz the refinement runs over real updates, for a
        holomorphic one over real and imaginary part. Needs the ``jaxopt`` package.
    lbfgs_tol : float, default=1e-7
        Target for the true residual; the refinement stops once it is reached.
    lbfgs_maxiter : int, default=1000
        Iterations per L-BFGS run.
    lbfgs_rtol : float, default=1e-3
        The L-BFGS run is restarted (warm) while the residual still improves by
        more than this relative amount per run and ``lbfgs_tol`` is not reached.
    lbfgs_max_restarts : int, default=10
        Upper bound on the number of L-BFGS runs per solve.

    stage_timings : bool, default=False
        Synchronize after every stage of the randomized decomposition and report the
        accumulated wall times in the info dictionary: ``t_redistribute`` (parameters
        layout: the all-to-all of the Jacobian), ``t_probe``, ``t_apply`` (all
        applications of the operator), ``t_qr``, ``t_ritz_matrix``, ``t_eigh``,
        ``t_ritz_vectors``, ``t_select`` and their sum ``t_rsvd``. Also on with
        ``JVMC_RSVD_DEBUG=1``.

    With ``lbfgs`` the info dictionary additionally holds ``residual_rsvd`` (the true
    residual before the refinement), ``n_lbfgs`` (runs), ``lbfgs_iterations``
    (iterations summed over the runs), ``t_lbfgs`` (wall time in seconds) and
    ``cos_lbfgs`` (overlap of the update before and after the refinement);
    ``true_residual`` is the residual after the refinement.
    """
    def __init__(
            self, n_components: int, n_oversamples: int = 0, n_iter: int = 4,
            seed=4321, which: Literal["LM", "LA"] = "LM",
            snr_tol=0, pinv_tol=1e-14, pinv_cutoff=1e-8,
            snr_mode: Literal["empiric", "gaussian"] = "empiric",
            diagonalization_mode: Literal["device", "distributed", "host"] = "device",
            T_A: int | None = None, compute_true_residual: bool = True,
            lbfgs: bool = False, lbfgs_tol: float = 1e-7, lbfgs_maxiter: int = 1000,
            lbfgs_rtol: float = 1e-3, lbfgs_max_restarts: int = 10,
            layout: Literal["samples", "parameters"] = "samples", mesh_shape=None,
            stage_timings: bool = False, qr_mode: Literal["cholesky", "distributed"] = "cholesky"
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
        self._compute_true_residual = bool(compute_true_residual) or bool(lbfgs)
        self._stage_timings = bool(stage_timings)
        self._last_stage_timings = {}
        self._key = seed if isinstance(seed, jax.Array) else random.PRNGKey(seed)

        if int(lbfgs_maxiter) <= 0 or int(lbfgs_max_restarts) <= 0:
            raise ValueError("lbfgs_maxiter and lbfgs_max_restarts have to be positive.")
        self._lbfgs = bool(lbfgs)
        self._lbfgs_tol = float(lbfgs_tol)
        self._lbfgs_maxiter = int(lbfgs_maxiter)
        self._lbfgs_rtol = float(lbfgs_rtol)
        self._lbfgs_max_restarts = int(lbfgs_max_restarts)
        self._lbfgs_solvers = {}

        if layout not in LAYOUTS:
            raise ValueError(f"layout has to be one of {LAYOUTS}, got {layout!r}.")
        if qr_mode not in QR_MODES:
            raise ValueError(f"qr_mode has to be one of {QR_MODES}, got {qr_mode!r}.")
        self._qr_mode = qr_mode
        self._layout = layout
        self._mesh_shape = None
        if layout == "parameters":
            self._mesh_shape = parameter_mesh(mesh_shape).devices.shape
        elif mesh_shape is not None:
            raise ValueError('mesh_shape is only used with layout="parameters".')

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
    def lbfgs(self):
        return self._lbfgs

    @property
    def lbfgs_tol(self):
        return self._lbfgs_tol

    @property
    def layout(self):
        return self._layout

    @property
    def qr_mode(self):
        return self._qr_mode

    @property
    def mesh_shape(self):
        """``(P_s, P_p)`` of the ``"parameters"`` layout, ``None`` for ``"samples"``."""
        return self._mesh_shape

    @property
    def _needs_dense_matrix(self) -> bool:
        return False

    def _lbfgs_solver(self, mode, has_diag, complex_solution, layout=None):
        """One ``jaxopt.LBFGS`` per operator type, so that its compiled loop is reused."""
        key = (mode, has_diag, complex_solution, layout)
        if key not in self._lbfgs_solvers:
            try:
                import jaxopt
            except ImportError as err:
                raise ImportError(
                    f"{type(self).__name__}(lbfgs=True) needs the 'jaxopt' package "
                    "(`pip install jaxopt`)."
                ) from err

            self._lbfgs_solvers[key] = jaxopt.LBFGS(
                fun=partial(
                    _refinement_loss, mode=mode, has_diag=has_diag,
                    complex_solution=complex_solution, layout=layout
                ),
                tol=1e-10, maxiter=self._lbfgs_maxiter
            )

        return self._lbfgs_solvers[key]

    def _refine(self, A: QGTOperator | ShardedQGTOperator, b, x, residual):
        """
        L-BFGS minimization of the true residual, warm-started from the randomized
        solution ``x`` with true residual ``residual``.
        """
        info = dict(residual_rsvd=residual, n_lbfgs=0, lbfgs_iterations=0, t_lbfgs=0.0, cos_lbfgs=1.0)
        if not residual > self._lbfgs_tol:
            return x, residual, info

        complex_solution = A.mode == "identity"
        layout = A.layout if isinstance(A, ShardedQGTOperator) else None
        solver = self._lbfgs_solver(A.mode, A.has_diagonal_terms, complex_solution, layout)
        args = (A.jacobian, A._diag, A._shift, b, jnp.linalg.norm(b))

        tic = time.perf_counter()
        z = _pack(x, complex_solution)
        for _ in range(self._lbfgs_max_restarts):
            z, state = solver.run(z, *args)
            new_residual = float(state.value)
            info["n_lbfgs"] += 1
            info["lbfgs_iterations"] += int(state.iter_num)
            improvement = (residual - new_residual) / residual
            residual = min(residual, new_residual)
            if new_residual <= self._lbfgs_tol or improvement < self._lbfgs_rtol:
                break
        jax.block_until_ready(z)
        info["t_lbfgs"] = time.perf_counter() - tic

        x_new = _unpack(z, complex_solution)
        norms = jnp.linalg.norm(x_new) * jnp.linalg.norm(x)
        info["cos_lbfgs"] = float(jnp.real(jnp.vdot(x_new, x)) / norms) if norms > 0 else 1.0

        return x_new, residual, info

    def _next_key(self):
        self._key, subkey = broadcast_split_key(self._key, 2)

        return subkey

    def _check_operator(self, A):
        if not isinstance(A, (QGTOperator, ShardedQGTOperator)):
            raise TypeError(
                f"{type(self).__name__} is matrix-free and expects the left hand side as a "
                f"jVMC_exp.solver.qgt_operator.QGTOperator, got {type(A).__name__}. Use "
                "PinvSNR for a dense matrix."
            )

    def _diagonalize(self, A: QGTOperator | ShardedQGTOperator, b, pad_size):
        self._check_operator(A)

        _begin_stages(self._stage_timings)
        ev, V = randomized_eigh(
            A, self.n_components, self._next_key(),
            n_oversamples=self.n_oversamples, n_iter=self.n_iter, which=self.which,
            diagonalization_mode=self._diagonalization_mode, T_A=self._T_A,
            qr_mode=self._qr_mode
        )
        self._last_stage_timings.update(_stage_timings())
        if jnp.max(jnp.abs(ev)) < 1e-14:
            raise RuntimeError(
                f"Largest eigenvalue of the matrix A to invert is {jnp.max(jnp.abs(ev))}. "
                "A is most likely highly ill-conditioned/zero "
            )
        if isinstance(A, ShardedQGTOperator):
            Vtb = _project_rows(V, b, layout=A.layout)
        else:
            Vtb = jnp.dot(jnp.transpose(jnp.conj(V)), b)

        return ev, V, Vtb

    def __call__(self, A: QGTOperator | ShardedQGTOperator, b, *, pad_size=0, **kwargs):
        """
        Solve ``A @ x = b`` in the truncated eigenbasis of ``A``. ``pad_size`` is
        accepted for interface compatibility and ignored: the operator acts on the
        unpadded parameter space.

        In the ``"parameters"`` layout a :class:`QGTOperator` is redistributed into a
        :class:`ShardedQGTOperator` for this solve (a given ``ShardedQGTOperator`` is
        used as it is); the returned update is replicated in either layout.
        """
        self._check_operator(A)
        sharded = self._layout == "parameters" or isinstance(A, ShardedQGTOperator)
        if sharded and self.snr_tol and self.snr_mode == "empiric" and not kwargs.get("exact_sampler", False):
            raise NotImplementedError(
                'The empirical SNR (snr_mode="empiric") is not implemented for '
                'layout="parameters"; use snr_mode="gaussian", snr_tol=0 or '
                'layout="samples".'
            )
        self._last_stage_timings = {}
        if isinstance(A, QGTOperator) and self._layout == "parameters":
            tic = time.perf_counter()
            A = ShardedQGTOperator.from_operator(A, self._mesh_shape)
            if self._stage_timings or _DEBUG:
                jax.block_until_ready(A.jacobian)
                self._last_stage_timings["redistribute"] = time.perf_counter() - tic

        x, info = super().__call__(A, b, pad_size=0, **kwargs)
        if self._stage_timings or _DEBUG:
            info.update({f"t_{key}": value for key, value in self._last_stage_timings.items()})
            info["t_rsvd"] = sum(self._last_stage_timings.values())
        if isinstance(A, ShardedQGTOperator):
            # PinvSNR's V (pinvEv * V^H b) is row-sharded and padded like V
            x = _gather_rows(x, layout=A.layout)

        info["n_components"] = self.n_components
        if self._compute_true_residual:
            residual = (jnp.linalg.norm(A(x) - b) / jnp.linalg.norm(b)).item()
            if self._lbfgs:
                x, residual, lbfgs_info = self._refine(A, b, x, residual)
                info.update(lbfgs_info)
            info["true_residual"] = residual

        return x, info
