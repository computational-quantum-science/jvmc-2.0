"""
Matrix-free left hand side of the TDVP / SR equation.

The quantum geometric tensor :math:`S_0 = J^\\dagger J` is never formed: products
with blocks of vectors are computed from the (mean-subtracted, :math:`\\sqrt{w}`
weighted) Jacobian :math:`J` of shape ``(num_samples, num_params)``, which is sharded
across the devices along the sample axis. Each device contracts its own samples
and the partial sums are reduced with a ``psum``, so the only replicated arrays are
the ``(num_params, r)`` blocks that go in and out.
"""
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax.custom_derivatives import linear_call

from jVMC_exp import global_defs
from jVMC_exp.sharding_config import MESH, DEVICE_SPEC, REPLICATED_SPEC, DEVICE_SHARDING, REPLICATED_SHARDING

TransformMode = Literal["identity", "real", "imag"]
TRANSFORM_MODES = ("identity", "real", "imag")


def _local_product(jac, v):
    """
    ``conj(jac).T @ (jac @ v)`` over the samples of this device, summed over chunks of
    at most ``global_defs.MATMUL_CHUNK_BYTES`` of the Jacobian: XLA autotunes a matrix
    product on its own copies of the operands, which would otherwise be as large as
    the Jacobian, and the intermediate ``jac @ v`` is bounded in the same way.
    """
    n_samples, n_params = jac.shape
    row_bytes = n_params * jac.dtype.itemsize
    chunk = max(1, min(n_samples, (global_defs.MATMUL_CHUNK_BYTES or n_samples * row_bytes) // row_bytes))

    out_dtype = jnp.result_type(jac.dtype, v.dtype)
    acc0 = jnp.zeros((n_params,) + v.shape[1:], out_dtype)

    def add_chunk(k, acc, size=chunk):
        rows = jax.lax.dynamic_slice_in_dim(jac, k * chunk, size)
        # Materialized: XLA would otherwise fuse the slice into the product and
        # autotune it on the whole operand
        rows = jax.lax.optimization_barrier(rows)
        return acc + jnp.conj(rows).T @ (rows @ v)

    # The carry has to be 'varying' over the devices from the start (shard_map's
    # scan rule), so the first chunk seeds it instead of a plain zero block
    n_full = n_samples // chunk
    acc = jax.lax.fori_loop(1, n_full, add_chunk, add_chunk(0, acc0))
    if n_samples % chunk:
        acc = add_chunk(n_full, acc, n_samples % chunk)

    return acc


@jax.jit
def _qgt_product_impl(jac, v):
    product = jax.shard_map(
        lambda j, x: jax.lax.psum(_local_product(j, x), "devices"),
        mesh=MESH,
        in_specs=(DEVICE_SPEC, REPLICATED_SPEC),
        out_specs=REPLICATED_SPEC,
    )

    return product(jac, v)


def _qgt_product_transpose(jac, ct):
    # S_0 = conj(J).T @ J is Hermitian, so its transpose is its complex conjugate
    return jnp.conj(_qgt_product_impl(jac, jnp.conj(ct)))


@jax.jit
def _qgt_product(jac, v):
    """
    ``conj(jac).T @ (jac @ v)`` for a Jacobian sharded over samples and a replicated
    ``v`` of shape ``(num_params,)`` or ``(num_params, r)``. The result is replicated.

    The transpose is given explicitly (``linear_call``): the chunked loop over the
    samples is not transposable by JAX itself, and ``jax.scipy.sparse.linalg.cg``
    transposes the operator it is given.
    """
    return linear_call(_qgt_product_impl, _qgt_product_transpose, jac, v)


@jax.jit
def _qgt_diagonal(jac):
    """``diag(conj(jac).T @ jac)``, replicated."""
    diag = jax.shard_map(
        lambda j: jax.lax.psum(jnp.sum(jnp.abs(j) ** 2, axis=0), "devices"),
        mesh=MESH,
        in_specs=(DEVICE_SPEC,),
        out_specs=REPLICATED_SPEC,
    )

    return diag(jac)


def _transform(x, mode: TransformMode):
    if mode == "identity":
        return x
    if mode == "real":
        return jnp.real(x)
    if mode == "imag":
        return 1j * jnp.imag(x)

    raise ValueError(f"The transformation mode has to be one of {TRANSFORM_MODES}, got {mode!r}.")


def transformed_product(jac, v, mode: TransformMode):
    """
    ``q[S_0] @ v`` for the matrix :math:`q[S_0]`, ``v`` of shape ``(num_params,)`` or
    ``(num_params, r)``. The transformations :math:`\\text{Re}` and :math:`i\\,\\text{Im}`
    are only real-linear, so ``q[S_0 v]`` equals ``q[S_0] v`` for a real ``v`` only; a
    complex ``v`` is split into real and imaginary part, which doubles the width of
    the product.
    """
    if mode == "identity" or not jnp.iscomplexobj(v):
        return _transform(_qgt_product(jac, v), mode)

    v2d = v.reshape(v.shape[0], -1)
    r = v2d.shape[1]
    parts = jnp.concatenate([jnp.real(v2d), jnp.imag(v2d)], axis=1)
    Sp = _transform(_qgt_product(jac, parts), mode)

    return (Sp[:, :r] + 1j * Sp[:, r:]).reshape(v.shape)


@partial(jax.jit, static_argnames=("mode", "has_diag"))
def _apply(jac, v, diag, diag_shift, *, mode, has_diag):
    Sv = transformed_product(jac, v, mode)
    if has_diag:
        Sv = Sv + diag.reshape(diag.shape + (1,) * (v.ndim - 1)) * v + diag_shift * v

    return Sv


class QGTOperator:
    """
    The left hand side of the TDVP / SR equation as a linear operator,

        :math:`A v = q[S_0 v] + \\rho\\, \\text{diag}(S_0)\\, v + \\delta\\, v`,

    with :math:`S_0 = J^\\dagger J` the quantum geometric tensor built from the
    Jacobian ``jacobian`` (mean-subtracted and :math:`\\sqrt{w}` weighted, shape
    ``(num_samples, num_params)``, sharded over the samples), :math:`q` the
    transformation selected by ``mode`` (``"identity"`` for a holomorphic ansatz,
    ``"real"`` for :math:`\\text{Re}`, ``"imag"`` for :math:`i\\,\\text{Im}`),
    :math:`\\rho` = ``diag_scale`` and :math:`\\delta` = ``diag_shift``.

    Calling the operator applies the matrix :math:`A` to a vector of shape
    ``(num_params,)`` or to a block of vectors of shape ``(num_params, r)``, real or
    complex (see :func:`transformed_product`); :meth:`raw` applies the bare
    :math:`S_0`. Both are jitted and shard the contraction over the samples, see
    :func:`_qgt_product`. The operator is also a valid ``A`` for
    ``jax.scipy.sparse.linalg.cg``.
    """
    def __init__(self, jacobian, mode: TransformMode, diag_scale: float = 0., diag_shift: float = 0.):
        if mode not in TRANSFORM_MODES:
            raise ValueError(f"The transformation mode has to be one of {TRANSFORM_MODES}, got {mode!r}.")
        if jacobian.ndim != 2:
            raise ValueError(
                "The Jacobian has to have shape (num_samples, num_params), "
                f"got shape {jacobian.shape}."
            )

        self._jacobian = jax.device_put(jacobian, DEVICE_SHARDING)
        self._mode = mode
        self._diag_scale = float(diag_scale)
        self._diag_shift = float(diag_shift)

        n = jacobian.shape[1]
        self._has_diag = self._diag_scale > 1e-15 or self._diag_shift > 1e-15
        if self._diag_scale > 1e-15:
            self._diag = self._diag_scale * _qgt_diagonal(self._jacobian)
        else:
            self._diag = jnp.zeros((n,), jnp.real(jacobian).dtype)
        self._diag = jax.device_put(self._diag, REPLICATED_SHARDING)
        self._shift = jnp.asarray(self._diag_shift if self._diag_shift > 1e-15 else 0., self._diag.dtype)

    @property
    def jacobian(self):
        return self._jacobian

    @property
    def mode(self) -> TransformMode:
        return self._mode

    @property
    def diag_scale(self):
        return self._diag_scale

    @property
    def diag_shift(self):
        return self._diag_shift

    @property
    def has_diagonal_terms(self) -> bool:
        return self._has_diag

    @property
    def shape(self):
        n = self._jacobian.shape[1]
        return (n, n)

    @property
    def dtype(self):
        """dtype of ``A v`` for a real ``v``."""
        return jnp.real(self._jacobian).dtype if self._mode == "real" else self._jacobian.dtype

    def raw(self, v):
        """The bare quantum geometric tensor applied to ``v``: :math:`S_0 v`."""
        return _qgt_product(self._jacobian, v)

    def __call__(self, v):
        return _apply(
            self._jacobian, v, self._diag, self._shift, mode=self._mode, has_diag=self._has_diag
        )
