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


Part = Literal["cpx", "re", "im"]


def _chunk_sizes(n_samples, n_params, width, jac_itemsize, out_itemsize):
    """
    Rows of the Jacobian per chunk and rows of the result per block, both bounded by
    ``global_defs.MATMUL_CHUNK_BYTES`` (0: no bound).
    """
    budget = global_defs.MATMUL_CHUNK_BYTES or n_samples * n_params * jac_itemsize
    chunk = max(1, min(n_samples, budget // (n_params * jac_itemsize)))
    block = max(1, min(n_params, budget // max(1, width * out_itemsize)))

    return chunk, block


# Use jax.lax.pcast / pvary for `_varying_zeros` when they exist (False only for testing the fallback)
_USE_VARYING_API = True


def _varying_zeros(shape, dtype, like):
    """
    Zeros that are 'varying' over the devices inside a ``shard_map``: the carry of a
    loop has to be, when a loop body makes it so (shard_map's rule for loop carries).
    ``jax.lax.pcast`` (``pvary`` before it) does exactly that but exists only in some
    JAX versions; otherwise the zeros are derived from ``like``, which is varying.
    """
    zeros = jnp.zeros(shape, dtype)
    if _USE_VARYING_API:
        if hasattr(jax.lax, "pcast"):
            return jax.lax.pcast(zeros, ("devices",), to="varying")
        if hasattr(jax.lax, "pvary"):
            return jax.lax.pvary(zeros, ("devices",))

    return zeros + jnp.zeros((), dtype) * jnp.real(like.reshape(-1)[0]).astype(dtype)


def _local_product(jac, v, part: Part):
    """
    Over the samples of this device, ``part`` of ``conj(jac).T @ (jac @ v)``:

    * ``"cpx"``: the complex product, for a real or complex ``v``;
    * ``"re"`` / ``"im"``: its real / imaginary part for a REAL ``v``, in real
      arithmetic. With ``jac = A + iB`` these are ``A^T (A v) + B^T (B v)`` and
      ``A^T (B v) - B^T (A v)``: a real accumulator (half the memory of the complex
      one) and two thirds of the flops of the complex product.

    The sum runs over chunks of at most ``global_defs.MATMUL_CHUNK_BYTES`` of the
    Jacobian, and the result is accumulated in blocks of parameter rows bounded by the
    same size: XLA autotunes a matrix product on its own copies of the operands,
    including the accumulator the product is fused into, so neither the operands nor
    the output of any single product may be large. (A single product with a
    ``(num_params, r)`` output was what ran out of memory at r = 16000.)
    """
    n_samples, n_params = jac.shape
    width = v.shape[1] if v.ndim == 2 else 1
    if part == "cpx":
        out_dtype = jnp.result_type(jac.dtype, v.dtype)
    else:
        out_dtype = jnp.result_type(jnp.real(jac).dtype, v.dtype)
    chunk, block = _chunk_sizes(
        n_samples, n_params, width, jac.dtype.itemsize, jnp.dtype(out_dtype).itemsize
    )

    acc = _varying_zeros((n_params,) + v.shape[1:], out_dtype, like=jac)

    def add_chunk(k, acc, size=chunk):
        rows = jax.lax.dynamic_slice_in_dim(jac, k * chunk, size)
        # Materialized: XLA would otherwise fuse the slice into the product and
        # autotune it on the whole operand
        rows = jax.lax.optimization_barrier(rows)

        if part == "cpx":
            jv = rows @ v

            def contribution(j, bsize):
                cols = jax.lax.dynamic_slice_in_dim(rows, j * block, bsize, axis=1)
                return jnp.conj(cols).T @ jv
        else:
            ar, ai = jnp.real(rows), jnp.imag(rows)
            av, bv = ar @ v, ai @ v

            def contribution(j, bsize):
                cr = jax.lax.dynamic_slice_in_dim(ar, j * block, bsize, axis=1)
                ci = jax.lax.dynamic_slice_in_dim(ai, j * block, bsize, axis=1)
                if part == "re":
                    return cr.T @ av + ci.T @ bv
                return cr.T @ bv - ci.T @ av

        def add_block(j, acc, bsize=block):
            old = jax.lax.dynamic_slice_in_dim(acc, j * block, bsize, axis=0)
            return jax.lax.dynamic_update_slice_in_dim(
                acc, old + contribution(j, bsize), j * block, axis=0
            )

        n_full = n_params // block
        acc = jax.lax.fori_loop(0, n_full, add_block, acc)
        if n_params % block:
            acc = add_block(n_full, acc, n_params % block)

        return acc

    n_full = n_samples // chunk
    acc = jax.lax.fori_loop(0, n_full, add_chunk, acc)
    if n_samples % chunk:
        acc = add_chunk(n_full, acc, n_samples % chunk)

    return acc


def _sharded_product(part: Part):
    return jax.shard_map(
        lambda j, x: jax.lax.psum(_local_product(j, x, part), "devices"),
        mesh=MESH,
        in_specs=(DEVICE_SPEC, REPLICATED_SPEC),
        out_specs=REPLICATED_SPEC,
    )


def _blocked_product(jac, v, part: Part):
    """
    The sharded product applied to column blocks of ``v``, each with an output of at
    most ``global_defs.MATMUL_CHUNK_BYTES``, written into one ``(num_params, r)``
    result. Bounds the temporaries of one application: the accumulator and the
    all-reduce buffer of a block instead of two full ``(num_params, r)`` blocks
    (12 GB each for 94640 x 16000 doubles) next to the input and the output.
    """
    product = _sharded_product(part)
    if v.ndim == 1:
        return product(jac, v)

    n, width = v.shape
    out_itemsize = jnp.dtype(
        jnp.result_type(jac.dtype, v.dtype) if part == "cpx" else jnp.result_type(jnp.real(jac).dtype, v.dtype)
    ).itemsize
    budget = global_defs.MATMUL_CHUNK_BYTES
    block = width if not budget else max(1, min(width, budget // (n * out_itemsize)))
    if block >= width:
        return product(jac, v)

    out = None
    for start in range(0, width, block):
        piece = product(jac, v[:, start:start + block])
        if out is None:
            out = jnp.zeros((n, width), piece.dtype)
        out = jax.lax.dynamic_update_slice_in_dim(out, piece, start, axis=1)

    return out


@jax.jit
def _qgt_product_impl(jac, v):
    return _blocked_product(jac, v, "cpx")


@jax.jit
def _qgt_product_re_impl(jac, v):
    return _blocked_product(jac, v, "re")


@jax.jit
def _qgt_product_im_impl(jac, v):
    return _blocked_product(jac, v, "im")


def _qgt_product_transpose(jac, ct):
    # S_0 = conj(J).T @ J is Hermitian, so its transpose is its complex conjugate
    return jnp.conj(_qgt_product_impl(jac, jnp.conj(ct)))


def _qgt_product_transpose_real(jac, ct):
    # For a real input the map is (S_0 . cast) and the cotangent of the cast is the
    # real part, as in JAX's own transpose of a real -> complex conversion
    return jnp.real(_qgt_product_transpose(jac, ct))


def _qgt_product_re_transpose(jac, ct):
    # Re(S_0) is symmetric
    return _qgt_product_re_impl(jac, ct)


def _qgt_product_im_transpose(jac, ct):
    # Im(S_0) is antisymmetric
    return -_qgt_product_im_impl(jac, ct)


@jax.jit
def _qgt_product(jac, v):
    """
    ``conj(jac).T @ (jac @ v)`` for a Jacobian sharded over samples and a replicated
    ``v`` of shape ``(num_params,)`` or ``(num_params, r)``. The result is replicated.

    The transpose is given explicitly (``linear_call``): the chunked loop over the
    samples is not transposable by JAX itself, and ``jax.scipy.sparse.linalg.cg``
    transposes the operator it is given.
    """
    transpose = _qgt_product_transpose if jnp.iscomplexobj(v) else _qgt_product_transpose_real
    return linear_call(_qgt_product_impl, transpose, jac, v)


@jax.jit
def _qgt_product_re(jac, v):
    """``Re(conj(jac).T @ jac) @ v`` for a REAL ``v``, in real arithmetic; see :func:`_qgt_product`."""
    return linear_call(_qgt_product_re_impl, _qgt_product_re_transpose, jac, v)


@jax.jit
def _qgt_product_im(jac, v):
    """``Im(conj(jac).T @ jac) @ v`` for a REAL ``v``, in real arithmetic; see :func:`_qgt_product`."""
    return linear_call(_qgt_product_im_impl, _qgt_product_im_transpose, jac, v)


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


def _transformed_product_real_input(jac, v, mode: TransformMode):
    """``q[S_0] v`` for a real ``v`` without ever forming the complex product."""
    if mode == "real":
        return _qgt_product_re(jac, v)
    if mode == "imag":
        return 1j * _qgt_product_im(jac, v)

    raise ValueError(f"The transformation mode has to be one of {TRANSFORM_MODES}, got {mode!r}.")


def transformed_product(jac, v, mode: TransformMode):
    """
    ``q[S_0] @ v`` for the matrix :math:`q[S_0]`, ``v`` of shape ``(num_params,)`` or
    ``(num_params, r)``. The transformations :math:`\\text{Re}` and :math:`i\\,\\text{Im}`
    are only real-linear, so ``q[S_0 v]`` equals ``q[S_0] v`` for a real ``v`` only; a
    complex ``v`` is split into real and imaginary part, which doubles the width of
    the product. For these two transformations the product is done in real
    arithmetic (:func:`_qgt_product_re`, :func:`_qgt_product_im`).
    """
    if mode == "identity":
        return _qgt_product(jac, v)
    if not jnp.iscomplexobj(v):
        return _transformed_product_real_input(jac, v, mode)

    v2d = v.reshape(v.shape[0], -1)
    r = v2d.shape[1]
    parts = jnp.concatenate([jnp.real(v2d), jnp.imag(v2d)], axis=1)
    Sp = _transformed_product_real_input(jac, parts, mode)

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
