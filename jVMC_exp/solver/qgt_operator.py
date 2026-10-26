"""
Matrix-free left hand side of the TDVP / SR equation.

The quantum geometric tensor :math:`S_0 = J^\\dagger J` is never formed: products
with blocks of vectors are computed from the (mean-subtracted, :math:`\\sqrt{w}`
weighted) Jacobian :math:`J` of shape ``(num_samples, num_params)``, which is sharded
across the devices along the sample axis. Each device contracts its own samples
and the partial sums are reduced with a ``psum``, so the only replicated arrays are
the ``(num_params, r)`` blocks that go in and out (:class:`QGTOperator`).

:class:`ShardedQGTOperator` splits the parameters as well: on a 2D mesh of
``P_s`` sample groups times ``P_p`` parameter groups it holds a copy of the Jacobian
sharded over both axes, and the blocks it is applied to are sharded over the
parameter rows, so that everything whose size grows with the number of parameters
is divided by ``P_p``.
"""
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax.custom_derivatives import linear_call
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

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


def _varying_zeros(shape, dtype, like, axes=("devices",)):
    """
    Zeros that are 'varying' over the mesh ``axes`` inside a ``shard_map``: the carry
    of a loop has to be, when a loop body makes it so (shard_map's rule for loop
    carries). ``jax.lax.pcast`` (``pvary`` before it) does exactly that but exists
    only in some JAX versions; otherwise the zeros are derived from ``like``, which
    is varying over ``axes``.
    """
    zeros = jnp.zeros(shape, dtype)
    if _USE_VARYING_API:
        if hasattr(jax.lax, "pcast"):
            return jax.lax.pcast(zeros, axes, to="varying")
        if hasattr(jax.lax, "pvary"):
            return jax.lax.pvary(zeros, axes)

    return zeros + jnp.zeros((), dtype) * jnp.real(like.reshape(-1)[0]).astype(dtype)


def _add_in_row_blocks(acc, contribution, block):
    """
    ``acc + contribution`` written in blocks of ``block`` rows, ``contribution(j, size)``
    being rows ``j * block`` to ``j * block + size`` of the summand: no single product
    has an output larger than one block.
    """
    n_rows = acc.shape[0]

    def add_block(j, acc, bsize=block):
        old = jax.lax.dynamic_slice_in_dim(acc, j * block, bsize, axis=0)
        return jax.lax.dynamic_update_slice_in_dim(
            acc, old + contribution(j, bsize), j * block, axis=0
        )

    n_full = n_rows // block
    acc = jax.lax.fori_loop(0, n_full, add_block, acc)
    if n_rows % block:
        acc = add_block(n_full, acc, n_rows % block)

    return acc


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

        return _add_in_row_blocks(acc, contribution, block)

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


def _transformed_product_real_input(jac, v, mode: TransformMode, products):
    """``q[S_0] v`` for a real ``v`` without ever forming the complex product."""
    _, product_re, product_im = products
    if mode == "real":
        return product_re(jac, v)
    if mode == "imag":
        return 1j * product_im(jac, v)

    raise ValueError(f"The transformation mode has to be one of {TRANSFORM_MODES}, got {mode!r}.")


def transformed_product(jac, v, mode: TransformMode, products=None):
    """
    ``q[S_0] @ v`` for the matrix :math:`q[S_0]`, ``v`` of shape ``(num_params,)`` or
    ``(num_params, r)``. The transformations :math:`\\text{Re}` and :math:`i\\,\\text{Im}`
    are only real-linear, so ``q[S_0 v]`` equals ``q[S_0] v`` for a real ``v`` only; a
    complex ``v`` is split into real and imaginary part, which doubles the width of
    the product. For these two transformations the product is done in real
    arithmetic (:func:`_qgt_product_re`, :func:`_qgt_product_im`).

    ``products`` are the three products ``(S_0, Re S_0, Im S_0)`` as functions of
    ``(jac, v)``; by default those of the sample-sharded Jacobian.
    """
    products = products or (_qgt_product, _qgt_product_re, _qgt_product_im)
    if mode == "identity":
        return products[0](jac, v)
    if not jnp.iscomplexobj(v):
        return _transformed_product_real_input(jac, v, mode, products)

    v2d = v.reshape(v.shape[0], -1)
    r = v2d.shape[1]
    parts = jnp.concatenate([jnp.real(v2d), jnp.imag(v2d)], axis=1)
    Sp = _transformed_product_real_input(jac, parts, mode, products)

    return (Sp[:, :r] + 1j * Sp[:, r:]).reshape(v.shape)


@partial(jax.jit, static_argnames=("mode", "has_diag", "layout"))
def _apply(jac, v, diag, diag_shift, *, mode, has_diag, layout=None):
    """
    ``A v`` for a replicated ``v``, from the sample-sharded Jacobian (``layout=None``)
    or from the column-sharded one of a :class:`ShardedQGTOperator` with that layout.
    """
    products = None if layout is None else _column_products(layout)
    Sv = transformed_product(jac, v, mode, products)
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


# ----------------------------------------------------------------------------------
# Parameter-sharded layout: the Jacobian sharded over samples AND parameters
# ----------------------------------------------------------------------------------

SAMPLES_AXIS = "samples"
PARAMS_AXIS = "params"
_MESH_AXES = (SAMPLES_AXIS, PARAMS_AXIS)
JACOBIAN_SPEC = P(SAMPLES_AXIS, PARAMS_AXIS)
ROWS_SPEC = P(PARAMS_AXIS)


@lru_cache(maxsize=None)
def _parameter_mesh(shape):
    return Mesh(MESH.devices.reshape(shape), _MESH_AXES)


def parameter_mesh(shape=None) -> Mesh:
    """
    The 2D mesh of ``P_s`` sample groups times ``P_p`` parameter groups, ``shape =
    (P_s, P_p)``, over the devices of ``MESH`` in the same order: device ``i`` of
    ``MESH`` sits at ``(i // P_p, i % P_p)``, so that the sample-sharded Jacobian is
    sharded over both axes of this mesh (samples major) without moving any data.
    ``None`` is ``(1, number of devices)``: pure parameter sharding.
    """
    n_devices = MESH.size
    shape = (1, n_devices) if shape is None else tuple(int(p) for p in shape)
    if len(shape) != 2 or min(shape) < 1 or shape[0] * shape[1] != n_devices:
        raise ValueError(
            "The mesh shape (P_s, P_p) has to be two positive integers whose product is "
            f"the number of devices ({n_devices}), got {shape}."
        )

    return _parameter_mesh(shape)


@dataclass(frozen=True)
class ParameterLayout:
    """
    ``num_params`` parameter rows split over the ``P_p`` parameter groups of ``mesh``:
    zero padded to ``padded``, a multiple of ``P_p``, ``local`` rows per group. Row
    ``i`` of a row-sharded array lives in group ``i // local``. Hashable, so that it
    is a static argument of the jitted kernels.
    """
    mesh: Mesh
    num_params: int

    @property
    def groups(self) -> int:
        return self.mesh.shape[PARAMS_AXIS]

    @property
    def sample_groups(self) -> int:
        return self.mesh.shape[SAMPLES_AXIS]

    @property
    def padded(self) -> int:
        return -(-self.num_params // self.groups) * self.groups

    @property
    def local(self) -> int:
        return self.padded // self.groups

    @property
    def pad(self) -> int:
        return self.padded - self.num_params

    def sharding(self, spec=ROWS_SPEC) -> NamedSharding:
        return NamedSharding(self.mesh, spec)


def _redistribution_chunk(jac, layout: ParameterLayout):
    """
    Rows of the local Jacobian shard per ``all_to_all`` of :func:`_column_sharded_jacobian`,
    at most ``global_defs.MATMUL_CHUNK_BYTES`` of (padded) rows (0: the whole shard at once).
    """
    n_local_rows = jac.shape[0] // layout.mesh.size
    row_bytes = layout.padded * jac.dtype.itemsize
    budget = global_defs.MATMUL_CHUNK_BYTES or n_local_rows * row_bytes

    return max(1, min(n_local_rows, budget // row_bytes))


def _column_sharded_jacobian(jac, *, layout: ParameterLayout):
    """
    The sample-sharded Jacobian redistributed to the 2D mesh: device ``(s, p)`` gets
    the rows of sample group ``s`` and the columns of parameter group ``p``, the
    columns zero padded to ``layout.padded``. An ``all_to_all`` within every sample
    group: each device splits its rows into the column blocks of the group and
    concatenates the row blocks it receives.

    The exchange is done for chunks of the local rows of at most
    ``global_defs.MATMUL_CHUNK_BYTES``, each written into the preallocated output. A
    single ``all_to_all`` of the whole shard needs a rearranged copy of its input next
    to the input and the output (three Jacobians per device, the out-of-memory error of
    the limits scans at 31 GB per Jacobian); in chunks the redistribution needs the
    input and the output plus a few chunks.
    """
    return _column_sharded_jacobian_impl(jac, layout=layout, chunk=_redistribution_chunk(jac, layout))


@partial(jax.jit, static_argnames=("layout", "chunk"))
def _column_sharded_jacobian_impl(jac, *, layout: ParameterLayout, chunk: int):
    groups = layout.groups

    def local(j):
        n_rows = j.shape[0]
        # Rows received from group member q go to rows q * n_rows + (rows of the chunk)
        out = _varying_zeros((groups, n_rows, layout.local), j.dtype, like=j, axes=_MESH_AXES)

        def exchange(k, out):
            # The last chunk is shifted back to end at the last row (and repeats a few
            # rows): a remainder chunk outside the loop would copy the output
            start = jnp.minimum(k * chunk, n_rows - chunk)
            rows = jax.lax.dynamic_slice_in_dim(j, start, chunk)
            if layout.pad:
                rows = jnp.pad(rows, ((0, 0), (0, layout.pad)))
            received = jax.lax.all_to_all(rows, PARAMS_AXIS, 1, 0, tiled=True)
            return jax.lax.dynamic_update_slice_in_dim(
                out, received.reshape(groups, chunk, layout.local), start, axis=1
            )

        out = jax.lax.fori_loop(0, -(-n_rows // chunk), exchange, out)

        return out.reshape(groups * n_rows, layout.local)

    return jax.shard_map(
        local, mesh=layout.mesh, in_specs=(P(_MESH_AXES),), out_specs=JACOBIAN_SPEC
    )(jac)


def _column_block_product(jac, q, part: Part):
    """
    On one device ``(s, p)`` of the 2D mesh, with ``jac`` its block ``J[s, p]`` and
    ``q`` the rows ``p`` of a block: rows ``p`` of ``part`` of ``J^\\dagger (J q)``,
    summed over all samples, as in :func:`_local_product`:

    * ``Z = J[s, :] q`` is the local product ``J[s, p] q[p]`` summed over the
      parameter groups (``psum``); it is formed for chunks of the samples of at most
      ``global_defs.MATMUL_CHUNK_BYTES`` (also as a ``Z`` chunk), so that it is
      never materialized for all samples;
    * the chunk's ``J[s, p]^\\dagger Z`` is accumulated locally, in blocks of rows of
      at most the same size, and summed over the sample groups at the end.

    For ``"re"`` / ``"im"`` (real ``q``) both products are real: ``Z`` is the pair
    ``(A q, B q)`` with ``J = A + iB``, reduced in one ``psum``.
    """
    n_samples, n_local = jac.shape
    width = q.shape[1]
    if part == "cpx":
        out_dtype = jnp.result_type(jac.dtype, q.dtype)
        n_z = 1
    else:
        out_dtype = jnp.result_type(jnp.real(jac).dtype, q.dtype)
        n_z = 2
    itemsize = jnp.dtype(out_dtype).itemsize
    jac_row = n_local * jac.dtype.itemsize
    z_row = n_z * width * itemsize
    budget = global_defs.MATMUL_CHUNK_BYTES or n_samples * max(jac_row, z_row)
    chunk = max(1, min(n_samples, budget // jac_row, budget // z_row))
    block = max(1, min(n_local, budget // max(1, width * itemsize)))

    acc = _varying_zeros((n_local, width), out_dtype, like=jac, axes=_MESH_AXES)

    def add_chunk(k, acc, size=chunk):
        rows = jax.lax.dynamic_slice_in_dim(jac, k * chunk, size)
        # Materialized, see _local_product
        rows = jax.lax.optimization_barrier(rows)

        if part == "cpx":
            z = jax.lax.psum(rows @ q, PARAMS_AXIS)

            def contribution(j, bsize):
                cols = jax.lax.dynamic_slice_in_dim(rows, j * block, bsize, axis=1)
                return jnp.conj(cols).T @ z
        else:
            ar, ai = jnp.real(rows), jnp.imag(rows)
            za, zb = jax.lax.psum((ar @ q, ai @ q), PARAMS_AXIS)

            def contribution(j, bsize):
                cr = jax.lax.dynamic_slice_in_dim(ar, j * block, bsize, axis=1)
                ci = jax.lax.dynamic_slice_in_dim(ai, j * block, bsize, axis=1)
                if part == "re":
                    return cr.T @ za + ci.T @ zb
                return cr.T @ zb - ci.T @ za

        return _add_in_row_blocks(acc, contribution, block)

    n_full = n_samples // chunk
    acc = jax.lax.fori_loop(0, n_full, add_chunk, acc)
    if n_samples % chunk:
        acc = add_chunk(n_full, acc, n_samples % chunk)

    return jax.lax.psum(acc, SAMPLES_AXIS)


@partial(jax.jit, static_argnames=("layout", "part"))
def _gathered_product_impl(jac, v, *, layout: ParameterLayout, part: Part):
    """
    ``part`` of ``S_0 v`` for the column-sharded ``jac`` and a replicated ``v`` of shape
    ``(num_params,)`` or ``(num_params, r)``: ``v`` is padded and split over the
    parameter groups (no communication), the row-sharded result all-gathered (a
    sharding constraint, the version-independent way out of a ``shard_map``) and
    stripped of the padding, so that the result is replicated.
    """
    n = layout.num_params
    v2d = jnp.pad(v.reshape(n, -1), ((0, layout.pad), (0, 0)))
    Sv = jax.shard_map(
        partial(_column_block_product, part=part),
        mesh=layout.mesh,
        in_specs=(JACOBIAN_SPEC, ROWS_SPEC),
        out_specs=ROWS_SPEC,
    )(jac, v2d)
    Sv = jax.lax.with_sharding_constraint(Sv, layout.sharding(REPLICATED_SPEC))

    return Sv[:n].reshape(v.shape)


@lru_cache(maxsize=None)
def _column_products(layout: ParameterLayout):
    """
    The products ``(S_0, Re S_0, Im S_0)`` of the column-sharded Jacobian with
    replicated vectors, as functions of ``(jac, v)`` for :func:`transformed_product`,
    with the same explicit transposes as :func:`_qgt_product`, :func:`_qgt_product_re`
    and :func:`_qgt_product_im`.
    """
    impl_cpx, impl_re, impl_im = (
        partial(_gathered_product_impl, layout=layout, part=part) for part in ("cpx", "re", "im")
    )

    def transpose(jac, ct):
        return jnp.conj(impl_cpx(jac, jnp.conj(ct)))

    def transpose_real(jac, ct):
        return jnp.real(transpose(jac, ct))

    def product(jac, v):
        return linear_call(impl_cpx, transpose if jnp.iscomplexobj(v) else transpose_real, jac, v)

    def product_re(jac, v):
        return linear_call(impl_re, impl_re, jac, v)

    def product_im(jac, v):
        return linear_call(impl_im, lambda jac, ct: -impl_im(jac, ct), jac, v)

    return product, product_re, product_im


@partial(jax.jit, static_argnames=("layout",))
def _column_qgt_product(jac, v, *, layout: ParameterLayout):
    return _column_products(layout)[0](jac, v)


class ShardedQGTOperator:
    """
    :class:`QGTOperator` with the parameters sharded as well as the samples.

    On the 2D mesh of ``P_s`` sample groups times ``P_p`` parameter groups
    (:func:`parameter_mesh`, ``mesh_shape = (P_s, P_p)``, default ``(1, number of
    devices)``) the operator holds its own copy of the Jacobian, ``J[s, p]`` of shape
    ``(num_samples / P_s, padded_num_params / P_p)`` on device ``(s, p)``, made from
    the sample-sharded Jacobian by one ``all_to_all`` within every sample group; the
    sample-sharded Jacobian itself is left untouched. ``num_params`` is padded with
    zero columns to a multiple of ``P_p``.

    The operator is applied to blocks in two ways:

    * :meth:`__call__` and :meth:`raw` take and return REPLICATED vectors or blocks of
      shape ``(num_params,)`` / ``(num_params, r)``, exactly like
      :class:`QGTOperator` (same transposes, so also as ``A`` of
      ``jax.scipy.sparse.linalg.cg`` or in an L-BFGS loss). The row-sharded result is
      all-gathered and the padding stripped.
    * The randomized eigensolver (:mod:`jVMC_exp.solver.rsvd`) applies it to
      ROW-SHARDED, padded blocks of shape ``(padded_num_params, r)``, i.e.
      ``padded_num_params / P_p`` rows per device, and keeps the result row-sharded.

    Per application to a block ``Q``: ``Z = J Q`` as a local product plus a ``psum``
    over the parameter groups, in sample chunks (``Z`` is never formed for all
    samples), then ``Y[p] = q[J[:, p]^\\dagger Z]`` locally plus a ``psum`` over the
    sample groups when ``P_s > 1``; see :func:`_column_block_product`.
    """
    def __init__(
            self, jacobian, mode: TransformMode, diag_scale: float = 0., diag_shift: float = 0.,
            mesh_shape=None
        ):
        self._init_from(QGTOperator(jacobian, mode, diag_scale, diag_shift), mesh_shape)

    @classmethod
    def from_operator(cls, operator: QGTOperator, mesh_shape=None) -> "ShardedQGTOperator":
        """The same operator as ``operator``, with its Jacobian redistributed to the 2D mesh."""
        self = cls.__new__(cls)
        self._init_from(operator, mesh_shape)

        return self

    def _init_from(self, operator: QGTOperator, mesh_shape):
        self._layout = ParameterLayout(parameter_mesh(mesh_shape), operator.shape[0])
        self._jacobian = _column_sharded_jacobian(operator.jacobian, layout=self._layout)
        self._mode = operator.mode
        self._diag_scale = operator.diag_scale
        self._diag_shift = operator.diag_shift
        self._has_diag = operator.has_diagonal_terms
        self._dtype = operator.dtype
        self._diag = operator._diag
        self._shift = operator._shift
        self._diag_rows = jax.device_put(
            jnp.pad(self._diag, (0, self._layout.pad)), self._layout.sharding(ROWS_SPEC)
        )

    @property
    def jacobian(self):
        """The column-sharded copy of the Jacobian, shape ``(num_samples, padded_num_params)``."""
        return self._jacobian

    @property
    def layout(self) -> ParameterLayout:
        return self._layout

    @property
    def mesh(self) -> Mesh:
        return self._layout.mesh

    @property
    def num_params(self) -> int:
        return self._layout.num_params

    @property
    def padded_num_params(self) -> int:
        return self._layout.padded

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
        n = self._layout.num_params
        return (n, n)

    @property
    def dtype(self):
        """dtype of ``A v`` for a real ``v``."""
        return self._dtype

    def raw(self, v):
        """The bare quantum geometric tensor applied to the replicated ``v``: :math:`S_0 v`."""
        return _column_qgt_product(self._jacobian, v, layout=self._layout)

    def __call__(self, v):
        return _apply(
            self._jacobian, v, self._diag, self._shift,
            mode=self._mode, has_diag=self._has_diag, layout=self._layout
        )
