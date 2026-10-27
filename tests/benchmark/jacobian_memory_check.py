"""
Compile-only memory check of the Jacobian-sized kernels of a TDVP step with the
randomized solver (RandomizedPinvSNR, layout="parameters"), on the GPUs of ONE node.

Nothing of Jacobian size is allocated: every kernel is lowered and compiled for
abstract inputs (jax.ShapeDtypeStruct) and XLA's memory analysis of the executable is
printed, per device:

* the redistribution of the Jacobian to the 2D mesh as it is now (output preallocated
  and donated) and as it was in 5019475 (loop buffer of its own; on GPU three
  Jacobians, the out-of-memory errors of limits_scan_16gpu_ritz_shard);
* the force (_get_weighted_covar), the in-place normalization;
* with --rank: one operator application on an (n, r) block and the r x r Cholesky
  factorization of CholeskyQR.

"peak" is arguments + outputs - aliased + temporaries of one executable, the memory it
needs while it runs (buffers alive outside it, e.g. the sample-sharded Jacobian during
the redistribution, are among its arguments here).

Usage, on a node with 4 GPUs (JVMC_USE_DISTRIBUTED unset; one process sees all GPUs):

    python tests/benchmark/jacobian_memory_check.py --channels 112 128 160 192 --rank 16000
    python tests/benchmark/jacobian_memory_check.py --params 492800 --devices 16   # 16-GPU shard sizes

``--devices P`` compiles for the Jacobian shard sizes of P GPUs on the GPUs at hand: the
number of samples is scaled by (GPUs here) / P, so that every Jacobian shard (sample- and
column-sharded) has the size it has on P GPUs. The (n, r) blocks are sharded over the
GPUs here, (P / GPUs here) times larger than on P GPUs: with --devices the operator
application row is an upper bound.
``--no-autotune`` switches off the GEMM autotuner (it allocates operand-sized scratch
while compiling).
"""
import argparse
import os
import sys

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
group = parser.add_mutually_exclusive_group()
group.add_argument("--channels", type=int, nargs="+", default=[112, 128, 160, 192],
                   help="ResNet channel counts of the scan (n = 30 nc^2 + 10 nc)")
group.add_argument("--params", type=int, nargs="+", help="parameter counts instead of --channels")
parser.add_argument("--num-samples", type=int, default=40000)
parser.add_argument("--devices", type=int, default=None, help="shard sizes of this many GPUs (default: the GPUs here)")
parser.add_argument("--rank", type=int, default=None, help="also the (n, r) operator application and the r x r Cholesky")
parser.add_argument("--chunk-bytes", type=int, default=None, help="MATMUL_CHUNK_BYTES (default: jVMC_exp's, 1 GiB)")
parser.add_argument("--no-autotune", action="store_true", help="XLA_FLAGS=--xla_gpu_autotune_level=0")
args = parser.parse_args()

if args.no_autotune:
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0").strip()
os.environ.pop("JVMC_USE_DISTRIBUTED", None)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import jVMC_exp
from jVMC_exp import global_defs, stats
from jVMC_exp.sharding_config import DEVICE_SHARDING, MESH, REPLICATED_SPEC
from jVMC_exp.solver import qgt_operator as qo
from jVMC_exp.solver import rsvd

GIB = 2**30


@partial(jax.jit, static_argnames=("layout", "chunk"))
def _redistribution_5019475(jac, *, layout, chunk):
    """The redistribution of commit 5019475: the loop starts from zeros of its own and is reshaped."""
    groups = layout.groups

    def local(j):
        n_rows = j.shape[0]
        out = qo._varying_zeros((groups, n_rows, layout.local), j.dtype, like=j, axes=qo._MESH_AXES)

        def exchange(k, out):
            start = jnp.minimum(k * chunk, n_rows - chunk)
            rows = jax.lax.dynamic_slice_in_dim(j, start, chunk)
            if layout.pad:
                rows = jnp.pad(rows, ((0, 0), (0, layout.pad)))
            received = jax.lax.all_to_all(rows, qo.PARAMS_AXIS, 1, 0, tiled=True)
            return jax.lax.dynamic_update_slice_in_dim(out, received.reshape(groups, chunk, layout.local), start, axis=1)

        out = jax.lax.fori_loop(0, -(-n_rows // chunk), exchange, out)
        return out.reshape(groups * n_rows, layout.local)

    return jax.shard_map(local, mesh=layout.mesh, in_specs=(P(qo._MESH_AXES),), out_specs=qo.JACOBIAN_SPEC)(jac)


def memory(fn, *fn_args, **fn_kwargs):
    m = fn.lower(*fn_args, **fn_kwargs).compile().memory_analysis()
    peak = m.argument_size_in_bytes + m.output_size_in_bytes - m.alias_size_in_bytes + m.temp_size_in_bytes
    return dict(arg=m.argument_size_in_bytes, out=m.output_size_in_bytes, alias=m.alias_size_in_bytes,
                temp=m.temp_size_in_bytes, peak=peak)


def row(name, m, shard):
    gib = lambda b: f"{b / GIB:8.2f}"
    print(f"  {name:<34}{gib(m['arg'])}{gib(m['out'])}{gib(m['alias'])}{gib(m['temp'])}{gib(m['peak'])}"
          f"{m['peak'] / shard:9.2f}")


def main():
    if args.chunk_bytes is not None:
        global_defs.update(matmul_chunk_bytes=args.chunk_bytes)
    here = MESH.size
    target = args.devices or here
    # Global shapes scaled so that every local shape is that of a run on `target` GPUs
    n_samples = args.num_samples * here // target
    n_samples -= n_samples % here
    params = args.params or [30 * nc**2 + 10 * nc for nc in args.channels]
    dtype = jnp.complex128

    dev = jax.devices()[0]
    stats_ = dev.memory_stats() or {}
    print(f"jax {jax.__version__}, jVMC_exp at {os.path.dirname(jVMC_exp.__file__)}")
    print(f"{here} x {dev.device_kind}; usable per device (bytes_limit): "
          f"{stats_.get('bytes_limit', float('nan')) / GIB:.1f} GiB "
          f"(XLA_PYTHON_CLIENT_MEM_FRACTION={os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', 'unset: 0.75')})")
    print(f"shard sizes of {target} GPUs: {args.num_samples} samples -> {n_samples} rows here, "
          f"MATMUL_CHUNK_BYTES = {global_defs.MATMUL_CHUNK_BYTES / GIB:.2f} GiB, complex128 Jacobian")
    print(f"  {'executable (GiB per device)':<34}{'args':>8}{'out':>8}{'alias':>8}{'temp':>8}{'peak':>8}{'peak/J':>9}")

    layout_mesh = qo.parameter_mesh((1, here))
    for n in params:
        layout = qo.ParameterLayout(layout_mesh, n)
        jac = jax.ShapeDtypeStruct((n_samples, n), dtype, sharding=DEVICE_SHARDING)
        shard = n_samples * n * 16 / here
        print(f"n = {n} (Jacobian shard {shard / GIB:.2f} GiB, {n_samples // here} x {n})")

        chunk = qo._redistribution_chunk(jac, layout)
        out = jax.ShapeDtypeStruct((n_samples, layout.padded), dtype, sharding=layout.sharding(qo.JACOBIAN_SPEC))
        row("redistribution (donated output)", memory(qo._column_sharded_jacobian_impl, jac, out,
                                                       layout=layout, chunk=chunk), shard)
        row("redistribution as in 5019475", memory(_redistribution_5019475, jac, layout=layout, chunk=chunk), shard)

        weights = jax.ShapeDtypeStruct((n_samples,), jnp.float64, sharding=DEVICE_SHARDING)
        e_loc = jax.ShapeDtypeStruct((n_samples, 1), dtype, sharding=DEVICE_SHARDING)
        row_bytes = n * 16
        covar_chunk = max(1, min(n_samples // here, (global_defs.MATMUL_CHUNK_BYTES or n_samples * row_bytes) // row_bytes))
        row("force (_get_weighted_covar)", memory(stats._weighted_covar_chunked, jac, weights, e_loc,
                                                  chunk=covar_chunk), shard)
        mean = jax.ShapeDtypeStruct((n,), dtype, sharding=jax.sharding.NamedSharding(MESH, REPLICATED_SPEC))
        row("normalization (in place)", memory(stats._normalize_in_place, jac, weights, mean), shard)

        if args.rank:
            r = args.rank
            col_jac = out
            block = jax.ShapeDtypeStruct((layout.padded, r), jnp.float64, sharding=layout.sharding(qo.ROWS_SPEC))
            diag = jax.ShapeDtypeStruct((layout.padded,), jnp.float64, sharding=layout.sharding(qo.ROWS_SPEC))
            shift = jax.ShapeDtypeStruct((), jnp.float64)
            row(f"operator application (r = {r})", memory(rsvd._apply_rows, col_jac, block, diag, shift, layout=layout,
                                                          apply_mode="real", has_diag=True), shard)
            gram = jax.ShapeDtypeStruct((r, r), jnp.float64, sharding=jax.sharding.NamedSharding(MESH, REPLICATED_SPEC))
            row(f"CholeskyQR Cholesky (r = {r})", memory(rsvd._cholesky, gram, 1e-12), shard)


if __name__ == "__main__":
    main()
