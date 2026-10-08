import sys
sys.path.append(sys.path[0] + "/../../..")
import argparse
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd

import jVMC_exp
import jVMC_exp.nets as nets
import jVMC_exp.operator.discrete as op
import jVMC_exp.sampler as sampler
from jVMC_exp.vqs import NQS
from jVMC_exp.stats import SampledObs, LazySampledObs
from jVMC_exp.optimizer.tdvp import TDVP

parser = argparse.ArgumentParser()
parser.add_argument("jacobian", choices=["dense", "batched"])
parser.add_argument("--n_batches", type=int, default=1)
parser.add_argument("--quantity", choices=["force", "qgt"], default="force")
parser.add_argument("--out_dir", default=".")
parser.add_argument("--n_reps", type=int, default=5, help="Timed repetitions after the compiling call")
parser.add_argument("--num_hidden", type=int, default=512)
parser.add_argument("--log2_samples", type=int, default=16)
parser.add_argument("--sample_batch", type=int, default=4096,
                    help="Batch size for sampling and E_loc, kept fixed so they don't set the memory peak")
args = parser.parse_args()

def build_hamiltonian(L, J=-1.0, hx=-0.7):
    H = 0
    for i in range(L):
        H += J * op.SigmaZ(i) * op.SigmaZ((i + 1) % L) + hx * op.SigmaX(i)
    return H

def device_memory():
    """
    Largest bytes_in_use and peak_bytes_in_use over the local devices (None on CPU).
    """
    stats = [d.memory_stats() for d in jax.local_devices()]
    if any(s is None for s in stats):
        return None, None
    return max(s["bytes_in_use"] for s in stats), max(s["peak_bytes_in_use"] for s in stats)

L = 10
n_samples = 2**args.log2_samples
n_chains = n_samples // 16
grad_batch_size = n_samples // args.n_batches
sample_batch_size = min(args.sample_batch, grad_batch_size)
batched = args.jacobian == "batched"

rbm = nets.CpxRBM(numHidden=args.num_hidden, bias=False)
psi = NQS(rbm, L, sample_batch_size, seed=123)
mc_sampler = sampler.MCSampler(
    psi, jVMC_exp.propose.SpinFlip(),
    key=123, numChains=n_chains, numSamples=n_samples
)
hamiltonian = build_hamiltonian(L)
loss_function = jVMC_exp.objective_function.Observable(hamiltonian, batched_jacobian=batched)
# Only used for the way it builds S, its solver is never called
optimizer = TDVP(mc_sampler, psi, make_real=False)

mc_sampler.sample()
o_loc = loss_function(mc_sampler)

# Only the Jacobian is batched with n_samples // n_batches.
# NQS has no setter for the batch size, so it is changed here directly.
psi._batchSize = grad_batch_size

def compute():
    if batched:
        grad_log_psi = LazySampledObs(psi.lazy_gradients(mc_sampler.samples), mc_sampler.weights)
    else:
        grad_log_psi = SampledObs(psi.gradients(mc_sampler.samples), mc_sampler.weights)

    if args.quantity == "force":
        # As in Observable.value_and_grad
        return grad_log_psi.get_covar(o_loc)

    # As in Evolution.get_update: the doubled holomorphic gradient is removed,
    # then S is built sharded across devices by _get_lhs_dense
    if psi.holomorphic:
        grad_log_psi.transform(optimizer._remove_double_trans)
    S = optimizer._get_lhs_dense(grad_log_psi)
    # _get_lhs_dense keeps a reference to S, which would stay alive during the next repetition
    optimizer._S0 = None

    return S

in_use_before, peak_before = device_memory()

# The first call compiles, the following ones are timed
times = []
for rep in range(args.n_reps + 1):
    t0 = time.perf_counter()
    result = compute()
    jax.block_until_ready(result)
    times.append(time.perf_counter() - t0)
    if rep < args.n_reps:
        del result

_, peak_after = device_memory()

# peak_bytes_in_use never resets, so the phase is only measured if it set a new peak
phase_peak_bytes = None
if peak_after is not None:
    if peak_after > peak_before:
        phase_peak_bytes = peak_after - in_use_before
    else:
        print(
            f"WARNING: the peak ({peak_before / 1e9:.2f} GB) was set before computing the "
            f"{args.quantity}; lower --sample_batch to measure it."
        )

results = dict(
    jacobian=args.jacobian,
    quantity=args.quantity,
    n_batches=args.n_batches,
    n_devices=jax.device_count(),
    n_samples=n_samples,
    n_params=psi.numParameters,
    holomorphic=bool(psi.holomorphic),
    grad_batch_size=grad_batch_size,
    out_shape=str(result.shape),
    out_sharding=str(getattr(result.sharding, "spec", result.sharding)),
    out_bytes_per_device=result.addressable_shards[0].data.nbytes,
    result_norm=float(jnp.linalg.norm(result)),
    time=float(np.median(times[1:])),
    time_first_call=times[0],
    in_use_before=in_use_before,
    peak_before=peak_before,
    phase_peak_bytes=phase_peak_bytes,
)
print(results)

os.makedirs(args.out_dir, exist_ok=True)
csv_name = os.path.join(args.out_dir, f"jacobian_{args.quantity}.csv")
df = pd.DataFrame([results])
df_old = pd.read_csv(csv_name) if os.path.exists(csv_name) else pd.DataFrame()
df = pd.concat([df_old, df], ignore_index=True)
df.to_csv(csv_name, index=False)