import sys
sys.path.append(sys.path[0] + "/../../..")
import argparse
import os
import jax

import jVMC_exp
import jVMC_exp.nets as nets
import jVMC_exp.operator.discrete as op
import jVMC_exp.sampler as sampler
from jVMC_exp.vqs import NQS

parser = argparse.ArgumentParser()
parser.add_argument("jacobian", choices=["dense", "batched"])
parser.add_argument("--n_batches", type=int, default=1)
parser.add_argument("--out_dir", default="mem_profiles")
args = parser.parse_args()

def build_hamiltonian(L, J=-1.0, hx=-0.7):
    H = 0
    for i in range(L):
        H += J * op.SigmaZ(i) * op.SigmaZ((i + 1) % L) + hx * op.SigmaX(i)
    return H

num_hidden = 512
L = 10
n_samples = 2**16
n_chains = n_samples // 16
batch_size = n_samples // args.n_batches
batched = args.jacobian == "batched"

rbm = nets.CpxRBM(numHidden=num_hidden, bias=False)
psi = NQS(rbm, L, batch_size, seed=123)
mc_sampler = sampler.MCSampler(
    psi, jVMC_exp.propose.SpinFlip(),
    key=123, numChains=n_chains, numSamples=n_samples
)
hamiltonian = build_hamiltonian(L)
loss_function = jVMC_exp.objective_function.Observable(hamiltonian, batched_jacobian=batched)

mc_sampler.sample()
o_loc = loss_function(mc_sampler)  # keeps sampler/psi state identical to the real benchmark, unused otherwise

os.makedirs(args.out_dir, exist_ok=True)
tag = f"{args.jacobian}_n{args.n_batches}"

# --- warmup pass: force compilation before profiling, so the profile reflects
#     steady-state execution rather than one-time compilation artifacts ---
if batched:
    grad_log_psi = jVMC_exp.stats.LazySampledObs(mc_sampler.psi.lazy_gradients(mc_sampler.samples), mc_sampler.weights)
else:
    grad_log_psi = jVMC_exp.stats.SampledObs(mc_sampler.psi.gradients(mc_sampler.samples), mc_sampler.weights)
jax.block_until_ready(grad_log_psi.get_covar())
del grad_log_psi

# --- profiled pass: get_covar() only, self-covariance (other=None) ---
if batched:
    grad_log_psi = jVMC_exp.stats.LazySampledObs(mc_sampler.psi.lazy_gradients(mc_sampler.samples), mc_sampler.weights)
else:
    grad_log_psi = jVMC_exp.stats.SampledObs(mc_sampler.psi.gradients(mc_sampler.samples), mc_sampler.weights)

jax.block_until_ready(0)  # make sure the device is quiescent before the "before" snapshot
before_path = os.path.join(args.out_dir, f"before_{tag}.prof")
jax.profiler.save_device_memory_profile(before_path)

result = grad_log_psi.get_covar()
jax.block_until_ready(result)

after_path = os.path.join(args.out_dir, f"after_{tag}.prof")
jax.profiler.save_device_memory_profile(after_path)

stats = jax.devices()[0].memory_stats()
print(f"[{tag}] bytes_in_use={stats.get('bytes_in_use')/1e9:.3f}GB peak_bytes_in_use={stats.get('peak_bytes_in_use')/1e9:.3f}GB")
print(f"[{tag}] wrote {before_path} and {after_path}")