import os
from pathlib import Path
import argparse

os.environ["JVMC_USE_DISTRIBUTED"] = "true"

import jax

import jVMC_exp
import jVMC_exp.nets as nets
import jVMC_exp.operator.discrete as op
from jVMC_exp.objective_function import Observable
from jVMC_exp.util.output_manager import OutputManager


parser = argparse.ArgumentParser()
parser.add_argument("-num_samples", default=10000, type=int)
parser.add_argument("-batch_size", default=2048, type=int)
parser.add_argument("-mc_chains", default=100, type=int)
parser.add_argument("-net", default="CpxCNN", type=str)
parser.add_argument("-mode", default="batch_size", type=str)
parser.add_argument("-info", default="", type=str)
args = parser.parse_args()


# parameters
L = 50
g = -0.7
num_samples = args.num_samples
mode = args.mode # "batch_size" or "mc_chains"
net_name = args.net # "CpxCNN", "RWKV" or "RNN"
batch_size = args.batch_size
mc_chains = args.mc_chains # has no effect for autoregressive nets

if mode not in ("batch_size", "mc_chains"):
    raise ValueError(f"Unknown mode {mode}.")
if net_name in ("RWKV", "RNN") and mode == "mc_chains":
    raise ValueError(f"{net_name} is sampled autoregressively, the mc_chains scan is useless.")


print(f"Running benchmark with batch_size={batch_size}, mc_chains={mc_chains}")

# Initialize net
if net_name == "CpxCNN":
    net = nets.CpxCNN(F=[20,], channels=[10, 8, 6, 4], bias=True, firstLayerBias=True) # same as in figure 3d of the old paper
elif net_name == "RWKV":
    net = nets.RWKVCPM(L=L, LHilDim=2, patch_size=1, embedding_size=16, num_heads=3, num_layers=3, hidden_size=8)
elif net_name == "RNN":
    net = nets.RNN1DGeneral(L=L, hiddenSize=40, depth=2, cell="RNN")
else:
    raise ValueError(f"Unknown net {net_name}. Please define it")

psi = jVMC_exp.vqs.NQS(net, L, batchSize=batch_size, seed=1234)
print(f"The variational ansatz has {psi.numParameters} parameters.")

# Set up TFI hamiltonian
hamiltonian = 0
for site in range(L):
    hamiltonian += -1.0 * op.SigmaZ(site) * op.SigmaZ((site + 1) % L)
    hamiltonian += g * op.SigmaX(site)
objective_function = Observable(hamiltonian)

# Set up sampler
if psi.is_generator: # Autoregressive nets
    sampler = jVMC_exp.sampler.MCSampler(
        psi,
        updateProposer=None,
        key=jax.random.PRNGKey(4321),
        numSamples=num_samples,
    )
else:
    sampler = jVMC_exp.sampler.MCSampler(
        psi,
        updateProposer=jVMC_exp.propose.SpinFlip(),
        key=jax.random.PRNGKey(4321),
        numChains=mc_chains,
        sweepSteps=L,
        numSamples=num_samples,
        thermalizationSweeps=20,
    )

# Set up OutputManager
if mode == "batch_size":
    output_manager = OutputManager(path=Path(f"./data/batch_size_scan_{net_name}_Ns={num_samples}_mc_chains={mc_chains}{args.info}.hdf5"), group=f"bc={batch_size}", append=True)
else:
    output_manager = OutputManager(path=Path(f"./data/mc_chain_scan_{net_name}_Ns={num_samples}_batch_size={batch_size}{args.info}.hdf5"), group=f"mc={mc_chains}", append=True)

def print_timings(outp: OutputManager):
    total = 0.0
    for key, value in outp.timings.items():
        latest = float(value.get("last_timing", 0.0))
        average = float(value["total"]) / max(int(value["count"]), 1)
        print("\taverage and latest timings of ", key)
        print("\t", average)
        print("\t", latest)
        total += latest
    print("\t=== Total: ", total)

# Set up TDVP
tdvp = jVMC_exp.optimizer.TDVP(
    sampler,
    psi,
    make_real=True,
    diagonalShift=10,
    output_manager=output_manager
)

stepper = jVMC_exp.stepper.Euler(timeStep=1e-2)  # ODE integrator

# Run a few steps of TDVP
for _ in range(3):
    # new_parameters, _ = stepper.step(0, tdvp, psi.parameters_flat, objective_function=objective_function)
    new_parameters, _ = tdvp.step(0, stepper, objective_function)
    psi.parameters = new_parameters

    print("Benchmarking data")
    print_timings(tdvp.output_manager)

tdvp.output_manager.flush_timings()
tdvp.output_manager.flush_memory_info()

# Print memory info
for i, device in enumerate(jax.devices()):
    if jax.process_index() == i:
        try:
            memory_info = device.memory_stats()
            print(f"Device {i}: {memory_info}")
        except:
            print(f"Device {i}: Memory stats not available")
