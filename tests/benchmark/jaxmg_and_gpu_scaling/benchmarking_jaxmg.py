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
parser.add_argument("-batch_size", default=500, type=int)
parser.add_argument("-mc_chains", default=50, type=int)
parser.add_argument("-net", default="CpxCNN", type=str)
parser.add_argument("-nb_gpus", default=1, type=int)
parser.add_argument("-jaxmg", default=0, type=int)
parser.add_argument("-info", default="", type=str)
args = parser.parse_args()

L = 100
g = -0.7
num_samples = args.num_samples
batch_size = args.batch_size
mc_chains = args.mc_chains
net_name = args.net # "CpxCNN", "RWKV" or "RNN"

print(f"Running {net_name} benchmark with Ns={num_samples}, batch_size={batch_size}, mc_chains={mc_chains}, jaxmg={args.jaxmg}")


# Initialize net
if args.net == "CpxCNN":
    net = nets.CpxCNN(F=[20,], channels=[10, 8, 6, 4], bias=True, firstLayerBias=True) # same as in figure 3d of the old paper
elif net_name == "RWKV":
    net = nets.RWKVCPM(L=L, LHilDim=2, patch_size=1, embedding_size=16, num_heads=3, num_layers=3, hidden_size=8)
elif net_name == "RNN":
    net = nets.RNN1DGeneral(L=L, hiddenSize=100, depth=5, cell="RNN")
else:
    raise ValueError(f"Unknown net {args.net}. Please define it")

psi = jVMC_exp.vqs.NQS(net, L, batchSize_per_device=batch_size, seed=1234)
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
output_manager = OutputManager(path=Path(f"./data/net={net_name}_bs={batch_size}_chains={mc_chains}_Ns={num_samples}_gpus={args.nb_gpus}{args.info}.hdf5"))

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

# Set up solver
if args.jaxmg:
    solver = jVMC_exp.solver.PinvSNR(diagonalization_mode="distributed")
else:
    solver = jVMC_exp.solver.PinvSNR(diagonalization_mode="device")

# Set up TDVP
tdvp = jVMC_exp.optimizer.TDVP(
    sampler,
    psi,
    make_real=True,
    diagonalShift=10,
    output_manager=output_manager,
    solver=solver
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