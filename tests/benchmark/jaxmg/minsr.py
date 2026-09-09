import jVMC_exp
import jax

L = 19

n_samples = 2**12
n_chains = n_samples // 4
batch_size = n_samples

net = jVMC_exp.nets.CpxRBM(256)
psi = jVMC_exp.vqs.NQS(net, L, batch_size, seed=123)
sampler = jVMC_exp.sampler.MCSampler(
    psi, 
    jVMC_exp.propose.SpinFlip(),
    123,
    n_chains, 
    n_samples
)

if jax.process_index() == 0:
    print("# parameters:", psi.numParameters)
    print("Holo:", psi.holomorphic)
    print("Real params:", psi.realParams)

J = -1
h = 0.5
H = 0
for i in range(L):
    H += J * jVMC_exp.operator.discrete.SigmaZ(i) * jVMC_exp.operator.discrete.SigmaZ((i + 1) % L)
    H += h * jVMC_exp.operator.discrete.SigmaX(i)

loss_function = jVMC_exp.objective_function.Observable(H)
stepper = jVMC_exp.stepper.Euler(1e-2)
solver = jVMC_exp.solver.Pinv(diagonalization_mode="distributed")
opt = jVMC_exp.optimizer.MinSR(sampler, psi, solver=solver)


psi.parameters = opt.step(0, stepper, loss_function)[0]

if jax.process_index() == 0:
    print("Done")