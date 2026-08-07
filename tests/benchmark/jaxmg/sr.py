import jVMC_exp
import jax

L = 19

n_samples = 2**12
n_chains = n_samples // 4
batch_size = n_samples

net = jVMC_exp.nets.RBM(L)
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
solver = jVMC_exp.solver.PinvSNR(diagonalization_mode="distributed")
stepper = jVMC_exp.stepper.Euler(1e-2)
opt = jVMC_exp.optimizer.SR(sampler, psi, solver=solver)

psi.parameters = opt.step(0, stepper, loss_function)[0]

if jax.process_index() == 0:
    print("Done")