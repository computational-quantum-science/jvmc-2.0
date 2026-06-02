import unittest
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import interp1d

import jVMC_exp
import jVMC_exp.nets as nets
from jVMC_exp.vqs import NQS
import jVMC_exp.operator.discrete as op
import jVMC_exp.sampler as sampler

class TestTimeEvolution(unittest.TestCase):
    def test_time_evolution(self):
        L = 4
        J = -1.0
        hx = -0.3

        weights = jnp.array(
            [0.23898957, 0.12614753, 0.19479055, 0.17325271, 0.14619853, 0.21392751,
             0.19648707, 0.17103704, -0.15457255, 0.10954413, 0.13228065, -0.14935214,
             -0.09963073, 0.17610707, 0.13386381, -0.14836467]
        )

        # Set up variational wave function
        rbm = nets.CpxRBM(numHidden=2, bias=False)
        psi = NQS(rbm, L, 2 ** L, seed=123)
        psi.parameters = weights

        # Set up exact sampler
        exactSampler = sampler.ExactSampler(psi)

        # Set up hamiltonian for time evolution and ZZ observable
        hamiltonian = 0
        ZZ = 0
        for l in range(L):
            hamiltonian += J * op.SigmaZ(l) * op.SigmaZ((l + 1) % L) + hx * op.SigmaX(l)
            ZZ += op.SigmaZ(l) * op.SigmaZ((l + 1) % L)

        loss_function = jVMC_exp.objective_function.Observable(hamiltonian)
        solver = jVMC_exp.solver.PinvSNR(snr_tol=1, pinv_tol=0.0, pinv_cutoff=1e-8)
        stepper = jVMC_exp.stepper.AdaptiveHeun(timeStep=1e-3, tol=1e-5)
        opt = jVMC_exp.optimizer.TDVP(exactSampler, psi, make_real=False, diagonalShift=0, solver=solver)

        t_max = 0.5
        observables = {'ZZ': ZZ}

        out = opt.time_evolution(t_max, loss_function, stepper, observables)
    
        # Check energy conservation
        energy = np.array(out['energy']['mean'])
        zz = np.array(out['ZZ']['mean'])
        times = np.array(out['times'])

        self.assertTrue(np.max(np.abs((energy - energy[0]) / energy[0])) < 1e-3)

        # Check observable dynamics
        zz = interp1d(times, zz)
        refTimes = np.arange(0, 0.5, 0.05)
        netZZ = zz(refTimes)
        refZZ = np.array(
            [0.882762129306284, 0.8936168721790617, 0.9257753299594491, 0.9779836185039352, 1.0482156449061142,
             1.1337654450614298, 1.231369697427413, 1.337354107391303, 1.447796176316155, 1.558696104640795,
             1.666147269524912, 1.7664978782554912, 1.8564960156892512, 1.9334113379450693, 1.9951280521882777,
             2.0402054805651546, 2.067904337137255, 2.078178742959828, 2.071635856483114, 2.049466698269522, 2.049466698269522]
        )
        max_err = np.max(np.abs(netZZ - refZZ[:len(netZZ)]))
        self.assertTrue(max_err < 1e-3)

class TestTimeEvolutionMCSampler(unittest.TestCase):
    def test_time_evolution(self):
        L = 4
        J = -1.0
        hx = -0.3

        num_samples = 2 ** 16
        num_chains = 2 ** 14
        batch_size = 2 ** 16

        weights = jnp.array(
            [0.23898957, 0.12614753, 0.19479055, 0.17325271, 0.14619853, 0.21392751,
             0.19648707, 0.17103704, -0.15457255, 0.10954413, 0.13228065, -0.14935214,
             -0.09963073, 0.17610707, 0.13386381, -0.14836467]
        )

        # Set up variational wave function
        rbm = nets.CpxRBM(numHidden=2, bias=False)
        psi = NQS(rbm, L, batch_size, seed=123)
        psi.parameters = weights

        # Set up hamiltonian for time evolution and ZZ observable
        hamiltonian = 0
        ZZ = 0
        for l in range(L):
            hamiltonian += J * op.SigmaZ(l) * op.SigmaZ((l + 1) % L) + hx * op.SigmaX(l)
            ZZ += op.SigmaZ(l) * op.SigmaZ((l + 1) % L)

        # Set up exact sampler
        proposer = jVMC_exp.propose.SpinFlip()
        mc_sampler = jVMC_exp.sampler.MCSampler(psi, proposer, 123, num_chains, num_samples, mu=1)

        loss_function = jVMC_exp.objective_function.Observable(hamiltonian)
        solver = jVMC_exp.solver.PinvSNR(snr_tol=1, pinv_cutoff=1e-8)
        stepper = jVMC_exp.stepper.AdaptiveHeun(timeStep=1e-3, tol=1e-4)
        opt = jVMC_exp.optimizer.TDVP(mc_sampler, psi, make_real=False, solver=solver, use_cross_valiadation=True)

        t_max = 0.5
        observables = {'ZZ': ZZ}

        out = opt.time_evolution(t_max, loss_function, stepper, observables)

        # Check energy conservation
        energy = np.array(out['energy']['mean'])
        zz = np.array(out['ZZ']['mean'])
        times = np.array(out['times'])

        self.assertTrue(np.max(np.abs((energy - energy[0]) / energy[0])) < 1e-1)

        # Check observable dynamics
        zz = interp1d(times, zz)
        refTimes = np.arange(0, 0.5, 0.05)
        netZZ = zz(refTimes)
        refZZ = np.array(
            [0.882762129306284, 0.8936168721790617, 0.9257753299594491, 0.9779836185039352, 1.0482156449061142,
             1.1337654450614298, 1.231369697427413, 1.337354107391303, 1.447796176316155, 1.558696104640795,
             1.666147269524912, 1.7664978782554912, 1.8564960156892512, 1.9334113379450693, 1.9951280521882777,
             2.0402054805651546, 2.067904337137255, 2.078178742959828, 2.071635856483114, 2.049466698269522, 2.049466698269522]
        )
        self.assertTrue(np.max(np.abs(netZZ - refZZ[:len(netZZ)])) < 2e-2)

if __name__ == "__main__":
    unittest.main()