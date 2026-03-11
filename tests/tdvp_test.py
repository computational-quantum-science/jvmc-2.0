import unittest
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import interp1d
import tqdm

import jVMC_exp
import jVMC_exp.util.stepper as jVMCstepper
import jVMC_exp.nets as nets
from jVMC_exp.vqs import NQS
import jVMC_exp.operator.discrete as op
import jVMC_exp.sampler as sampler
from jVMC_exp.util import measure

class TestGsSearch(unittest.TestCase):
    def test_gs_search_cpx(self):
        L = 4
        J = -1.0
        hxs = [-1.3, -0.3]
        exEs = [-6.10160339, -4.09296160]

        for hx, exE in zip(hxs, exEs):
            # Set up variational wave function
            rbm = nets.CpxRBM(numHidden=6, bias=False)
            psi = NQS(rbm, L, 2 ** 4)
            exactSampler = sampler.ExactSampler(psi)

            # Set up hamiltonian for ground state search
            hamiltonian = 0
            for l in range(L):
                hamiltonian += J * (op.SigmaZ(l) * op.SigmaZ((l + 1) % L)) + hx * op.SigmaX(l)

            delta = 2
            tdvpEquation = jVMC_exp.util.TDVP(
                exactSampler, snrTol=1, pinvTol=0.0, pinvCutoff=1e-8, 
                rhsPrefactor=1., diagonalShift=delta, makeReal='real'
            )
            stepper = jVMC_exp.util.Euler(5e-2)

            for _ in tqdm.tqdm(range(100)):
                psi.parameters, _ = stepper.step(0, tdvpEquation, psi.parameters_flat, psi=psi, hamiltonian=hamiltonian)

            obs = measure({"energy": hamiltonian}, psi, exactSampler)

            self.assertTrue(jnp.max(jnp.abs((obs['energy']['mean'] - exE) / exE)) < 1e-3)

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
        psi = NQS(rbm, L, 2 ** L)
        psi.parameters = weights

        # Set up exact sampler
        exactSampler = sampler.ExactSampler(psi)

        # Set up hamiltonian for time evolution
        hamiltonian = 0
        for l in range(L):
            hamiltonian += J * op.SigmaZ(l) * op.SigmaZ((l + 1) % L) + hx * op.SigmaX(l)

        # Set up ZZ observable
        ZZ = 0
        for l in range(L):
            ZZ += op.SigmaZ(l) * op.SigmaZ((l + 1) % L)

        # Set up adaptive time stepper
        stepper = jVMCstepper.AdaptiveHeun(timeStep=1e-3, tol=1e-5)
        tdvpEquation = jVMC_exp.util.TDVP(exactSampler, snrTol=1, pinvTol=0.0, pinvCutoff=1e-8, rhsPrefactor=1.j, diagonalShift=0., makeReal='imag')

        t = 0
        t_max = 0.5
        obs = []
        times = []
        times.append(t)
        newMeas = measure({'E': hamiltonian, 'ZZ': ZZ}, psi, exactSampler)
        obs.append([newMeas['E']['mean'], newMeas['ZZ']['mean']])

        pbar = tqdm.tqdm(total=t_max)
        while t < t_max:
            psi.parameters, dt = stepper.step(0, tdvpEquation, psi.parameters_flat, hamiltonian=hamiltonian, psi=psi)
            t += dt
            times.append(t)
            newMeas = measure({'E': hamiltonian, 'ZZ': ZZ}, psi, exactSampler)
            obs.append([newMeas['E']['mean'], newMeas['ZZ']['mean']])
            pbar.update(float(dt))
        pbar.close()

        obs = np.array(obs)

        # Check energy conservation
        obs[:, 0] = np.abs((obs[:, 0] - obs[0, 0]) / obs[0, 0])
        self.assertTrue(np.max(obs[:, 0]) < 1e-3)

        # Check observable dynamics
        zz = interp1d(np.array(times), obs[:, 1])
        refTimes = np.arange(0, 0.5, 0.05)
        netZZ = zz(refTimes)
        refZZ = np.array(
            [0.882762129306284, 0.8936168721790617, 0.9257753299594491, 0.9779836185039352, 1.0482156449061142,
             1.1337654450614298, 1.231369697427413, 1.337354107391303, 1.447796176316155, 1.558696104640795,
             1.666147269524912, 1.7664978782554912, 1.8564960156892512, 1.9334113379450693, 1.9951280521882777,
             2.0402054805651546, 2.067904337137255, 2.078178742959828, 2.071635856483114, 2.049466698269522, 2.049466698269522]
        )
        self.assertTrue(np.max(np.abs(netZZ - refZZ[:len(netZZ)])) < 1e-3)

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
        psi = NQS(rbm, L, batch_size)
        psi.parameters = weights

        # Set up hamiltonian for time evolution
        hamiltonian = 0
        for l in range(L):
            hamiltonian += J * op.SigmaZ(l) * op.SigmaZ((l + 1) % L) + hx * op.SigmaX(l)

        # Set up ZZ observable
        ZZ = 0
        for l in range(L):
            ZZ += op.SigmaZ(l) * op.SigmaZ((l + 1) % L)

        # Set up exact sampler
        proposer = jVMC_exp.propose.SpinFlip()
        mc_sampler = sampler.MCSampler(psi, proposer, 123, num_chains, num_samples, mu=1)

        # Set up adaptive time stepper
        stepper = jVMCstepper.AdaptiveHeun(timeStep=1e-3, tol=1e-4)

        tdvpEquation = jVMC_exp.util.TDVP(mc_sampler, snrTol=1, pinvTol=1e-8, rhsPrefactor=1.j, diagonalShift=0., makeReal='imag', crossValidation=True)

        t = 0
        t_max = 0.2
        obs = []
        times = []
        times.append(t)
        newMeas = measure({'E': hamiltonian, 'ZZ': ZZ}, psi, mc_sampler)
        obs.append([newMeas['E']['mean'], newMeas['ZZ']['mean']])
        pbar = tqdm.tqdm(total=t_max)
        while t < t_max:
            psi.parameters, dt = stepper.step(0, tdvpEquation, psi.parameters_flat, hamiltonian=hamiltonian, psi=psi)
            t += dt
            times.append(t)
            newMeas = measure({'E': hamiltonian, 'ZZ': ZZ}, psi, mc_sampler)
            obs.append([newMeas['E']['mean'], newMeas['ZZ']['mean']])
            pbar.update(float(dt))
        pbar.close()

        obs = np.array(obs)

        # Check energy conservation
        obs[:, 0] = np.abs((obs[:, 0] - obs[0, 0]) / obs[0, 0])
        self.assertTrue(np.max(obs[:, 0]) < 1e-1)

        # Check observable dynamics
        zz = interp1d(np.array(times), obs[:, 1])
        refTimes = np.arange(0, 0.2, 0.05)
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
