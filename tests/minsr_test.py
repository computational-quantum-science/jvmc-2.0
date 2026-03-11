import unittest
import tqdm
import jax.numpy as jnp

import jVMC_exp
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
        
        batch_size = int(2 ** L)
        learning_rate = 1e-2
        num_steps = 200

        for hx, exE in zip(hxs, exEs):
            # Set up variational wave function
            rbm = nets.CpxRBM(numHidden=3, bias=False)
            psi = NQS(rbm, L, batch_size, seed=1234)

            # Set up hamiltonian for ground state search
            H = 0
            for l in range(L):
                H += J * op.SigmaZ(l) * op.SigmaZ((l + 1) % L) + hx * op.SigmaX(l)

            # Set up exact sampler
            exact_sampler = sampler.ExactSampler(psi, 2)

            tdvp_equation = jVMC_exp.util.MinSR(exact_sampler, pinvTol=1e-6, diagonalShift=1e-3)
            stepper = jVMC_exp.util.Euler(timeStep=learning_rate)

            for _ in tqdm.tqdm(range(num_steps)):
                psi.parameters, _ = stepper.step(0, tdvp_equation, psi.parameters_flat, hamiltonian=H, psi=psi)

            obs = measure({"energy": H}, psi, exact_sampler)
            print(jnp.abs((obs['energy']['mean'] - exE) / exE))
            self.assertTrue(jnp.max(jnp.abs((obs['energy']['mean'] - exE) / exE)) < 1e-3)

if __name__ == "__main__":
    unittest.main()
