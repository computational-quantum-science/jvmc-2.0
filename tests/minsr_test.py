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
        num_steps = 150

        for hx, exE in zip(hxs, exEs):
            # Set up variational wave function
            rbm = nets.CpxRBM(numHidden=3, bias=False)
            psi = NQS(rbm, L, batch_size, seed=1234)

            # Set up hamiltonian for ground state search
            H = 0
            for l in range(L):
                H += J * op.SigmaZ(l) * op.SigmaZ((l + 1) % L) + hx * op.SigmaX(l)

            # Set up exact sampler
            exact_sampler = sampler.ExactSampler(psi, lDim=2)

            # set up the minsr solver
            solver = jVMC_exp.optimizer.MinSR(exact_sampler, psi, pinvTol=1e-6, diagonalShift=1e-3, centered=True)
            stepper = jVMC_exp.stepper.Euler(timeStep=learning_rate)
            objective = jVMC_exp.objective_function.Observable(operator=H)

            for _ in tqdm.tqdm(range(num_steps)):
                psi.parameters, _ = stepper.step(0, solver, psi.parameters_flat, objective_function=objective)

            obs = solver.o_loc.mean[0]
            print(jnp.abs((obs - exE) / exE))
            self.assertTrue(jnp.max(jnp.abs((obs - exE) / exE)) < 1e-3)

if __name__ == "__main__":
    unittest.main()
