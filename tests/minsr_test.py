import unittest
import jax.numpy as jnp

import jVMC_exp
import jVMC_exp.nets as nets
from jVMC_exp.vqs import NQS
import jVMC_exp.operator.discrete as op
import jVMC_exp.sampler as sampler

class TestGsSearch(unittest.TestCase):
    def test_gs_search_cpx(self):
        L = 4
        J = -1.0
        hxs = [-1.3, -0.3]
        exEs = [-6.10160339, -4.09296160]
        
        batch_size = int(2 ** L)
        learning_rate = 1e-2
        num_steps = 300

        for hx, exE in zip(hxs, exEs):
            # Set up variational wave function
            rbm = nets.CpxRBM(numHidden=3, bias=False)
            psi = NQS(rbm, L, batch_size, seed=1234)

            # Set up hamiltonian for ground state search
            H = 0
            for l in range(L):
                H += J * op.SigmaZ(l) * op.SigmaZ((l + 1) % L) + hx * op.SigmaX(l)

            # Set up exact sampler
            exact_sampler = sampler.ExactSampler(psi)
            
            loss_function = jVMC_exp.objective_function.Observable(H)
            stepper = jVMC_exp.stepper.Euler(timeStep=learning_rate)
            opt = jVMC_exp.optimizer.MinSR(exact_sampler, psi, pinv_tol=1e-6, diagonalShift=1e-3)

            opt.ground_state_search(num_steps, loss_function, stepper)

            E = exact_sampler(H)
            print(jnp.abs((E.mean.item() - exE) / exE))
            self.assertTrue(jnp.max(jnp.abs((E.mean.item() - exE) / exE)) < 1e-3)

if __name__ == "__main__":
    unittest.main()