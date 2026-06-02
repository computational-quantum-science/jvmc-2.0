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

        for hx, exE in zip(hxs, exEs):
            # Set up variational wave function
            rbm = nets.CpxRBM(numHidden=6, bias=False)
            psi = NQS(rbm, L, 2 ** 4, seed=123)
            exactSampler = sampler.ExactSampler(psi)

            # Set up hamiltonian for ground state search
            hamiltonian = 0
            for l in range(L):
                hamiltonian += J * (op.SigmaZ(l) * op.SigmaZ((l + 1) % L)) + hx * op.SigmaX(l)

            delta = 2
            loss_function = jVMC_exp.objective_function.Observable(hamiltonian)
            solver = jVMC_exp.solver.CG()
            stepper = jVMC_exp.stepper.Euler(5e-2)
            opt = jVMC_exp.optimizer.SR(exactSampler, psi, diagonalShift=0, diagonalScale=delta, solver=solver)

            opt.ground_state_search(500, loss_function, stepper)
            eps_rel = jnp.abs((exactSampler(hamiltonian).mean.item() - exE) / exE)
            self.assertTrue(eps_rel < 1e-3)

if __name__ == "__main__":
    unittest.main()