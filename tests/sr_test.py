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
            solver = jVMC_exp.solver.PinvSNR(snr_tol=1, pinv_tol=0.0, pinv_cutoff=1e-8)
            stepper = jVMC_exp.stepper.Euler(5e-2)
            opt = jVMC_exp.optimizer.SR(exactSampler, psi, diagonalShift=0, diagonalScale=delta, solver=solver)

            opt.ground_state_search(500, loss_function, stepper)
            eps_rel = jnp.abs((exactSampler(hamiltonian).mean.item() - exE) / exE)
            self.assertTrue(eps_rel < 1e-3)

class TestGsSearchBatchedJacobian(unittest.TestCase):
    def test_gs_search_batched_jacobian_dense_solver(self):
        L = 4
        J = -1.0
        hx = -0.3
        exE = -4.09296160

        rbm = nets.CpxRBM(numHidden=6, bias=False)
        psi = NQS(rbm, L, 2 ** 4, seed=123)
        exactSampler = sampler.ExactSampler(psi)

        hamiltonian = 0
        for l in range(L):
            hamiltonian += J * (op.SigmaZ(l) * op.SigmaZ((l + 1) % L)) + hx * op.SigmaX(l)

        delta = 2
        loss_function = jVMC_exp.objective_function.Observable(hamiltonian, batched_jacobian=True)
        solver = jVMC_exp.solver.PinvSNR(snr_tol=1, pinv_tol=0.0, pinv_cutoff=1e-8)
        stepper = jVMC_exp.stepper.Euler(5e-2)
        opt = jVMC_exp.optimizer.SR(exactSampler, psi, diagonalShift=0, diagonalScale=delta, solver=solver)

        opt.ground_state_search(500, loss_function, stepper)
        eps_rel = jnp.abs((exactSampler(hamiltonian).mean.item() - exE) / exE)
        self.assertTrue(eps_rel < 1e-3)

    def test_matrix_free_solver_rejects_batched_jacobian(self):
        """
        CG is matrix-free: it builds its matvec directly from the fully
        materialized Jacobian, so it cannot accept a batched (LazySampledObs)
        Jacobian -- this must raise loudly instead of silently misbehaving.
        """
        L = 4
        rbm = nets.CpxRBM(numHidden=6, bias=False)
        psi = NQS(rbm, L, 2 ** 4, seed=123)
        exactSampler = sampler.ExactSampler(psi)

        hamiltonian = 0
        for l in range(L):
            hamiltonian += -1.0 * (op.SigmaZ(l) * op.SigmaZ((l + 1) % L)) - 0.3 * op.SigmaX(l)

        loss_function = jVMC_exp.objective_function.Observable(hamiltonian, batched_jacobian=True)
        solver = jVMC_exp.solver.CG()
        stepper = jVMC_exp.stepper.Euler(5e-2)
        opt = jVMC_exp.optimizer.SR(exactSampler, psi, diagonalShift=0, solver=solver)

        with self.assertRaises(ValueError):
            opt.ground_state_search(1, loss_function, stepper)

if __name__ == "__main__":
    unittest.main()