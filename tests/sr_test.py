import unittest
import jax.numpy as jnp

import jVMC_exp
import jVMC_exp.nets as nets
from jVMC_exp.vqs import NQS
import jVMC_exp.operator.discrete as op
import jVMC_exp.sampler as sampler


class TestBatchedSR(unittest.TestCase):
    def test_batched_jacobian_lazy_matvec_matches_dense(self):
        L = 3
        psi = NQS(nets.RBM(numHidden=2, bias=True), L, 4, seed=123)
        exact_sampler = sampler.ExactSampler(psi)
        exact_sampler.sample()

        hamiltonian = 0
        for l in range(L):
            hamiltonian += -1.0 * op.SigmaZ(l) * op.SigmaZ((l + 1) % L) - 0.7 * op.SigmaX(l)

        loss_function = jVMC_exp.objective_function.Observable(hamiltonian)
        dense_out = loss_function.value_and_grad(exact_sampler)
        batched_out = loss_function.value_and_grad(
            exact_sampler,
            jacobian_mode="batched",
            compute_grad=False,
        )

        opt_dense = jVMC_exp.optimizer.SR(exact_sampler, psi, solver=jVMC_exp.solver.CG())
        opt_batched = jVMC_exp.optimizer.SR(
            exact_sampler,
            psi,
            solver=jVMC_exp.solver.CG(),
            jacobian_mode="batched",
        )
        S_dense = opt_dense._get_lhs_lazy(dense_out.grad_log_psi)
        S_batched = opt_batched._get_lhs_lazy(batched_out.grad_log_psi)
        v = jnp.arange(psi.numParameters, dtype=psi.parameters_flat.dtype)

        self.assertTrue(jnp.allclose(S_batched(v), S_dense(v), rtol=1e-10, atol=1e-10))

    def test_batched_jacobian_rejects_dense_solver(self):
        L = 3
        psi = NQS(nets.RBM(numHidden=2, bias=True), L, 4, seed=123)
        exact_sampler = sampler.ExactSampler(psi)

        with self.assertRaises(NotImplementedError):
            jVMC_exp.optimizer.SR(
                exact_sampler,
                psi,
                solver=jVMC_exp.solver.PinvSNR(),
                jacobian_mode="batched",
            )


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

if __name__ == "__main__":
    unittest.main()
