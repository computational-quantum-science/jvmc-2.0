import unittest
import jax.numpy as jnp
import numpy as np

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
            solver = jVMC_exp.solver.Pinv(pinv_cutoff=1e-6)
            opt = jVMC_exp.optimizer.MinSR(exact_sampler, psi, solver=solver, diagonalShift=1e-3)

            opt.ground_state_search(num_steps, loss_function, stepper)

            E = exact_sampler(H)
            print(jnp.abs((E.mean.item() - exE) / exE))
            self.assertTrue(jnp.max(jnp.abs((E.mean.item() - exE) / exE)) < 1e-3)

class TestGsSearchBatchedJacobian(unittest.TestCase):
    def test_gs_search_batched_jacobian(self):
        """
        MinSR builds its (small, N_samples x N_samples) kernel matrix from
        the Jacobian via `_solve_lazy` when grad_log_psi is a LazySampledObs,
        so batched_jacobian=True must converge to the same ground state as
        the dense Jacobian path.
        """
        L = 4
        J = -1.0
        hx = -0.3
        exE = -4.09296160

        batch_size = 6
        learning_rate = 1e-2
        num_steps = 300

        rbm = nets.CpxRBM(numHidden=3, bias=False)
        psi = NQS(rbm, L, batch_size, seed=1234)

        H = 0
        for l in range(L):
            H += J * op.SigmaZ(l) * op.SigmaZ((l + 1) % L) + hx * op.SigmaX(l)

        exact_sampler = sampler.ExactSampler(psi)

        loss_function = jVMC_exp.objective_function.Observable(H, batched_jacobian=True)
        stepper = jVMC_exp.stepper.Euler(timeStep=learning_rate)
        solver = jVMC_exp.solver.Pinv(pinv_cutoff=1e-6)
        opt = jVMC_exp.optimizer.MinSR(exact_sampler, psi, solver=solver, diagonalShift=1e-3)

        opt.ground_state_search(num_steps, loss_function, stepper)

        E = exact_sampler(H)
        eps_rel = jnp.abs((E.mean.item() - exE) / exE)
        self.assertTrue(eps_rel < 1e-3)

class TestPinvSolve(unittest.TestCase):
    """
    MinSR used to pseudo-invert the tangent kernel and only then apply it to the local
    energies. ``Pinv`` instead solves in the eigenbasis, ``V (ev^-1 * (V^H b))``, which
    avoids reassembling ``A^+`` but must give the same answer.
    """
    def test_matches_explicit_pseudo_inverse(self):
        rng = np.random.default_rng(0)

        for N, complex_valued in ((64, True), (65, False), (128, True)):
            def draw(*shape):
                x = rng.standard_normal(shape)
                return x + 1j * rng.standard_normal(shape) if complex_valued else x

            # Hermitian positive semi-definite, like MinSR's tangent kernel
            X = draw(N, N)
            A = jnp.array(X @ X.conj().T) + 1e-3 * jnp.eye(N)
            b = jnp.array(draw(N))

            for cutoff in (1e-14, 1e-6, 1e-3):
                with self.subTest(N=N, complex_valued=complex_valued, cutoff=cutoff):
                    expected = jnp.linalg.pinv(A, rtol=cutoff, hermitian=True) @ b
                    x, info = jVMC_exp.solver.Pinv(pinv_cutoff=cutoff)(A, b)

                    self.assertTrue(
                        jnp.allclose(x, expected, rtol=1e-10, atol=1e-10)
                    )
                    self.assertGreater(info["condition_number"], 1.0)

    def test_reports_discarded_residual(self):
        """
        ``residual`` is the fraction of ``b`` thrown away by the cutoff; cross validation
        normalizes against it, so it has to be present and to grow with the cutoff.
        """
        rng = np.random.default_rng(1)
        X = rng.standard_normal((64, 64))
        A = jnp.array(X @ X.T) + 1e-3 * jnp.eye(64)
        b = jnp.array(rng.standard_normal(64))

        loose = jVMC_exp.solver.Pinv(pinv_cutoff=1e-14)(A, b)[1]["residual"]
        tight = jVMC_exp.solver.Pinv(pinv_cutoff=1e-1)(A, b)[1]["residual"]

        self.assertLess(loose, 1e-12)
        self.assertGreater(tight, loose)

class TestSolverCompatibility(unittest.TestCase):
    def _psi_and_sampler(self):
        L = 4
        psi = NQS(nets.CpxRBM(numHidden=3, bias=False), L, 2 ** L, seed=1234)
        return psi, sampler.ExactSampler(psi)

    def test_rejects_matrix_free_solver(self):
        psi, exact_sampler = self._psi_and_sampler()
        with self.assertRaises(ValueError):
            jVMC_exp.optimizer.MinSR(exact_sampler, psi, solver=jVMC_exp.solver.CG())

    def test_rejects_snr_regularization(self):
        psi, exact_sampler = self._psi_and_sampler()
        with self.assertRaises(ValueError):
            jVMC_exp.optimizer.MinSR(
                exact_sampler, psi, solver=jVMC_exp.solver.PinvSNR(snr_tol=1)
            )

    def test_accepts_pinv_snr_without_snr(self):
        """
        Only the SNR regularization is meaningless for MinSR; PinvSNR's adaptive
        eigenvalue cutoff is not, so snr_tol=0 has to go through and solve.
        """
        L = 4
        psi, exact_sampler = self._psi_and_sampler()

        H = 0
        for l in range(L):
            H += -1.0 * op.SigmaZ(l) * op.SigmaZ((l + 1) % L) + -0.3 * op.SigmaX(l)
        loss_function = jVMC_exp.objective_function.Observable(H)

        opt = jVMC_exp.optimizer.MinSR(
            exact_sampler, psi, solver=jVMC_exp.solver.PinvSNR(pinv_cutoff=1e-6)
        )
        new_parameters, _ = opt.step(0, jVMC_exp.stepper.Euler(timeStep=1e-2), loss_function)

        self.assertEqual(new_parameters.shape, psi.parameters_flat.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(jnp.abs(new_parameters)))))
        # info from the solve must reach the metadata MinSR reports
        self.assertIn("residual", opt.meta_data)

if __name__ == "__main__":
    unittest.main()