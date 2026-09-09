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
        stepper = jVMC_exp.stepper.AdaptiveHeun(timeStep=1e-3, rtol=1e-5, atol=1e-5)
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
        stepper = jVMC_exp.stepper.AdaptiveHeun(timeStep=1e-3, rtol=0.0, atol=2.5e-5)
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

class TestTimeEvolutionBatchedJacobian(unittest.TestCase):
    def test_batched_jacobian_matches_dense(self):
        L = 4
        J = -1.0
        hx = -0.3

        weights = jnp.array(
            [0.23898957, 0.12614753, 0.19479055, 0.17325271, 0.14619853, 0.21392751,
             0.19648707, 0.17103704, -0.15457255, 0.10954413, 0.13228065, -0.14935214,
             -0.09963073, 0.17610707, 0.13386381, -0.14836467]
        )

        def run(batch_size, batched_jacobian):
            rbm = nets.CpxRBM(numHidden=2, bias=False)
            psi = NQS(rbm, L, batch_size, seed=123)
            psi.parameters = weights
            exactSampler = sampler.ExactSampler(psi)

            hamiltonian = 0
            for l in range(L):
                hamiltonian += J * op.SigmaZ(l) * op.SigmaZ((l + 1) % L) + hx * op.SigmaX(l)

            loss_function = jVMC_exp.objective_function.Observable(hamiltonian, batched_jacobian=batched_jacobian)
            solver = jVMC_exp.solver.PinvSNR(pinv_tol=0.0, pinv_cutoff=1e-8)
            stepper = jVMC_exp.stepper.Euler(timeStep=1e-3)
            opt = jVMC_exp.optimizer.TDVP(exactSampler, psi, make_real=False, diagonalShift=0, solver=solver)

            out = opt.time_evolution(5e-3, loss_function, stepper)
            return np.array(out['energy']['mean'])

        energy_dense = run(batch_size=2 ** L, batched_jacobian=False)
        energy_lazy = run(batch_size=6, batched_jacobian=True)

        self.assertTrue(np.allclose(energy_dense, energy_lazy, atol=1e-6))

class TestTimeEvolutionPinv(unittest.TestCase):
    def test_time_evolution_pinv(self):
        """
        Same reference dynamics as ``TestTimeEvolution``, but driven by the plain
        ``Pinv`` solver instead of ``PinvSNR``. Pins down that ``Evolution`` works with
        any dense solver, not just the SNR-regularized one it used to default to.
        """
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
        solver = jVMC_exp.solver.Pinv(pinv_cutoff=1e-8)
        stepper = jVMC_exp.stepper.AdaptiveHeun(timeStep=1e-3, rtol=1e-5, atol=1e-5)
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

    def test_pinv_matches_pinv_snr(self):
        """
        With SNR regularization off and the adaptive residual target disabled, ``PinvSNR``
        reduces to a smoothly-cutoff pseudo-inverse. On a well-conditioned problem it must
        agree with ``Pinv`` step for step.
        """
        L = 4
        J = -1.0
        hx = -0.3

        weights = jnp.array(
            [0.23898957, 0.12614753, 0.19479055, 0.17325271, 0.14619853, 0.21392751,
             0.19648707, 0.17103704, -0.15457255, 0.10954413, 0.13228065, -0.14935214,
             -0.09963073, 0.17610707, 0.13386381, -0.14836467]
        )

        def run(solver):
            rbm = nets.CpxRBM(numHidden=2, bias=False)
            psi = NQS(rbm, L, 2 ** L, seed=123)
            psi.parameters = weights
            exactSampler = sampler.ExactSampler(psi)

            hamiltonian = 0
            for l in range(L):
                hamiltonian += J * op.SigmaZ(l) * op.SigmaZ((l + 1) % L) + hx * op.SigmaX(l)

            loss_function = jVMC_exp.objective_function.Observable(hamiltonian)
            stepper = jVMC_exp.stepper.Euler(timeStep=1e-3)
            opt = jVMC_exp.optimizer.TDVP(
                exactSampler, psi, make_real=False, diagonalShift=0, solver=solver
            )
            out = opt.time_evolution(1e-2, loss_function, stepper)
            return np.array(out['energy']['mean'])

        energy_pinv = run(jVMC_exp.solver.Pinv(pinv_cutoff=1e-10))
        energy_snr = run(jVMC_exp.solver.PinvSNR(pinv_tol=0.0, pinv_cutoff=1e-10))

        self.assertTrue(np.allclose(energy_pinv, energy_snr, atol=1e-8))

class TestNonHolomorphicUpdate(unittest.TestCase):
    """
    A non-holomorphic ansatz is parametrized by a real vector (``psi.parameters_flat``
    is real even when the parameters themselves are complex), so the update handed to
    the stepper has to be real. The solver works in the -- generally complex --
    eigenbasis of the left hand side, so the projection back onto the real
    parameterization is done by ``Evolution.get_update`` and must therefore hold for
    every solver, not just the one that used to do it internally.
    """
    def _setup(self):
        L = 4
        J = -1.0
        hx = -0.3

        model = nets.TwoNets((
            nets.RBM(numHidden=3, bias=False),
            nets.RBM(numHidden=3, bias=False)
        ))
        psi = NQS(model, L, 2 ** L, seed=1234)
        self.assertFalse(psi.holomorphic)

        hamiltonian = 0
        for l in range(L):
            hamiltonian += J * op.SigmaZ(l) * op.SigmaZ((l + 1) % L) + hx * op.SigmaX(l)

        return psi, sampler.ExactSampler(psi), jVMC_exp.objective_function.Observable(hamiltonian)

    def test_update_is_real_for_every_dense_solver(self):
        for make_real in (True, False):
            for solver in (
                jVMC_exp.solver.Pinv(pinv_cutoff=1e-8),
                jVMC_exp.solver.PinvSNR(pinv_cutoff=1e-8),
            ):
                with self.subTest(make_real=make_real, solver=type(solver).__name__):
                    psi, exactSampler, loss_function = self._setup()
                    opt = jVMC_exp.optimizer.TDVP(
                        exactSampler, psi, make_real=make_real,
                        diagonalShift=1e-3, solver=solver
                    )

                    update = opt.get_update(
                        loss_function.value_and_grad(exactSampler, compute_grad=True, t=0)
                    )
                    self.assertFalse(jnp.iscomplexobj(update))
                    self.assertEqual(update.shape, psi.parameters_flat.shape)

                    # The full step must stay in the real parameterization too
                    stepper = jVMC_exp.stepper.Euler(timeStep=1e-3)
                    new_parameters, _ = opt.step(0, stepper, loss_function)
                    self.assertEqual(new_parameters.dtype, psi.parameters_flat.dtype)

    def test_solver_output_is_complex_before_projection(self):
        """
        Guards the test above against silently passing for the wrong reason: with
        ``make_real=False`` the raw solver output really is complex, so dropping the
        projection in ``Evolution.get_update`` would be caught.
        """
        psi, exactSampler, loss_function = self._setup()
        solver = jVMC_exp.solver.Pinv(pinv_cutoff=1e-8)
        opt = jVMC_exp.optimizer.TDVP(
            exactSampler, psi, make_real=False, diagonalShift=1e-3, solver=solver
        )

        objective_fn_out = loss_function.value_and_grad(exactSampler, compute_grad=True, t=0)
        b = opt._get_rhs(objective_fn_out.grad)
        A = opt._get_lhs(objective_fn_out.grad_log_psi)
        raw, _ = solver(A, b, **opt.solver_state)

        self.assertTrue(jnp.iscomplexobj(raw))
        self.assertGreater(float(jnp.max(jnp.abs(jnp.imag(raw)))), 1e-6)

if __name__ == "__main__":
    unittest.main()