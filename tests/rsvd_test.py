import unittest
import warnings
import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import interp1d

import jVMC_exp
import jVMC_exp.nets as nets
from jVMC_exp.vqs import NQS
import jVMC_exp.operator.discrete as op
import jVMC_exp.sampler as sampler
from jax.sharding import PartitionSpec as P
from jVMC_exp.sharding_config import MESH, DEVICE_SHARDING
from jVMC_exp.solver.qgt_operator import (
    QGTOperator, ShardedQGTOperator, ParameterLayout, parameter_mesh, SAMPLES_AXIS, PARAMS_AXIS
)
from jVMC_exp.solver.rsvd import RandomizedPinvSNR, randomized_eigh


def _random_jacobian(key, num_samples, num_params, decay=0.5, complex_valued=True):
    """
    A 'Jacobian' with a geometrically decaying singular spectrum, so that the
    dominant subspace is well separated and the randomized method converges.
    """
    k1, k2, k3 = jax.random.split(key, 3)
    U, _ = jnp.linalg.qr(jax.random.normal(k1, (num_samples, num_params)))
    W, _ = jnp.linalg.qr(jax.random.normal(k2, (num_params, num_params)))
    s = decay ** jnp.arange(num_params)
    jac = (U * s) @ W.T
    if complex_valued:
        phase = jnp.exp(1j * jax.random.uniform(k3, (num_params,), maxval=2 * np.pi))
        jac = jac * phase

    return jax.device_put(jac, DEVICE_SHARDING)


def _projector_distance(V1, V2):
    P1 = V1 @ jnp.conj(V1).T
    P2 = V2 @ jnp.conj(V2).T

    return float(jnp.max(jnp.abs(P1 - P2)))


class TestQGTOperator(unittest.TestCase):
    def setUp(self):
        self.N = 8 * MESH.size
        self.n = 6
        self.jac = _random_jacobian(jax.random.PRNGKey(0), self.N, self.n)
        self.S0 = jnp.conj(self.jac).T @ self.jac

    def test_matvec_matches_dense(self):
        """The operator is the matrix q[S_0], also on complex vectors."""
        k1, k2 = jax.random.split(jax.random.PRNGKey(1))
        v_real = jax.random.normal(k1, (self.n,))
        v_cpx = v_real + 1j * jax.random.normal(k2, (self.n,))
        for mode, trans in (
            ("identity", lambda x: x),
            ("real", jnp.real),
            ("imag", lambda x: 1j * jnp.imag(x)),
        ):
            with self.subTest(mode=mode):
                A = QGTOperator(self.jac, mode, diag_scale=0.0, diag_shift=0.0)
                S = trans(self.S0)
                self.assertEqual(A.shape, (self.n, self.n))
                for v in (v_real, v_cpx):
                    self.assertTrue(jnp.allclose(A(v), S @ v, atol=1e-12))
                    self.assertTrue(jnp.allclose(A.raw(v), self.S0 @ v, atol=1e-12))

    def test_block_product_and_diagonal_terms(self):
        Q = jax.random.normal(jax.random.PRNGKey(2), (self.n, 3))
        A = QGTOperator(self.jac, "real", diag_scale=0.3, diag_shift=0.1)
        diag = jnp.real(jnp.diag(self.S0))
        S = jnp.real(self.S0) + 0.3 * jnp.diag(diag) + 0.1 * jnp.eye(self.n)
        self.assertTrue(jnp.allclose(A(Q), S @ Q, atol=1e-12))
        self.assertTrue(jnp.allclose(A(Q[:, 0]), S @ Q[:, 0], atol=1e-12))

    def test_chunked_product_matches(self):
        """
        The product is summed over chunks of samples and accumulated in blocks of
        parameter rows; a tiny chunk budget (one sample per chunk, several blocks)
        must not change it, for any reduction and real or complex input.
        """
        k1, k2 = jax.random.split(jax.random.PRNGKey(3))
        Q_real = jax.random.normal(k1, (self.n, 2))
        Q_cpx = Q_real + 1j * jax.random.normal(k2, (self.n, 2))
        trans = {"identity": lambda x: x, "real": jnp.real, "imag": lambda x: 1j * jnp.imag(x)}
        old = jVMC_exp.global_defs.MATMUL_CHUNK_BYTES
        try:
            jVMC_exp.global_defs.update(matmul_chunk_bytes=64)
            for mode, tr in trans.items():
                A = QGTOperator(self.jac, mode, diag_scale=0.0, diag_shift=0.0)
                for Q in (Q_real, Q_cpx, Q_real[:, 0]):
                    with self.subTest(mode=mode, dtype=str(Q.dtype), ndim=Q.ndim):
                        self.assertTrue(jnp.allclose(A(Q), tr(self.S0) @ Q, atol=1e-12))
                        self.assertTrue(jnp.allclose(A.raw(Q), self.S0 @ Q, atol=1e-12))
        finally:
            jVMC_exp.global_defs.update(matmul_chunk_bytes=old)

    def test_varying_zeros_fallback(self):
        """
        JAX versions without ``jax.lax.pcast`` / ``pvary`` get the accumulator seeded
        from the Jacobian instead; the product must not change.
        """
        from jVMC_exp.solver import qgt_operator
        Q = jax.random.normal(jax.random.PRNGKey(5), (self.n, 2))
        old_api, old_chunk = qgt_operator._USE_VARYING_API, jVMC_exp.global_defs.MATMUL_CHUNK_BYTES
        try:
            qgt_operator._USE_VARYING_API = False
            jVMC_exp.global_defs.update(matmul_chunk_bytes=64)
            for mode, tr in (("identity", lambda x: x), ("real", jnp.real), ("imag", lambda x: 1j * jnp.imag(x))):
                A = QGTOperator(self.jac, mode, diag_scale=0.0, diag_shift=0.0)
                self.assertTrue(jnp.allclose(A(Q), tr(self.S0) @ Q, atol=1e-12))
                self.assertTrue(jnp.allclose(A(Q + 0j), tr(self.S0) @ Q, atol=1e-12))
        finally:
            qgt_operator._USE_VARYING_API = old_api
            jVMC_exp.global_defs.update(matmul_chunk_bytes=old_chunk)

    def test_real_reduction_stays_in_real_arithmetic(self):
        """For q = Re and a real block the result is real (no complex accumulator)."""
        Q = jax.random.normal(jax.random.PRNGKey(4), (self.n, 3))
        A = QGTOperator(self.jac, "real", diag_scale=0.0, diag_shift=0.0)
        out = A(Q)
        self.assertFalse(jnp.iscomplexobj(out))
        self.assertTrue(jnp.allclose(out, jnp.real(self.S0) @ Q, atol=1e-12))
        A = QGTOperator(self.jac, "imag", diag_scale=0.0, diag_shift=0.0)
        self.assertTrue(jnp.allclose(A(Q), 1j * jnp.imag(self.S0) @ Q, atol=1e-12))


class TestRandomizedEigh(unittest.TestCase):
    def setUp(self):
        self.N = 16 * MESH.size
        self.n = 12
        self.jac = _random_jacobian(jax.random.PRNGKey(10), self.N, self.n, decay=0.6)
        self.S0 = jnp.conj(self.jac).T @ self.jac

    def _check_against_dense(self, A, S, k, n_oversamples, n_iter, which="LM",
                             tol=1e-8, mode="device"):
        ev, V = randomized_eigh(
            A, k, jax.random.PRNGKey(5), n_oversamples=n_oversamples, n_iter=n_iter,
            which=which, diagonalization_mode=mode
        )
        ev_ref, V_ref = jnp.linalg.eigh(S)
        if which == "LM":
            idx = jnp.argsort(jnp.abs(ev_ref))[::-1][:k]
        else:
            idx = jnp.argsort(ev_ref)[::-1][:k]
        idx = idx[jnp.argsort(ev_ref[idx])]

        self.assertEqual(ev.shape, (k,))
        self.assertEqual(V.shape, (self.n, k))
        # ascending, as jnp.linalg.eigh returns them
        self.assertTrue(bool(jnp.all(jnp.diff(ev) >= 0)))
        self.assertTrue(jnp.allclose(ev, ev_ref[idx], atol=tol, rtol=tol), msg=f"{ev} vs {ev_ref[idx]}")
        self.assertLess(_projector_distance(V, V_ref[:, idx]), 1e2 * tol)
        # Ritz pairs are eigenpairs of the operator
        self.assertTrue(jnp.allclose(A(V), V * ev, atol=1e2 * tol))

        return ev, V

    def test_full_rank_is_exact(self):
        for mode, S in (
            ("identity", self.S0),
            ("real", jnp.real(self.S0)),
            ("imag", 1j * jnp.imag(self.S0)),
        ):
            with self.subTest(mode=mode):
                A = QGTOperator(self.jac, mode, 0.0, 0.0)
                self._check_against_dense(A, S, k=self.n, n_oversamples=0, n_iter=1, tol=1e-9)

    def test_low_rank_captures_dominant_eigenpairs(self):
        A = QGTOperator(self.jac, "identity", 0.0, 0.0)
        self._check_against_dense(A, self.S0, k=4, n_oversamples=4, n_iter=4, tol=1e-6)
        A = QGTOperator(self.jac, "real", 0.0, 0.0)
        self._check_against_dense(A, jnp.real(self.S0), k=4, n_oversamples=4, n_iter=4, tol=1e-6)
        self._check_against_dense(A, jnp.real(self.S0), k=4, n_oversamples=4, n_iter=4, which="LA", tol=1e-6)

    def test_imag_mode_keeps_real_iterate_and_matches_complex_path(self):
        A = QGTOperator(self.jac, "imag", 0.0, 0.0)
        M = 1j * jnp.imag(self.S0)
        self._check_against_dense(A, M, k=6, n_oversamples=4, n_iter=4, tol=1e-6)
        # With diagonal terms the operator is no longer i * (real), the generic path is used
        A = QGTOperator(self.jac, "imag", 0.0, 0.05)
        self._check_against_dense(A, M + 0.05 * jnp.eye(self.n), k=self.n, n_oversamples=0, n_iter=1, tol=1e-9)

    def test_diagonal_terms_enter_the_operator(self):
        A = QGTOperator(self.jac, "real", 0.2, 0.01)
        S = jnp.real(self.S0)
        S = S + 0.2 * jnp.diag(jnp.diag(S)) + 0.01 * jnp.eye(self.n)
        self._check_against_dense(A, S, k=self.n, n_oversamples=0, n_iter=1, tol=1e-9)

    def test_distributed_mode_pads_the_small_matrix(self):
        """
        On CPU the distributed diagonalization falls back to the device one, but the
        padding of the Rayleigh-Ritz matrix to a multiple of the number of devices is
        exercised and must not change the result.
        """
        A = QGTOperator(self.jac, "identity", 0.0, 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._check_against_dense(
                A, self.S0, k=5, n_oversamples=2, n_iter=3, tol=1e-6, mode="distributed"
            )

    def test_rejects_bad_arguments(self):
        A = QGTOperator(self.jac, "identity", 0.0, 0.0)
        with self.assertRaises(ValueError):
            randomized_eigh(A, self.n + 1, jax.random.PRNGKey(0))
        with self.assertRaises(ValueError):
            randomized_eigh(A, 0, jax.random.PRNGKey(0))
        with self.assertRaises(ValueError):
            randomized_eigh(A, 2, jax.random.PRNGKey(0), which="SM")

    def test_oversampling_is_capped_at_the_dimension(self):
        A = QGTOperator(self.jac, "identity", 0.0, 0.0)
        ev, V = randomized_eigh(A, self.n, jax.random.PRNGKey(0), n_oversamples=100, n_iter=1)
        ev_ref, _ = jnp.linalg.eigh(self.S0)
        self.assertTrue(jnp.allclose(ev, ev_ref, atol=1e-9))


def _reference_problem(batch_size=2 ** 4):
    L = 4
    J = -1.0
    hx = -0.3

    weights = jnp.array(
        [0.23898957, 0.12614753, 0.19479055, 0.17325271, 0.14619853, 0.21392751,
         0.19648707, 0.17103704, -0.15457255, 0.10954413, 0.13228065, -0.14935214,
         -0.09963073, 0.17610707, 0.13386381, -0.14836467]
    )

    rbm = nets.CpxRBM(numHidden=2, bias=False)
    psi = NQS(rbm, L, batch_size, seed=123)
    psi.parameters = weights
    exactSampler = sampler.ExactSampler(psi)

    hamiltonian = 0
    ZZ = 0
    for l in range(L):
        hamiltonian += J * op.SigmaZ(l) * op.SigmaZ((l + 1) % L) + hx * op.SigmaX(l)
        ZZ += op.SigmaZ(l) * op.SigmaZ((l + 1) % L)

    return psi, exactSampler, hamiltonian, ZZ


REF_ZZ = np.array(
    [0.882762129306284, 0.8936168721790617, 0.9257753299594491, 0.9779836185039352, 1.0482156449061142,
     1.1337654450614298, 1.231369697427413, 1.337354107391303, 1.447796176316155, 1.558696104640795,
     1.666147269524912, 1.7664978782554912, 1.8564960156892512, 1.9334113379450693, 1.9951280521882777,
     2.0402054805651546, 2.067904337137255, 2.078178742959828, 2.071635856483114, 2.049466698269522, 2.049466698269522]
)


class TestTimeEvolutionRandomizedPinvSNR(unittest.TestCase):
    def test_solver_is_matrix_free(self):
        self.assertFalse(RandomizedPinvSNR(n_components=4)._needs_dense_matrix)

    def test_time_evolution_full_rank(self):
        psi, exactSampler, hamiltonian, ZZ = _reference_problem()
        # holomorphic: the Jacobian has one complex column per parameter
        n_params = psi.numParameters

        loss_function = jVMC_exp.objective_function.Observable(hamiltonian)
        solver = RandomizedPinvSNR(
            n_components=n_params, n_oversamples=0, n_iter=1,
            snr_tol=1, pinv_tol=0.0, pinv_cutoff=1e-8
        )
        stepper = jVMC_exp.stepper.AdaptiveHeun(timeStep=1e-3, rtol=1e-5, atol=1e-5)
        opt = jVMC_exp.optimizer.TDVP(exactSampler, psi, make_real=False, diagonalShift=0, solver=solver)

        out, meta = opt.time_evolution(0.5, loss_function, stepper, {'ZZ': ZZ}, save_meta_data=True)

        energy = np.array(out['energy']['mean'])
        times = np.array(out['times'])
        self.assertTrue(np.max(np.abs((energy - energy[0]) / energy[0])) < 1e-3)

        zz = interp1d(times, np.array(out['ZZ']['mean']))
        refTimes = np.arange(0, 0.5, 0.05)
        netZZ = zz(refTimes)
        self.assertTrue(np.max(np.abs(netZZ - REF_ZZ[:len(netZZ)])) < 1e-3)

        # Diagnostics of the randomized solve are reported
        for key in ("residual", "true_residual", "pinv_cutoff", "spectrum", "n_components", "tdvp_error"):
            self.assertIn(key, meta)
        self.assertEqual(np.asarray(meta["spectrum"][-1]).shape, (n_params,))
        self.assertLess(float(np.max(meta["true_residual"])), 1e-6)

    def test_full_rank_matches_pinv_snr(self):
        """
        At full rank the randomized decomposition is exact, so the update has to agree
        with the dense PinvSNR solve step for step.
        """
        def run(solver):
            psi, exactSampler, hamiltonian, _ = _reference_problem()
            loss_function = jVMC_exp.objective_function.Observable(hamiltonian)
            stepper = jVMC_exp.stepper.Euler(timeStep=1e-3)
            opt = jVMC_exp.optimizer.TDVP(exactSampler, psi, make_real=False, diagonalShift=0, solver=solver)
            out = opt.time_evolution(1e-2, loss_function, stepper)
            return np.array(out['energy']['mean'])

        n_params = _reference_problem()[0].numParameters
        energy_dense = run(jVMC_exp.solver.PinvSNR(pinv_tol=0.0, pinv_cutoff=1e-10))
        energy_rsvd = run(RandomizedPinvSNR(n_components=n_params, n_iter=1, pinv_tol=0.0, pinv_cutoff=1e-10))
        self.assertTrue(np.allclose(energy_dense, energy_rsvd, atol=1e-8))

    def test_low_rank_solve_is_close_to_dense(self):
        """
        Dropping the eigenvalues far below the pinv cutoff does not change the update,
        so a rank below the number of parameters reproduces the dense solve as long as
        the retained spectrum reaches the cutoff.
        """
        psi, exactSampler, hamiltonian, _ = _reference_problem()
        loss_function = jVMC_exp.objective_function.Observable(hamiltonian)
        objective_fn_out = loss_function.value_and_grad(exactSampler, compute_grad=True, t=0)

        dense = jVMC_exp.optimizer.TDVP(
            exactSampler, psi, make_real=False, diagonalShift=0,
            solver=jVMC_exp.solver.PinvSNR(pinv_tol=0.0, pinv_cutoff=1e-6)
        )
        update_dense = dense.get_update(objective_fn_out)
        spectrum = np.asarray(dense.meta_data.get("spectrum", dense._additional_info["spectrum"]))
        n_above = int(np.sum(np.abs(spectrum / spectrum[-1]) > 1e-8))
        self.assertLess(n_above, spectrum.size)

        rsvd = jVMC_exp.optimizer.TDVP(
            exactSampler, psi, make_real=False, diagonalShift=0,
            solver=RandomizedPinvSNR(
                n_components=n_above, n_oversamples=2, n_iter=4, pinv_tol=0.0, pinv_cutoff=1e-6
            )
        )
        update_rsvd = rsvd.get_update(objective_fn_out)
        self.assertTrue(np.allclose(update_dense, update_rsvd, atol=1e-5), msg=f"{update_dense}\n{update_rsvd}")

    def test_batched_jacobian_is_rejected(self):
        psi, exactSampler, hamiltonian, _ = _reference_problem(batch_size=3 * jax.device_count())
        loss_function = jVMC_exp.objective_function.Observable(hamiltonian, batched_jacobian=True)
        opt = jVMC_exp.optimizer.TDVP(
            exactSampler, psi, make_real=False, diagonalShift=0,
            solver=RandomizedPinvSNR(n_components=4)
        )
        with self.assertRaises(ValueError):
            opt.time_evolution(1e-3, loss_function, jVMC_exp.stepper.Euler(timeStep=1e-3))

    def test_mc_sampler_with_snr_regularization(self):
        psi, _, hamiltonian, _ = _reference_problem(batch_size=2 ** 10)
        mc_sampler = jVMC_exp.sampler.MCSampler(
            psi, jVMC_exp.propose.SpinFlip(), 123, 2 ** 6, 2 ** 10, mu=1
        )
        loss_function = jVMC_exp.objective_function.Observable(hamiltonian)
        for snr_mode in ("empiric", "gaussian"):
            with self.subTest(snr_mode=snr_mode):
                solver = RandomizedPinvSNR(
                    n_components=6, n_oversamples=2, n_iter=2, snr_tol=1, snr_mode=snr_mode
                )
                opt = jVMC_exp.optimizer.TDVP(mc_sampler, psi, make_real=False, solver=solver)
                stepper = jVMC_exp.stepper.Euler(timeStep=1e-3)
                opt.time_evolution(3e-3, loss_function, stepper)
                self.assertEqual(np.asarray(opt.meta_data["snr"]).shape, (6,))
                self.assertEqual(np.asarray(opt.meta_data["spectrum"]).shape, (6,))


class TestStageTimings(unittest.TestCase):
    """RandomizedPinvSNR(stage_timings=True) reports the wall time of every stage."""

    def test_stage_timings_in_info(self):
        for layout in ("samples", "parameters"):
            with self.subTest(layout=layout):
                psi, exactSampler, hamiltonian, _ = _reference_problem()
                loss_function = jVMC_exp.objective_function.Observable(hamiltonian)
                objective_fn_out = loss_function.value_and_grad(exactSampler, compute_grad=True, t=0)
                opt = jVMC_exp.optimizer.TDVP(
                    exactSampler, psi, make_real=False, diagonalShift=0,
                    solver=RandomizedPinvSNR(n_components=6, n_oversamples=2, n_iter=1, layout=layout, stage_timings=True)
                )
                opt.get_update(objective_fn_out)
                info = opt._additional_info
                for key in ("t_probe", "t_apply", "t_qr", "t_ritz_matrix", "t_eigh", "t_ritz_vectors", "t_rsvd"):
                    self.assertIn(key, info)
                    self.assertGreaterEqual(info[key], 0.0)
                self.assertAlmostEqual(
                    info["t_rsvd"], sum(v for k, v in info.items() if k.startswith("t_") and k != "t_rsvd"), places=9
                )
                if layout == "parameters":
                    self.assertIn("t_redistribute", info)
                # off by default
                opt = jVMC_exp.optimizer.TDVP(
                    exactSampler, psi, make_real=False, diagonalShift=0,
                    solver=RandomizedPinvSNR(n_components=6, n_oversamples=2, n_iter=1, layout=layout)
                )
                opt.get_update(objective_fn_out)
                self.assertNotIn("t_rsvd", opt._additional_info)


class TestLBFGSRefinement(unittest.TestCase):
    """The optional L-BFGS refinement recovers the force discarded by the truncation."""

    def _problem(self):
        psi, exactSampler, hamiltonian, _ = _reference_problem()
        loss_function = jVMC_exp.objective_function.Observable(hamiltonian)
        objective_fn_out = loss_function.value_and_grad(exactSampler, compute_grad=True, t=0)

        def update_with(solver, make_real=False):
            opt = jVMC_exp.optimizer.TDVP(
                exactSampler, psi, make_real=make_real, diagonalShift=0, solver=solver
            )
            update = opt.get_update(objective_fn_out)
            return np.asarray(update), dict(opt._additional_info)

        return update_with

    def test_refinement_reduces_true_residual_towards_dense_solution(self):
        update_with = self._problem()
        dense, _ = update_with(jVMC_exp.solver.PinvSNR(pinv_tol=0.0, pinv_cutoff=1e-6))

        kwargs = dict(n_components=2, n_oversamples=1, n_iter=2, pinv_tol=0.0, pinv_cutoff=1e-6)
        truncated, info_t = update_with(RandomizedPinvSNR(**kwargs))
        refined, info_r = update_with(
            RandomizedPinvSNR(**kwargs, lbfgs=True, lbfgs_tol=1e-6, lbfgs_maxiter=500)
        )

        # the truncation discards force, the refinement recovers it
        self.assertGreater(info_t["true_residual"], 1e-3)
        self.assertAlmostEqual(info_r["residual_rsvd"], info_t["true_residual"], places=10)
        self.assertLess(info_r["true_residual"], info_r["residual_rsvd"])
        self.assertLess(info_r["true_residual"], 1e-5)
        self.assertGreaterEqual(info_r["n_lbfgs"], 1)
        self.assertGreater(info_r["lbfgs_iterations"], 0)
        self.assertGreater(info_r["t_lbfgs"], 0.0)
        self.assertLess(info_r["cos_lbfgs"], 1.0)
        self.assertLess(
            np.linalg.norm(refined - dense), 0.1 * np.linalg.norm(truncated - dense)
        )
        # the regularized quantities of the truncated basis are untouched
        self.assertAlmostEqual(info_r["residual"], info_t["residual"], places=10)

    def test_refinement_is_skipped_below_tolerance(self):
        update_with = self._problem()
        n_params = _reference_problem()[0].numParameters
        plain, _ = update_with(RandomizedPinvSNR(n_components=n_params, n_iter=1, pinv_tol=0.0))
        refined, info = update_with(
            RandomizedPinvSNR(n_components=n_params, n_iter=1, pinv_tol=0.0, lbfgs=True, lbfgs_tol=1e-3)
        )
        self.assertEqual(info["n_lbfgs"], 0)
        self.assertEqual(info["cos_lbfgs"], 1.0)
        self.assertTrue(np.allclose(plain, refined))

    def test_refinement_for_non_holomorphic_reductions(self):
        L = 4
        model = nets.TwoNets((nets.RBM(numHidden=3, bias=False), nets.RBM(numHidden=3, bias=False)))
        psi = NQS(model, L, 2 ** L, seed=1234)
        exactSampler = sampler.ExactSampler(psi)
        hamiltonian = 0
        for l in range(L):
            hamiltonian += -1.0 * op.SigmaZ(l) * op.SigmaZ((l + 1) % L) - 0.3 * op.SigmaX(l)
        loss_function = jVMC_exp.objective_function.Observable(hamiltonian)
        objective_fn_out = loss_function.value_and_grad(exactSampler, compute_grad=True, t=0)

        # The refinement runs over real updates. With q = i Im a diagonal shift adds a
        # real component to the purely imaginary equation, which no real update can
        # satisfy, so the shift is only used with q = Re here.
        for make_real, diagonal_shift in ((True, 1e-3), (False, 0.0)):
            with self.subTest(make_real=make_real):
                solver = RandomizedPinvSNR(
                    n_components=4, n_oversamples=2, n_iter=2, pinv_tol=0.0, pinv_cutoff=1e-6,
                    lbfgs=True, lbfgs_tol=1e-6, lbfgs_maxiter=500
                )
                opt = jVMC_exp.optimizer.TDVP(
                    exactSampler, psi, make_real=make_real, diagonalShift=diagonal_shift, solver=solver
                )
                update = opt.get_update(objective_fn_out)
                info = opt._additional_info
                self.assertFalse(jnp.iscomplexobj(update))
                self.assertGreaterEqual(info["n_lbfgs"], 1)
                self.assertLess(info["true_residual"], info["residual_rsvd"])
                self.assertLess(info["true_residual"], 1e-5)

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            RandomizedPinvSNR(n_components=2, lbfgs=True, lbfgs_maxiter=0)


class TestNonHolomorphicRandomizedPinvSNR(unittest.TestCase):
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

    def test_full_rank_matches_dense_for_both_reductions(self):
        for make_real in (True, False):
            with self.subTest(make_real=make_real):
                psi, exactSampler, loss_function = self._setup()
                n_params = psi.parameters_flat.size
                objective_fn_out = loss_function.value_and_grad(exactSampler, compute_grad=True, t=0)

                updates = []
                for solver in (
                    jVMC_exp.solver.PinvSNR(pinv_tol=0.0, pinv_cutoff=1e-8),
                    RandomizedPinvSNR(n_components=n_params, n_iter=1, pinv_tol=0.0, pinv_cutoff=1e-8),
                ):
                    opt = jVMC_exp.optimizer.TDVP(
                        exactSampler, psi, make_real=make_real, diagonalShift=1e-3, solver=solver
                    )
                    update = opt.get_update(objective_fn_out)
                    self.assertFalse(jnp.iscomplexobj(update))
                    self.assertEqual(update.shape, psi.parameters_flat.shape)
                    updates.append(np.asarray(update))

                self.assertTrue(np.allclose(updates[0], updates[1], atol=1e-6), msg=f"{updates}")

    def test_step_stays_real(self):
        psi, exactSampler, loss_function = self._setup()
        opt = jVMC_exp.optimizer.TDVP(
            exactSampler, psi, make_real=True, diagonalShift=1e-3,
            solver=RandomizedPinvSNR(n_components=8, n_oversamples=4, n_iter=2)
        )
        new_parameters, _ = opt.step(0, jVMC_exp.stepper.Euler(timeStep=1e-3), loss_function)
        self.assertEqual(new_parameters.dtype, psi.parameters_flat.dtype)


class TestGroundStateSearchRandomizedPinvSNR(unittest.TestCase):
    def test_gs_search_cpx(self):
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

        loss_function = jVMC_exp.objective_function.Observable(hamiltonian)
        solver = RandomizedPinvSNR(
            n_components=16, n_oversamples=4, n_iter=2, snr_tol=1, pinv_tol=0.0, pinv_cutoff=1e-8
        )
        stepper = jVMC_exp.stepper.Euler(5e-2)
        opt = jVMC_exp.optimizer.SR(exactSampler, psi, diagonalShift=0, diagonalScale=2, solver=solver)

        opt.ground_state_search(500, loss_function, stepper)
        eps_rel = jnp.abs((exactSampler(hamiltonian).mean.item() - exE) / exE)
        self.assertTrue(eps_rel < 1e-3)


# ----------------------------------------------------------------------------------
# Parameter-sharded layout (ShardedQGTOperator, layout="parameters")
# ----------------------------------------------------------------------------------

def _mesh_shapes():
    """(P_s, P_p) meshes to test on: pure sample, pure parameter and, if possible, mixed sharding."""
    n = MESH.size
    shapes = [(1, n), (n, 1)]
    if n % 2 == 0 and n > 2:
        shapes.append((2, n // 2))

    return list(dict.fromkeys(shapes))


def _assert_sharded(test, x, sharding):
    """``x`` is laid out as ``sharding`` (specs are compared by equivalence: JAX drops axes of size 1)."""
    test.assertTrue(x.sharding.is_equivalent_to(sharding, x.ndim), msg=f"{x.sharding} vs {sharding}")


_TRANSFORMS = {"identity": lambda x: x, "real": jnp.real, "imag": lambda x: 1j * jnp.imag(x)}


class TestShardedQGTOperator(unittest.TestCase):
    def setUp(self):
        self.N = 8 * MESH.size
        # not divisible by the number of devices (> 1): the columns are padded
        self.n = 4 * MESH.size + 3
        self.jac = _random_jacobian(jax.random.PRNGKey(0), self.N, self.n)
        self.S0 = jnp.conj(self.jac).T @ self.jac

    def test_column_sharded_copy(self):
        for shape in _mesh_shapes():
            with self.subTest(mesh_shape=shape):
                A = ShardedQGTOperator(self.jac, "real", mesh_shape=shape)
                self.assertEqual(A.mesh.devices.shape, shape)
                p = shape[1]
                n_pad = -(-self.n // p) * p
                self.assertEqual(A.padded_num_params, n_pad)
                self.assertEqual(A.jacobian.shape, (self.N, n_pad))
                _assert_sharded(self, A.jacobian, A.layout.sharding(P(SAMPLES_AXIS, PARAMS_AXIS)))
                jac = np.asarray(A.jacobian)
                self.assertTrue(np.array_equal(jac[:, :self.n], np.asarray(self.jac)))
                self.assertFalse(np.any(jac[:, self.n:]))
                # the shard of every device has num_samples / P_s rows and n_pad / P_p columns
                for shard in A.jacobian.addressable_shards:
                    self.assertEqual(shard.data.shape, (self.N // shape[0], n_pad // p))

    def test_products_match_dense(self):
        """
        Vectors and blocks, real and complex, all three reductions, with and without
        diagonal terms, with the default and with a tiny chunk budget (one sample per
        chunk, several blocks of rows), on every mesh shape.
        """
        k1, k2 = jax.random.split(jax.random.PRNGKey(3))
        Q_real = jax.random.normal(k1, (self.n, 3))
        Q_cpx = Q_real + 1j * jax.random.normal(k2, (self.n, 3))
        diag = jnp.real(jnp.diag(self.S0))
        old = jVMC_exp.global_defs.MATMUL_CHUNK_BYTES
        try:
            for budget in (old, 64):
                jVMC_exp.global_defs.update(matmul_chunk_bytes=budget)
                for shape in _mesh_shapes():
                    for mode, tr in _TRANSFORMS.items():
                        for scale, shift in ((0.0, 0.0), (0.3, 0.1)):
                            A = ShardedQGTOperator(self.jac, mode, scale, shift, mesh_shape=shape)
                            S = tr(self.S0) + scale * jnp.diag(diag) + shift * jnp.eye(self.n)
                            for Q in (Q_real, Q_cpx, Q_real[:, 0], Q_cpx[:, 0]):
                                with self.subTest(budget=budget, mesh_shape=shape, mode=mode, shift=shift,
                                                  dtype=str(Q.dtype), ndim=Q.ndim):
                                    out = A(Q)
                                    self.assertEqual(out.shape, Q.shape)
                                    self.assertTrue(jnp.allclose(out, S @ Q, atol=1e-12))
                                    self.assertTrue(jnp.allclose(A.raw(Q), self.S0 @ Q, atol=1e-12))
                                    self.assertTrue(out.sharding.is_fully_replicated)
        finally:
            jVMC_exp.global_defs.update(matmul_chunk_bytes=old)

    def test_real_reduction_stays_real(self):
        Q = jax.random.normal(jax.random.PRNGKey(4), (self.n, 3))
        A = ShardedQGTOperator(self.jac, "real")
        self.assertFalse(jnp.iscomplexobj(A(Q)))

    def test_row_sharded_block_application(self):
        """The kernel of the randomized solver: padded, row-sharded blocks in and out."""
        from jVMC_exp.solver import rsvd
        Q = jax.random.normal(jax.random.PRNGKey(6), (self.n, 4))
        old = jVMC_exp.global_defs.MATMUL_CHUNK_BYTES
        try:
            jVMC_exp.global_defs.update(matmul_chunk_bytes=64)
            for shape in _mesh_shapes():
                for mode, apply_mode, probe in (
                    ("identity", "identity", Q + 0.5j * Q[::-1]),
                    ("real", "real", Q),
                    ("imag", "imag_real", Q),
                    ("imag", "imag_cpx", Q + 0.5j * Q[::-1]),
                ):
                    with self.subTest(mesh_shape=shape, apply_mode=apply_mode):
                        A = ShardedQGTOperator(self.jac, mode, 0.0, 0.1, mesh_shape=shape)
                        layout = A.layout
                        rows = jax.device_put(jnp.pad(probe, ((0, layout.pad), (0, 0))), layout.sharding())
                        out = rsvd._apply_rows(
                            A.jacobian, rows, A._diag_rows, A._shift,
                            layout=layout, apply_mode=apply_mode, has_diag=True
                        )
                        _assert_sharded(self, out, layout.sharding())
                        M = _TRANSFORMS[mode](self.S0)
                        if apply_mode == "imag_real":
                            M = jnp.imag(self.S0)
                        expected = M @ probe + 0.1 * probe
                        out = np.asarray(out)
                        self.assertTrue(np.allclose(out[:self.n], expected, atol=1e-12))
                        self.assertFalse(np.any(out[self.n:]))
        finally:
            jVMC_exp.global_defs.update(matmul_chunk_bytes=old)

    def test_cg_with_sharded_operator(self):
        """jax.scipy.sparse.linalg.cg transposes the operator: the explicit transposes are needed."""
        A = ShardedQGTOperator(self.jac, "identity", 0.0, 1e-2)
        S = self.S0 + 1e-2 * jnp.eye(self.n)
        b = jax.random.normal(jax.random.PRNGKey(7), (self.n,)) + 0j
        x, _ = jax.scipy.sparse.linalg.cg(A, b, tol=1e-14, maxiter=200)
        self.assertTrue(jnp.allclose(S @ x, b, atol=1e-8))
        A = ShardedQGTOperator(self.jac, "real", 0.0, 1e-2)
        x, _ = jax.scipy.sparse.linalg.cg(A, jnp.real(b), tol=1e-14, maxiter=200)
        self.assertTrue(jnp.allclose((jnp.real(self.S0) + 1e-2 * jnp.eye(self.n)) @ x, jnp.real(b), atol=1e-8))
        # the CG solver of the library on the sharded operator
        x, _ = jVMC_exp.solver.CG(tol=1e-14, maxiter=200)(A, jnp.real(b))
        self.assertTrue(jnp.allclose((jnp.real(self.S0) + 1e-2 * jnp.eye(self.n)) @ x, jnp.real(b), atol=1e-8))

    def test_gradient_through_vector_product(self):
        """The L-BFGS loss differentiates through the product (transpose of linear_call)."""
        for mode in ("identity", "real", "imag"):
            with self.subTest(mode=mode):
                A = ShardedQGTOperator(self.jac, mode, 0.0, 0.0)
                B = QGTOperator(self.jac, mode, 0.0, 0.0)
                b = jax.random.normal(jax.random.PRNGKey(8), (self.n,))
                x = jax.random.normal(jax.random.PRNGKey(9), (self.n,))
                grad_A = jax.grad(lambda x: jnp.linalg.norm(A(x) - b))(x)
                grad_B = jax.grad(lambda x: jnp.linalg.norm(B(x) - b))(x)
                self.assertTrue(jnp.allclose(grad_A, grad_B, atol=1e-10))

    def test_rejects_bad_mesh_shapes(self):
        for shape in ((MESH.size + 1, 1), (0, MESH.size), (1, 1, MESH.size)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                ShardedQGTOperator(self.jac, "real", mesh_shape=shape)
        with self.assertRaises(ValueError):
            ShardedQGTOperator(self.jac, "conj")


class TestCholeskyQR(unittest.TestCase):
    def _rows(self, Y, layout):
        return jax.device_put(jnp.pad(Y, ((0, layout.pad), (0, 0))), layout.sharding())

    def _check(self, Y, layout, tol):
        from jVMC_exp.solver import rsvd
        n, r = Y.shape
        Q = rsvd._cholesky_qr(self._rows(Y, layout), layout)
        _assert_sharded(self, Q, layout.sharding())
        Q = np.asarray(Q)
        self.assertFalse(np.any(Q[n:]))
        Q = Q[:n]
        Y = np.asarray(Y)
        self.assertLess(np.linalg.norm(np.conj(Q).T @ Q - np.eye(r)), tol)
        # same column space: Y = Q (Q^H Y)
        self.assertLess(np.linalg.norm(Q @ (np.conj(Q).T @ Y) - Y), tol * np.linalg.norm(Y))

    def test_ill_conditioned_block(self):
        """Singular values spanning 1 ... 1e-8: CholeskyQR2 alone would lose orthogonality."""
        n, r = 4 * MESH.size + 37, 12
        U, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(1), (n, r)))
        W, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(2), (r, r)))
        Y = (U * jnp.logspace(0, -8, r)) @ W.T
        Yc = Y * jnp.exp(1j * jax.random.uniform(jax.random.PRNGKey(3), (n, 1), maxval=2 * np.pi))
        for shape in _mesh_shapes():
            layout = ParameterLayout(parameter_mesh(shape), n)
            for Y_ in (Y, Yc):
                with self.subTest(mesh_shape=shape, dtype=str(Y_.dtype)):
                    self._check(Y_, layout, tol=1e-12)

    def test_rank_deficient_block(self):
        """
        A numerically rank deficient block (rank 6 of 9 columns) needs more than one
        shifted pass; the result is still an orthonormal basis containing range(Y).
        """
        n = 4 * MESH.size + 13
        U, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(4), (n, 6)))
        Y = U @ jax.random.normal(jax.random.PRNGKey(5), (6, 9))
        layout = ParameterLayout(parameter_mesh(None), n)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._check(Y, layout, tol=1e-12)


class TestRandomizedEighParameterLayout(unittest.TestCase):
    def setUp(self):
        self.N = 16 * MESH.size
        self.n = 4 * MESH.size + 9
        self.jac = _random_jacobian(jax.random.PRNGKey(10), self.N, self.n, decay=0.6)
        self.S0 = jnp.conj(self.jac).T @ self.jac

    def _check(self, A, S, k, n_oversamples, n_iter, which="LM", tol=1e-8, mode="device"):
        ev, V = randomized_eigh(
            A, k, jax.random.PRNGKey(5), n_oversamples=n_oversamples, n_iter=n_iter,
            which=which, diagonalization_mode=mode
        )
        ev_ref, V_ref = jnp.linalg.eigh(S)
        idx = jnp.argsort(jnp.abs(ev_ref) if which == "LM" else ev_ref)[::-1][:k]
        idx = idx[jnp.argsort(ev_ref[idx])]

        self.assertEqual(ev.shape, (k,))
        self.assertEqual(V.shape, (A.padded_num_params, k))
        _assert_sharded(self, V, A.layout.sharding())
        V = np.asarray(V)
        self.assertFalse(np.any(V[self.n:]))
        V = jnp.asarray(V[:self.n])
        self.assertTrue(bool(jnp.all(jnp.diff(ev) >= 0)))
        self.assertTrue(jnp.allclose(ev, ev_ref[idx], atol=tol, rtol=tol), msg=f"{ev} vs {ev_ref[idx]}")
        self.assertLess(_projector_distance(V, V_ref[:, idx]), 1e2 * tol)
        self.assertTrue(jnp.allclose(A(V), V * ev, atol=1e2 * tol))

    def test_full_rank_is_exact(self):
        for shape in _mesh_shapes():
            for mode, tr in _TRANSFORMS.items():
                with self.subTest(mesh_shape=shape, mode=mode):
                    A = ShardedQGTOperator(self.jac, mode, 0.0, 0.0, mesh_shape=shape)
                    self._check(A, tr(self.S0), k=self.n, n_oversamples=0, n_iter=1, tol=1e-9)

    def test_low_rank_captures_dominant_eigenpairs(self):
        for shape in _mesh_shapes():
            with self.subTest(mesh_shape=shape):
                A = ShardedQGTOperator(self.jac, "identity", 0.0, 0.0, mesh_shape=shape)
                self._check(A, self.S0, k=4, n_oversamples=4, n_iter=4, tol=1e-6)
                A = ShardedQGTOperator(self.jac, "real", 0.0, 0.0, mesh_shape=shape)
                self._check(A, jnp.real(self.S0), k=4, n_oversamples=4, n_iter=4, tol=1e-6)
                self._check(A, jnp.real(self.S0), k=4, n_oversamples=4, n_iter=4, which="LA", tol=1e-6)
                A = ShardedQGTOperator(self.jac, "imag", 0.0, 0.0, mesh_shape=shape)
                self._check(A, 1j * jnp.imag(self.S0), k=6, n_oversamples=4, n_iter=4, tol=1e-6)

    def test_diagonal_terms(self):
        S = jnp.real(self.S0)
        for shape in _mesh_shapes():
            with self.subTest(mesh_shape=shape):
                A = ShardedQGTOperator(self.jac, "real", 0.2, 0.01, mesh_shape=shape)
                M = S + 0.2 * jnp.diag(jnp.diag(S)) + 0.01 * jnp.eye(self.n)
                self._check(A, M, k=self.n, n_oversamples=0, n_iter=1, tol=1e-9)
                # i Im with a shift: complex probe (imag_cpx)
                A = ShardedQGTOperator(self.jac, "imag", 0.0, 0.05, mesh_shape=shape)
                M = 1j * jnp.imag(self.S0) + 0.05 * jnp.eye(self.n)
                self._check(A, M, k=self.n, n_oversamples=0, n_iter=1, tol=1e-9)

    def test_distributed_mode_pads_the_small_matrix(self):
        A = ShardedQGTOperator(self.jac, "identity", 0.0, 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._check(A, self.S0, k=5, n_oversamples=2, n_iter=3, tol=1e-6, mode="distributed")

    def test_matches_sample_layout(self):
        """Same probes up to the layout, same subspace: the eigenvalues agree with the replicated layout."""
        ev_s, _ = randomized_eigh(QGTOperator(self.jac, "real"), 5, jax.random.PRNGKey(1), n_oversamples=3, n_iter=3)
        ev_p, _ = randomized_eigh(ShardedQGTOperator(self.jac, "real"), 5, jax.random.PRNGKey(1), n_oversamples=3, n_iter=3)
        self.assertTrue(jnp.allclose(ev_s, ev_p, rtol=1e-6))


class TestRandomizedPinvSNRParameterLayout(unittest.TestCase):
    """RandomizedPinvSNR(layout="parameters") on the reference TDVP problems."""

    @staticmethod
    def _non_holomorphic(bias):
        L = 4
        model = nets.TwoNets((nets.RBM(numHidden=3, bias=bias), nets.RBM(numHidden=3, bias=bias)))
        psi = NQS(model, L, 2 ** L, seed=1234)
        hamiltonian = 0
        for l in range(L):
            hamiltonian += -1.0 * op.SigmaZ(l) * op.SigmaZ((l + 1) % L) - 0.3 * op.SigmaX(l)

        return psi, sampler.ExactSampler(psi), jVMC_exp.objective_function.Observable(hamiltonian)

    def test_arguments(self):
        self.assertEqual(RandomizedPinvSNR(n_components=4).layout, "samples")
        solver = RandomizedPinvSNR(n_components=4, layout="parameters")
        self.assertEqual(solver.mesh_shape, (1, MESH.size))
        self.assertEqual(RandomizedPinvSNR(n_components=4, layout="parameters", mesh_shape=(MESH.size, 1)).mesh_shape, (MESH.size, 1))
        with self.assertRaises(ValueError):
            RandomizedPinvSNR(n_components=4, layout="rows")
        with self.assertRaises(ValueError):
            RandomizedPinvSNR(n_components=4, layout="parameters", mesh_shape=(MESH.size + 1, 1))
        with self.assertRaises(ValueError):
            RandomizedPinvSNR(n_components=4, mesh_shape=(1, MESH.size))

    def test_full_rank_matches_pinv_snr_holomorphic(self):
        """Step for step equal to the dense PinvSNR, as for the sample layout."""
        def run(solver):
            psi, exactSampler, hamiltonian, _ = _reference_problem()
            loss_function = jVMC_exp.objective_function.Observable(hamiltonian)
            opt = jVMC_exp.optimizer.TDVP(exactSampler, psi, make_real=False, diagonalShift=0, solver=solver)
            out = opt.time_evolution(1e-2, loss_function, jVMC_exp.stepper.Euler(timeStep=1e-3))
            return np.array(out['energy']['mean'])

        n_params = _reference_problem()[0].numParameters
        energy_dense = run(jVMC_exp.solver.PinvSNR(pinv_tol=0.0, pinv_cutoff=1e-10))
        for shape in _mesh_shapes():
            with self.subTest(mesh_shape=shape):
                energy = run(RandomizedPinvSNR(
                    n_components=n_params, n_iter=1, pinv_tol=0.0, pinv_cutoff=1e-10,
                    layout="parameters", mesh_shape=shape
                ))
                self.assertTrue(np.allclose(energy_dense, energy, atol=1e-8))

    def test_low_rank_solve_is_close_to_dense(self):
        """As for the sample layout: a rank that reaches the pinv cutoff reproduces the dense solve."""
        psi, exactSampler, hamiltonian, _ = _reference_problem()
        loss_function = jVMC_exp.objective_function.Observable(hamiltonian)
        objective_fn_out = loss_function.value_and_grad(exactSampler, compute_grad=True, t=0)

        def update_with(solver):
            opt = jVMC_exp.optimizer.TDVP(exactSampler, psi, make_real=False, diagonalShift=0, solver=solver)
            return np.asarray(opt.get_update(objective_fn_out)), opt._additional_info

        dense, info_d = update_with(jVMC_exp.solver.PinvSNR(pinv_tol=0.0, pinv_cutoff=1e-6))
        spectrum = np.asarray(info_d["spectrum"])
        n_above = int(np.sum(np.abs(spectrum / spectrum[-1]) > 1e-8))
        self.assertLess(n_above, spectrum.size)
        for shape in _mesh_shapes():
            with self.subTest(mesh_shape=shape):
                update, info = update_with(RandomizedPinvSNR(
                    n_components=n_above, n_oversamples=2, n_iter=4, pinv_tol=0.0, pinv_cutoff=1e-6,
                    layout="parameters", mesh_shape=shape
                ))
                self.assertEqual(update.shape, dense.shape)
                self.assertTrue(np.allclose(update, dense, atol=1e-5), msg=f"{update}\n{dense}")
                self.assertTrue(np.allclose(info["spectrum"], spectrum[-n_above:], rtol=1e-6))
                self.assertLess(info["true_residual"], 1e-5)

    def test_full_rank_matches_dense_non_holomorphic(self):
        """
        Both reductions; with biases the 30 parameters are not divisible by 4 devices,
        so the parameter padding is exercised inside the optimizer as well.
        """
        for bias in (False, True):
            psi, exactSampler, loss_function = self._non_holomorphic(bias)
            n_params = psi.parameters_flat.size
            objective_fn_out = loss_function.value_and_grad(exactSampler, compute_grad=True, t=0)
            for make_real in (True, False):
                def update_with(solver):
                    opt = jVMC_exp.optimizer.TDVP(
                        exactSampler, psi, make_real=make_real, diagonalShift=1e-3, solver=solver
                    )
                    return np.asarray(opt.get_update(objective_fn_out))

                dense = update_with(jVMC_exp.solver.PinvSNR(pinv_tol=0.0, pinv_cutoff=1e-8))
                for shape in _mesh_shapes():
                    with self.subTest(bias=bias, make_real=make_real, mesh_shape=shape):
                        update = update_with(RandomizedPinvSNR(
                            n_components=n_params, n_iter=1, pinv_tol=0.0, pinv_cutoff=1e-8,
                            layout="parameters", mesh_shape=shape
                        ))
                        self.assertFalse(np.iscomplexobj(update))
                        self.assertEqual(update.shape, psi.parameters_flat.shape)
                        self.assertTrue(np.allclose(dense, update, atol=1e-6))

    def test_s_norm_and_tdvp_error_unchanged(self):
        """
        The optimizer keeps the sample-sharded operator for opt.qgt, and the sharded
        operator's vector product gives the same S-norm and TDVP error.
        """
        from jVMC_exp.util.util import s_norm_fn
        psi, exactSampler, loss_function = self._non_holomorphic(True)
        objective_fn_out = loss_function.value_and_grad(exactSampler, compute_grad=True, t=0)
        results = {}
        for layout in ("samples", "parameters"):
            # full rank: both layouts give the exact update
            opt = jVMC_exp.optimizer.TDVP(
                exactSampler, psi, make_real=True, diagonalShift=1e-3, store_qgt=True, compute_tdvp_error=True,
                solver=RandomizedPinvSNR(n_components=psi.parameters_flat.size, n_iter=1, pinv_tol=0.0, layout=layout)
            )
            update = opt.get_update(objective_fn_out)
            norm = s_norm_fn(opt)
            results[layout] = (np.asarray(update), opt._additional_info["tdvp_error"], float(norm(update)))
            if layout == "parameters":
                # the same quantities from the parameter-sharded operator
                lhs = opt._get_lhs(objective_fn_out.grad_log_psi)
                opt._qgt = ShardedQGTOperator.from_operator(lhs).raw
                results["sharded qgt"] = (
                    results[layout][0],
                    opt._get_tdvp_error(update, objective_fn_out.grad, objective_fn_out.o_loc.var).item(),
                    float(norm(update)),
                )
        ref = results["samples"]
        for name in ("parameters", "sharded qgt"):
            with self.subTest(name=name):
                update, tdvp_error, s_norm = results[name]
                self.assertTrue(np.allclose(update, ref[0], atol=1e-6))
                self.assertAlmostEqual(tdvp_error, ref[1], places=6)
                self.assertAlmostEqual(s_norm, ref[2], places=6)

    def test_lbfgs_refinement(self):
        psi, exactSampler, hamiltonian, _ = _reference_problem()
        loss_function = jVMC_exp.objective_function.Observable(hamiltonian)
        objective_fn_out = loss_function.value_and_grad(exactSampler, compute_grad=True, t=0)
        dense = jVMC_exp.optimizer.TDVP(
            exactSampler, psi, make_real=False, diagonalShift=0,
            solver=jVMC_exp.solver.PinvSNR(pinv_tol=0.0, pinv_cutoff=1e-6)
        ).get_update(objective_fn_out)
        opt = jVMC_exp.optimizer.TDVP(
            exactSampler, psi, make_real=False, diagonalShift=0,
            solver=RandomizedPinvSNR(
                n_components=2, n_oversamples=1, n_iter=2, pinv_tol=0.0, pinv_cutoff=1e-6,
                lbfgs=True, lbfgs_tol=1e-6, lbfgs_maxiter=500, layout="parameters"
            )
        )
        refined = opt.get_update(objective_fn_out)
        info = opt._additional_info
        self.assertGreater(info["residual_rsvd"], 1e-3)
        self.assertGreaterEqual(info["n_lbfgs"], 1)
        self.assertLess(info["true_residual"], 1e-5)
        self.assertLess(np.linalg.norm(np.asarray(refined) - np.asarray(dense)), 1e-3 * np.linalg.norm(np.asarray(dense)))

    def test_snr_modes(self):
        psi, _, hamiltonian, _ = _reference_problem(batch_size=2 ** 10)
        mc_sampler = jVMC_exp.sampler.MCSampler(psi, jVMC_exp.propose.SpinFlip(), 123, 2 ** 6, 2 ** 10, mu=1)
        loss_function = jVMC_exp.objective_function.Observable(hamiltonian)
        stepper = jVMC_exp.stepper.Euler(timeStep=1e-3)
        solver = RandomizedPinvSNR(n_components=6, n_oversamples=2, n_iter=2, snr_tol=1, snr_mode="gaussian", layout="parameters")
        opt = jVMC_exp.optimizer.TDVP(mc_sampler, psi, make_real=False, solver=solver)
        opt.time_evolution(2e-3, loss_function, stepper)
        self.assertEqual(np.asarray(opt.meta_data["snr"]).shape, (6,))
        self.assertEqual(np.asarray(opt.meta_data["spectrum"]).shape, (6,))

        solver = RandomizedPinvSNR(n_components=6, snr_tol=1, snr_mode="empiric", layout="parameters")
        opt = jVMC_exp.optimizer.TDVP(mc_sampler, psi, make_real=False, solver=solver)
        with self.assertRaises(NotImplementedError):
            opt.time_evolution(1e-3, loss_function, stepper)


if __name__ == "__main__":
    unittest.main()
