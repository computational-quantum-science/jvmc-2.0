import unittest
import jax
import jax.numpy as jnp
import numpy as np

import jVMC_exp
import jVMC_exp.nets
import jVMC_exp.vqs
import jVMC_exp.sampler
import jVMC_exp.operator.discrete as op
from jVMC_exp.stats import SampledObs, LazySampledObs
from jVMC_exp.objective_function import (
    ObjectiveFunctionOutput,
    Observable,
    Estimator,
    ParametricObservable,
)

L = 4 
N_SAMPLES = 16

def build_sampler(batch_size=N_SAMPLES):
    model = jVMC_exp.nets.CpxRBM(1, True)
    psi = jVMC_exp.vqs.NQS(model, L, batchSize=batch_size)
    sampler = jVMC_exp.sampler.ExactSampler(psi, 2)
    return sampler, psi

def build_hamiltonian():
    J, h = -1, 1
    H = 0
    for i in range(L):
        H += J * op.SigmaZ(i) * op.SigmaZ((i + 1) % L) + h * op.SigmaX(i)
    return H

def dummy_estimator_fn(parameters, samples):
    """
    Differentiable estimator: depends on both the spin configuration and
    the network parameters
    """
    param_sum = sum(jnp.sum(p) for p in jax.tree_util.tree_leaves(parameters))
    return jnp.mean(samples.astype(jnp.float32), axis=-1) + param_sum

class TestEstimator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sampler, cls.psi = build_sampler()
        cls.obj = Estimator(dummy_estimator_fn)

    def test_call_returns_sampled_obs(self):
        result = self.obj(self.sampler)
        self.assertIsInstance(result, SampledObs)

    def test_call_observations_shape(self):
        result = self.obj(self.sampler)
        self.assertEqual(result.observations.shape[0], N_SAMPLES)

    def test_value_and_grad_output_type(self):
        out = self.obj.value_and_grad(self.sampler)
        self.assertIsInstance(out, ObjectiveFunctionOutput)

    def test_value_and_grad_o_loc_present(self):
        out = self.obj.value_and_grad(self.sampler)
        self.assertIsNotNone(out.o_loc)

    def test_value_and_grad_grad_present(self):
        out = self.obj.value_and_grad(self.sampler)
        self.assertIsNotNone(out.grad)

    def test_value_and_grad_no_grad_log_psi(self):
        out = self.obj.value_and_grad(self.sampler)
        self.assertIsNone(out.grad_log_psi)

    def test_grad_values_are_finite(self):
        out = self.obj.value_and_grad(self.sampler)
        self.assertTrue(np.all(np.isfinite(np.array(out.grad))))

class TestParametricObservable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sampler, cls.psi = build_sampler()
        cls.sigma_z_0 = op.SigmaZ(0)
        cls.obj = ParametricObservable(cls.sigma_z_0, dummy_estimator_fn)

    def test_call_returns_sampled_obs(self):
        result = self.obj(self.sampler)
        self.assertIsInstance(result, SampledObs)

    def test_call_observations_shape(self):
        result = self.obj(self.sampler)
        self.assertEqual(result.observations.shape[0], N_SAMPLES)

    def test_value_and_grad_output_type(self):
        out = self.obj.value_and_grad(self.sampler)
        self.assertIsInstance(out, ObjectiveFunctionOutput)

    def test_grad_values_are_finite(self):
        out = self.obj.value_and_grad(self.sampler)
        self.assertTrue(np.all(np.isfinite(np.array(out.grad))))

    def test_grad_log_psi_shape(self):
        out = self.obj.value_and_grad(self.sampler)
        self.assertEqual(out.grad_log_psi.observations.shape[0], N_SAMPLES)

    def test_grad_differs_from_observable_only(self):
        # ParametricObservable grad = covar_term + estimator_grad, so it must
        # differ from a plain Observable grad (the estimator contribution is non-zero)
        plain_out = Observable(self.sigma_z_0).value_and_grad(self.sampler)
        param_out = self.obj.value_and_grad(self.sampler)
        differs = not np.allclose(np.array(param_out.grad), np.array(plain_out.grad))
        
        self.assertTrue(differs, "ParametricObservable grad should include estimator contribution")

class TestObservableBatchedJacobian(unittest.TestCase):
    """
    batched_jacobian=True routes grad_log_psi through LazySampledObs
    (batches of 6 over N_SAMPLES=16 -> uneven [6, 6, 4]) instead of a
    materialized Jacobian. The physics must be identical either way.
    """
    @classmethod
    def setUpClass(cls):
        cls.sampler, cls.psi = build_sampler(batch_size=6)
        cls.hamiltonian = build_hamiltonian()

    def test_grad_log_psi_is_lazy(self):
        out = Observable(self.hamiltonian, batched_jacobian=True).value_and_grad(self.sampler)
        self.assertIsInstance(out.grad_log_psi, LazySampledObs)

    def test_matches_dense_observable(self):
        dense_out = Observable(self.hamiltonian, batched_jacobian=False).value_and_grad(self.sampler)
        lazy_out = Observable(self.hamiltonian, batched_jacobian=True).value_and_grad(self.sampler)

        self.assertTrue(np.allclose(np.array(lazy_out.grad), np.array(dense_out.grad), atol=1e-10))
        self.assertTrue(np.allclose(lazy_out.o_loc.mean, dense_out.o_loc.mean))

class TestEstimatorBatchedJacobian(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sampler, cls.psi = build_sampler(batch_size=6)

    def test_matches_dense_estimator(self):
        dense_out = Estimator(dummy_estimator_fn, batched_jacobian=False).value_and_grad(self.sampler)
        lazy_out = Estimator(dummy_estimator_fn, batched_jacobian=True).value_and_grad(self.sampler)

        self.assertTrue(np.allclose(np.array(lazy_out.grad), np.array(dense_out.grad), atol=1e-10))

class TestParametricObservableBatchedJacobian(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sampler, cls.psi = build_sampler(batch_size=6)
        cls.sigma_z_0 = op.SigmaZ(0)

    def test_grad_log_psi_is_lazy(self):
        obj = ParametricObservable(self.sigma_z_0, dummy_estimator_fn, batched_jacobian=True)
        out = obj.value_and_grad(self.sampler)
        self.assertIsInstance(out.grad_log_psi, LazySampledObs)

    def test_matches_dense_parametric_observable(self):
        dense_out = ParametricObservable(
            self.sigma_z_0, dummy_estimator_fn, batched_jacobian=False
        ).value_and_grad(self.sampler)
        lazy_out = ParametricObservable(
            self.sigma_z_0, dummy_estimator_fn, batched_jacobian=True
        ).value_and_grad(self.sampler)

        self.assertTrue(np.allclose(np.array(lazy_out.grad), np.array(dense_out.grad), atol=1e-10))

if __name__ == "__main__":
    unittest.main()