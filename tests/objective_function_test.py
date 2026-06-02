import unittest
import jax.numpy as jnp
import numpy as np

import jVMC_exp
import jVMC_exp.nets
import jVMC_exp.vqs
import jVMC_exp.sampler
import jVMC_exp.operator.discrete as op
from jVMC_exp.stats import SampledObs
from jVMC_exp.objective_function import (
    ObjectiveFunctionOutput,
    Observable,
    Estimator,
    ParametricObservable,
)

L = 4 
N_SAMPLES = 16

def build_sampler():
    model = jVMC_exp.nets.CpxRBM(1, True)
    psi = jVMC_exp.vqs.NQS(model, L, batchSize=N_SAMPLES)
    sampler = jVMC_exp.sampler.ExactSampler(psi, 2)
    return sampler, psi

def build_hamiltonian():
    J, h = -1, 1
    H = 0
    for i in range(L):
        H += J * op.SigmaZ(i) * op.SigmaZ((i + 1) % L) + h * op.SigmaX(i)
    return H

def dummy_estimator_fn(parameters, samples):
    """Differentiable estimator: returns mean of spin values along last axis."""
    return jnp.mean(samples.astype(jnp.float32), axis=-1)

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
        self.assertTrue(np.all(np.isfinite(np.array(out.grad.observations))))

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
        self.assertTrue(np.all(np.isfinite(np.array(out.grad.observations))))

    def test_grad_log_psi_shape(self):
        out = self.obj.value_and_grad(self.sampler)
        self.assertEqual(out.grad_log_psi.observations.shape[0], N_SAMPLES)

    def test_grad_differs_from_observable_only(self):
        # ParametricObservable grad = covar_term + estimator_grad, so it must
        # differ from a plain Observable grad (the estimator contribution is non-zero)
        plain_out = Observable(self.sigma_z_0).value_and_grad(self.sampler)
        param_out = self.obj.value_and_grad(self.sampler)
        differs = not np.allclose(
            np.array(param_out.grad.observations),
            np.array(plain_out.grad.observations.squeeze()),
            atol=1e-8,
        )
        self.assertTrue(differs, "ParametricObservable grad should include estimator contribution")

if __name__ == "__main__":
    unittest.main()