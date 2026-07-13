import unittest
import jax
import jax.numpy as jnp

import jVMC_exp.nets as nets
from jVMC_exp.stats import BatchedJacobian, SampledObs
from jVMC_exp.vqs import NQS
import jVMC_exp.sampler as sampler


class TestStats(unittest.TestCase):
    def _make_batched_jacobian_reference(self):
        L = 3
        psi = NQS(nets.CpxRBM(numHidden=2, bias=True), L, 4, seed=1234)
        exact_sampler = sampler.ExactSampler(psi)
        exact_sampler.sample()
        dense = SampledObs(psi.gradients(exact_sampler.samples), exact_sampler.weights)
        batched = BatchedJacobian(psi, exact_sampler.samples, exact_sampler.weights, batch_size=4)

        return dense, batched

    def test_batched_jacobian_materialize_matches_dense(self):
        dense, batched = self._make_batched_jacobian_reference()

        self.assertTrue(jnp.allclose(batched.materialize().observations, dense.observations))
        self.assertTrue(jnp.allclose(batched.materialize().weights, dense.weights))

    def test_batched_jacobian_normalized_blocks_match_dense(self):
        dense, batched = self._make_batched_jacobian_reference()
        normalized = jnp.concatenate(tuple(batched.iter_normalized_blocks()), axis=0)

        self.assertTrue(jnp.allclose(normalized, dense._normalized_obs))

    def test_batched_jacobian_linear_operations_match_dense(self):
        dense, batched = self._make_batched_jacobian_reference()
        normalized = dense._normalized_obs
        v = jnp.arange(normalized.shape[1]).astype(normalized.dtype)
        y = jnp.arange(normalized.shape[0]).astype(normalized.dtype)

        self.assertTrue(jnp.allclose(batched.get_covar(), dense.get_covar()))
        self.assertTrue(jnp.allclose(batched.tangent_kernel(), dense.tangent_kernel))
        self.assertTrue(jnp.allclose(batched.matvec(v), dense.get_covar() @ v))
        self.assertTrue(jnp.allclose(
            batched.conj_transpose_matvec(y),
            jnp.conj(jnp.transpose(normalized)) @ y,
        ))

    def test_batched_jacobian_subset_matches_dense(self):
        dense, batched = self._make_batched_jacobian_reference()
        dense_subset = dense.get_subset(0, 4)
        batched_subset = batched.get_subset(0, 4)

        self.assertTrue(jnp.allclose(batched_subset.materialize().get_covar(), dense_subset.get_covar()))

    def test_real_imag_doubled_batched_jacobian_matches_dense_blocks(self):
        dense, batched = self._make_batched_jacobian_reference()
        doubled = batched.real_imag_doubled()
        expected_blocks = []
        for idx in range(batched.num_blocks):
            sl = batched._slice(idx)
            block = dense._normalized_obs[sl]
            expected_blocks.append(jnp.concatenate([jnp.real(block), jnp.imag(block)], axis=0))
        expected = jnp.concatenate(tuple(expected_blocks), axis=0)

        self.assertTrue(jnp.allclose(doubled._normalized_obs, expected))
        self.assertTrue(jnp.allclose(doubled.tangent_kernel(), expected @ jnp.transpose(expected)))
        
    def test_subset_function(self):

        N = 10
        Obs1 = jnp.reshape(jnp.arange(N), (N, 1))
        p = jax.random.uniform(jax.random.PRNGKey(123), (N,))
        p = p / jnp.sum(p)

        obs1 = SampledObs(Obs1, p)
        obs2 = obs1.get_subset(0, N // 2)

        self.assertTrue(jnp.allclose(obs1.mean[0], jnp.sum(jnp.reshape(Obs1, (N,)) * p)))

        self.assertTrue( 
            jnp.allclose(
                obs2.mean, 
                jnp.sum(jnp.reshape(Obs1, (N,))[0:N//2] * p[0:N//2]) / jnp.sum(p[0:N//2])
            )
        )

        obs3 = SampledObs(Obs1[0:N//2, :], p[0:N//2] / jnp.sum(p[0:N//2]))

        self.assertTrue(jnp.allclose(obs3.get_covar(), obs2.get_covar()))

    def _make_converged_obs(self, n_chains, chain_length, seed=0):
        """Independent draws from the same Gaussian — should give R-hat ≈ 1."""
        key = jax.random.PRNGKey(seed)
        samples = jax.random.normal(key, shape=(n_chains * chain_length,))
        return SampledObs(samples[:, None])

    def _make_diverged_obs(self, n_chains, chain_length, seed=0):
        """Chains with clearly different means — R-hat should be >> 1."""
        key = jax.random.PRNGKey(seed)
        noise = jax.random.normal(key, shape=(n_chains, chain_length)) * 0.01
        means = jnp.arange(n_chains, dtype=float)[:, None]
        samples = (noise + means).ravel()
        return SampledObs(samples[:, None])

    def _make_autocorrelated_obs(self, n_chains, chain_length, rho=0.9, seed=0):
        """AR(1) process with known autocorrelation — tau ≈ (1+rho)/(1-rho)."""
        key = jax.random.PRNGKey(seed)
        noise = jax.random.normal(key, shape=(n_chains, chain_length))
        samples = jnp.zeros((n_chains, chain_length))
        for t in range(1, chain_length):
            samples = samples.at[:, t].set(rho * samples[:, t-1] + jnp.sqrt(1 - rho**2) * noise[:, t])
        return SampledObs(samples.ravel()[:, None]), rho

    def test_rhat_converged_close_to_one(self):
        """R-hat from i.i.d. samples should be close to 1."""
        obs = self._make_converged_obs(n_chains=4, chain_length=256)
        r_hat = obs.get_R_hat(n_chains=4)
        self.assertLess(float(r_hat), 1.1)

    def test_rhat_diverged_greater_than_one(self):
        """R-hat from chains with different means should be well above 1."""
        obs = self._make_diverged_obs(n_chains=4, chain_length=256)
        r_hat = obs.get_R_hat(n_chains=4)
        self.assertGreater(float(r_hat), 1.1)

    def test_rhat_raises_for_multi_obs(self):
        """get_R_hat should raise NotImplementedError when num_obs > 1."""
        samples = jnp.ones((100, 3))
        obs = SampledObs(samples)
        with self.assertRaises(NotImplementedError):
            obs.get_R_hat(n_chains=4)

    def test_rhat_is_scalar(self):
        """R-hat should return a scalar."""
        obs = self._make_converged_obs(n_chains=4, chain_length=256)
        r_hat = obs.get_R_hat(n_chains=4)
        self.assertEqual(jnp.array(r_hat).ndim, 0)

    def test_rhat_num_samples_not_divisible(self):
        """get_R_hat should raise when num_samples is not divisible by n_chains."""
        key = jax.random.PRNGKey(0)
        samples = jax.random.normal(key, shape=(101, 1))
        obs = SampledObs(samples)
        with self.assertRaises((TypeError, ValueError)):
            obs.get_R_hat(n_chains=4)

    def test_act_iid_samples_close_to_one(self):
        """Autocorrelation time of i.i.d. samples should be close to 1."""
        obs = self._make_converged_obs(n_chains=4, chain_length=512)
        tau = obs.get_autocorrelation_time(n_chains=4)
        self.assertLess(float(jnp.mean(tau.mean)), 5.0)

    def test_act_correlated_larger_than_iid(self):
        """Autocorrelation time of AR(1) samples should exceed that of i.i.d."""
        obs_iid = self._make_converged_obs(n_chains=4, chain_length=512)
        obs_ar1, _ = self._make_autocorrelated_obs(n_chains=4, chain_length=512, rho=0.9)

        tau_iid = float(jnp.mean(obs_iid.get_autocorrelation_time(n_chains=4).mean))
        tau_ar1 = float(jnp.mean(obs_ar1.get_autocorrelation_time(n_chains=4).mean))

        self.assertGreater(tau_ar1, tau_iid)

    def test_act_raises_for_multi_obs(self):
        """get_autocorrelation_time should raise NotImplementedError when num_obs > 1."""
        samples = jnp.ones((100, 3))
        obs = SampledObs(samples)
        with self.assertRaises(NotImplementedError):
            obs.get_autocorrelation_time(n_chains=4)

    def test_act_returns_sampled_obs(self):
        """get_autocorrelation_time should return a SampledObs instance."""
        obs = self._make_converged_obs(n_chains=4, chain_length=256)
        tau = obs.get_autocorrelation_time(n_chains=4)
        self.assertIsInstance(tau, SampledObs)

    def test_act_is_finite(self):
        """Autocorrelation time should be finite for well-behaved input."""
        obs = self._make_converged_obs(n_chains=4, chain_length=512)
        tau = obs.get_autocorrelation_time(n_chains=4)
        self.assertTrue(jnp.all(jnp.isfinite(tau.mean)))

    def test_act_higher_rho_gives_larger_tau(self):
        """Higher AR(1) correlation coefficient should yield larger autocorrelation time."""
        obs_lo, _ = self._make_autocorrelated_obs(n_chains=4, chain_length=1024, rho=0.5)
        obs_hi, _ = self._make_autocorrelated_obs(n_chains=4, chain_length=1024, rho=0.95)

        tau_lo = float(jnp.mean(obs_lo.get_autocorrelation_time(n_chains=4).mean))
        tau_hi = float(jnp.mean(obs_hi.get_autocorrelation_time(n_chains=4).mean))

        self.assertGreater(tau_hi, tau_lo)

if __name__ == "__main__":
    unittest.main()
