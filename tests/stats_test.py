import unittest
import jax
import jax.numpy as jnp

from jVMC_exp.stats import SampledObs, LazySampledObs, _reshape_in_batches
from jVMC_exp.sharding_config import SizedIterable

class TestStats(unittest.TestCase):
        
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

    def test_transform_updates_mean(self):
        N = 20
        key = jax.random.PRNGKey(0)
        data = jax.random.normal(key, shape=(N, 1))
        obs = SampledObs(data)

        obs.transform(lambda x: 2 * x)

        expected = SampledObs(2 * data)
        self.assertTrue(jnp.allclose(obs.mean, expected.mean))
        self.assertTrue(jnp.allclose(obs.var, expected.var))

    def test_state_roundtrip_recovers_observations(self):
        """
        `SampledObs` keeps a single buffer internally and destructively
        converts it between "observations" (o), "centered" (c) and
        "normalized" (n) representations on access. Cycling through every
        transition (o->c, c->n, n->o, o->n, n->c, c->o) must reproduce the
        exact formula at each step and lose no information: the buffer
        must be back to the original input once the cycle returns to "o".
        """
        N = 12
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(7), 3)
        data = jax.random.normal(k1, (N, 3)) + 1j * jax.random.normal(k2, (N, 3))
        weights = jax.random.uniform(k3, (N,))
        weights = weights / jnp.sum(weights)

        obs = SampledObs(data, weights)

        mean_ref = jnp.tensordot(weights, data, axes=(0, 0))
        centered_ref = data - mean_ref
        normalized_ref = jnp.einsum("i,i...->i...", jnp.sqrt(weights), centered_ref)

        self.assertTrue(jnp.allclose(obs._centered_obs, centered_ref))      # o -> c
        self.assertTrue(jnp.allclose(obs._normalized_obs, normalized_ref))  # c -> n
        self.assertTrue(jnp.allclose(obs.observations, data))               # n -> o
        self.assertTrue(jnp.allclose(obs._normalized_obs, normalized_ref))  # o -> n
        self.assertTrue(jnp.allclose(obs._centered_obs, centered_ref))      # n -> c
        self.assertTrue(jnp.allclose(obs.observations, data))               # c -> o

    def test_state_roundtrip_zero_weight_rows(self):
        """
        A row with weight exactly 0 (e.g. a padded row, or a genuinely
        zero-weighted sample from a CutoffSampler) is destroyed the moment
        it passes through the normalized state (multiplied by sqrt(0));
        by convention it is then reconstructed as exactly 0 in every
        representation, rather than leaking a stale placeholder like
        `mean`. Rows with non-zero weight must still round-trip exactly.
        """
        N = 8
        key = jax.random.PRNGKey(3)
        data = jax.random.normal(key, (N, 2))
        weights = jnp.ones(N).at[2].set(0.0).at[5].set(0.0)
        weights = weights / jnp.sum(weights)
        zero_rows = jnp.array([2, 5])
        nonzero_rows = jnp.array([i for i in range(N) if i not in (2, 5)])

        obs = SampledObs(data, weights)
        mean_ref = jnp.tensordot(weights, data, axes=(0, 0))

        _ = obs._normalized_obs      # o -> n: zero-weight rows collapse to 0 here
        raw = obs.observations       # n -> o
        centered = obs._centered_obs  # o -> c

        self.assertTrue(jnp.allclose(raw[nonzero_rows], data[nonzero_rows]))
        self.assertTrue(jnp.allclose(raw[zero_rows], 0.0))

        self.assertTrue(jnp.allclose(centered[nonzero_rows], data[nonzero_rows] - mean_ref))
        self.assertTrue(jnp.allclose(centered[zero_rows], 0.0))

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

def _make_lazy(observations, weights, batch_size):
    """
    Build a `LazySampledObs` the same way `sharded(..., yield_iter=True)`
    would: batches of `batch_size` (last one padded/trimmed), re-iterable
    from scratch on every pass.
    """
    batches = _reshape_in_batches(observations, batch_size)
    iterable = SizedIterable(
        reusable_iterable=lambda: iter(batches),
        n_iterations=len(batches),
        batch_size=batch_size,
    )
    return LazySampledObs(iterable, weights)

class TestLazySampledObs(unittest.TestCase):
    """
    `LazySampledObs` is the batched_jacobian backend for SampledObs: it must
    reproduce SampledObs's numbers exactly, just computed batch-by-batch
    instead of on one materialized array. batch_size=6 over 20 samples is
    chosen deliberately so batches are uneven ([6, 6, 6, 2]), which is the
    case that would break a naive equal-split implementation.
    """
    def _make_data(self, seed=0, n=20, n_obs=3):
        key = jax.random.PRNGKey(seed)
        k1, k2, k3 = jax.random.split(key, 3)
        obs = jax.random.normal(k1, (n, n_obs)) + 1j * jax.random.normal(k2, (n, n_obs))
        weights = jax.random.uniform(k3, (n,))
        weights = weights / jnp.sum(weights)
        return obs, weights

    def test_mean_matches_sampled_obs(self):
        obs, weights = self._make_data()
        dense = SampledObs(obs, weights)
        lazy = _make_lazy(obs, weights, batch_size=6)

        self.assertTrue(jnp.allclose(lazy.mean, dense.mean))

    def test_var_matches_sampled_obs(self):
        obs, weights = self._make_data()
        dense = SampledObs(obs, weights)
        lazy = _make_lazy(obs, weights, batch_size=6)

        self.assertTrue(jnp.allclose(lazy.var, dense.var))

    def test_error_of_mean_matches_sampled_obs(self):
        obs, weights = self._make_data()
        dense = SampledObs(obs, weights)
        lazy = _make_lazy(obs, weights, batch_size=6)

        self.assertTrue(jnp.allclose(lazy.error_of_mean, dense.error_of_mean))

    def test_get_covar_no_other_matches_sampled_obs(self):
        obs, weights = self._make_data()
        dense = SampledObs(obs, weights)
        lazy = _make_lazy(obs, weights, batch_size=6)

        self.assertTrue(jnp.allclose(lazy.get_covar(), dense.get_covar(), atol=1e-10))

    def test_get_covar_against_sampled_obs_other(self):
        obs1, weights = self._make_data(seed=0)
        obs2, _ = self._make_data(seed=1)
        dense1 = SampledObs(obs1, weights)
        dense2 = SampledObs(obs2, weights)
        lazy1 = _make_lazy(obs1, weights, batch_size=6)

        self.assertTrue(jnp.allclose(lazy1.get_covar(dense2), dense1.get_covar(dense2), atol=1e-10))

    def test_get_covar_against_lazy_sampled_obs_other(self):
        obs1, weights = self._make_data(seed=0)
        obs2, _ = self._make_data(seed=1)
        dense1 = SampledObs(obs1, weights)
        dense2 = SampledObs(obs2, weights)
        lazy1 = _make_lazy(obs1, weights, batch_size=6)
        lazy2 = _make_lazy(obs2, weights, batch_size=6)

        self.assertTrue(jnp.allclose(lazy1.get_covar(lazy2), dense1.get_covar(dense2), atol=1e-10))

    def test_get_covar_and_covar_var_matches_sampled_obs(self):
        obs, weights = self._make_data()
        dense = SampledObs(obs, weights)
        lazy = _make_lazy(obs, weights, batch_size=6)

        covar_d, var_re_d, var_im_d, cov_re_im_d = dense.get_covar_and_covar_var()
        covar_l, var_re_l, var_im_l, cov_re_im_l = lazy.get_covar_and_covar_var()

        self.assertTrue(jnp.allclose(covar_l, covar_d, atol=1e-10))
        self.assertTrue(jnp.allclose(var_re_l, var_re_d, atol=1e-10))
        self.assertTrue(jnp.allclose(var_im_l, var_im_d, atol=1e-10))
        self.assertTrue(jnp.allclose(cov_re_im_l, cov_re_im_d, atol=1e-10))

    def test_repeated_iteration_gives_consistent_results(self):
        """
        `.mean` and `.get_covar()` each do a full pass over `observations`;
        since the iterable recomputes from scratch every time it's iterated
        (that's the whole point of "lazy"), a second pass must reproduce
        identical numbers instead of silently drifting or erroring out.
        """
        obs, weights = self._make_data()
        lazy = _make_lazy(obs, weights, batch_size=6)

        mean_first = lazy.mean
        covar_first = lazy.get_covar()
        covar_second = lazy.get_covar()

        self.assertTrue(jnp.allclose(mean_first, lazy.mean))
        self.assertTrue(jnp.allclose(covar_first, covar_second))

    def test_transform_invalidates_cache(self):
        obs, weights = self._make_data(n_obs=1)
        lazy = _make_lazy(obs, weights, batch_size=6)

        _ = lazy.mean  # populate the cached_property before transforming

        lazy.transform(lambda x: 2 * x)

        self.assertTrue(jnp.allclose(lazy.mean, 2 * SampledObs(obs, weights).mean))

    def test_batch_weight_mismatch_raises(self):
        obs, weights = self._make_data(n=20)
        # `observations` batched with batch_size=6 -> 4 batches ([6,6,6,2]),
        # but told it has batch_size=4 -> weights would split into 5 batches.
        batches = _reshape_in_batches(obs, 6)
        iterable = SizedIterable(
            reusable_iterable=lambda: iter(batches),
            n_iterations=len(batches),
            batch_size=4,
        )
        with self.assertRaises(ValueError):
            LazySampledObs(iterable, weights)

if __name__ == "__main__":
    unittest.main()