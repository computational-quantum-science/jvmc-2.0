import gc
import unittest
import warnings
import jax
import jax.numpy as jnp

import jVMC_exp

from jVMC_exp.stats import SampledObs, LazySampledObs
from jVMC_exp.sharding_config import DEVICE_SHARDING, SizedIterable, BatchLayout, pad_to_devices

# 3 samples per device and batch: over 20 samples the last batch is never full
BATCH_SIZE = 3 * jax.device_count()

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
    would: samples padded to a multiple of the number of devices, batches of
    `batch_size` laid out by a `BatchLayout` (last one trimmed), re-iterable
    from scratch on every pass.
    """
    observations = pad_to_devices(observations)
    layout = BatchLayout(observations.shape[0], batch_size)
    batches = layout.split(observations)
    iterable = SizedIterable(
        reusable_iterable=lambda: iter(batches),
        n_iterations=len(batches),
        batch_size=batch_size,
        layout=layout,
    )
    return LazySampledObs(iterable, weights)

class TestLazySampledObs(unittest.TestCase):
    """
    `LazySampledObs` is the batched_jacobian backend for SampledObs: it must
    reproduce SampledObs's numbers exactly, just computed batch-by-batch
    instead of on one materialized array. 20 samples and 3 samples per batch
    on every device are chosen deliberately so batches are uneven ([3, 3, 3,
    3, 3, 3, 2] per device), which is the case that would break a naive
    equal-split implementation.
    """
    def _make_data(self, seed=0, n=20 * jax.device_count(), n_obs=3):
        key = jax.random.PRNGKey(seed)
        k1, k2, k3 = jax.random.split(key, 3)
        obs = jax.random.normal(k1, (n, n_obs)) + 1j * jax.random.normal(k2, (n, n_obs))
        weights = jax.random.uniform(k3, (n,))
        weights = weights / jnp.sum(weights)
        return obs, weights

    def test_mean_matches_sampled_obs(self):
        obs, weights = self._make_data()
        dense = SampledObs(obs, weights)
        lazy = _make_lazy(obs, weights, batch_size=BATCH_SIZE)

        self.assertTrue(jnp.allclose(lazy.mean, dense.mean))

    def test_var_matches_sampled_obs(self):
        obs, weights = self._make_data()
        dense = SampledObs(obs, weights)
        lazy = _make_lazy(obs, weights, batch_size=BATCH_SIZE)

        self.assertTrue(jnp.allclose(lazy.var, dense.var))

    def test_error_of_mean_matches_sampled_obs(self):
        obs, weights = self._make_data()
        dense = SampledObs(obs, weights)
        lazy = _make_lazy(obs, weights, batch_size=BATCH_SIZE)

        self.assertTrue(jnp.allclose(lazy.error_of_mean, dense.error_of_mean))

    def test_get_covar_no_other_matches_sampled_obs(self):
        obs, weights = self._make_data()
        dense = SampledObs(obs, weights)
        lazy = _make_lazy(obs, weights, batch_size=BATCH_SIZE)

        self.assertTrue(jnp.allclose(lazy.get_covar(), dense.get_covar(), atol=1e-10))

    def test_get_covar_against_sampled_obs_other(self):
        obs1, weights = self._make_data(seed=0)
        obs2, _ = self._make_data(seed=1)
        dense1 = SampledObs(obs1, weights)
        dense2 = SampledObs(obs2, weights)
        lazy1 = _make_lazy(obs1, weights, batch_size=BATCH_SIZE)

        self.assertTrue(jnp.allclose(lazy1.get_covar(dense2), dense1.get_covar(dense2), atol=1e-10))

    def test_get_covar_against_lazy_sampled_obs_other(self):
        obs1, weights = self._make_data(seed=0)
        obs2, _ = self._make_data(seed=1)
        dense1 = SampledObs(obs1, weights)
        dense2 = SampledObs(obs2, weights)
        lazy1 = _make_lazy(obs1, weights, batch_size=BATCH_SIZE)
        lazy2 = _make_lazy(obs2, weights, batch_size=BATCH_SIZE)

        self.assertTrue(jnp.allclose(lazy1.get_covar(lazy2), dense1.get_covar(dense2), atol=1e-10))

    def test_get_covar_and_covar_var_matches_sampled_obs(self):
        obs, weights = self._make_data()
        dense = SampledObs(obs, weights)
        lazy = _make_lazy(obs, weights, batch_size=BATCH_SIZE)

        covar_d = dense.get_covar()
        covar_l = lazy.get_covar()

        self.assertTrue(jnp.allclose(covar_l, covar_d, atol=1e-10))

    def test_repeated_iteration_gives_consistent_results(self):
        """
        `.mean` and `.get_covar()` each do a full pass over `observations`;
        since the iterable recomputes from scratch every time it's iterated
        (that's the whole point of "lazy"), a second pass must reproduce
        identical numbers instead of silently drifting or erroring out.
        """
        obs, weights = self._make_data()
        lazy = _make_lazy(obs, weights, batch_size=BATCH_SIZE)

        mean_first = lazy.mean
        covar_first = lazy.get_covar()
        covar_second = lazy.get_covar()

        self.assertTrue(jnp.allclose(mean_first, lazy.mean))
        self.assertTrue(jnp.allclose(covar_first, covar_second))

    def test_transform_invalidates_cache(self):
        obs, weights = self._make_data(n_obs=1)
        lazy = _make_lazy(obs, weights, batch_size=BATCH_SIZE)

        _ = lazy.mean  # populate the cached_property before transforming

        lazy.transform(lambda x: 2 * x)

        self.assertTrue(jnp.allclose(lazy.mean, 2 * SampledObs(obs, weights).mean))

    def test_batch_weight_mismatch_raises(self):
        obs, weights = self._make_data()
        # The layout of `observations` describes all samples, but fewer weights are given
        lazy = _make_lazy(obs, weights, batch_size=BATCH_SIZE)
        iterable = lazy.observations
        with self.assertRaises(ValueError):
            LazySampledObs(iterable, weights[:-jax.device_count()])

    def test_iterable_without_layout_raises(self):
        obs, weights = self._make_data()
        batches = list(_make_lazy(obs, weights, batch_size=BATCH_SIZE).observations)
        iterable = SizedIterable(
            reusable_iterable=lambda: iter(batches),
            n_iterations=len(batches),
            batch_size=BATCH_SIZE,
        )
        with self.assertRaises(ValueError):
            LazySampledObs(iterable, weights)

    def test_get_covar_with_different_layouts_raises(self):
        obs1, weights = self._make_data(seed=0)
        obs2, _ = self._make_data(seed=1)
        lazy1 = _make_lazy(obs1, weights, batch_size=BATCH_SIZE)
        lazy2 = _make_lazy(obs2, weights, batch_size=2 * BATCH_SIZE)
        with self.assertRaises(ValueError):
            lazy1.get_covar(lazy2)

    @unittest.skipIf(jax.device_count() == 1, "Every number of samples is divisible by one device")
    def test_indivisible_number_of_samples_matches_sampled_obs(self):
        """
        Samples that are not divisible by the number of devices are padded, and
        the padding samples get zero weight, like in SampledObs.
        """
        obs1, weights = self._make_data(seed=0, n=20 * jax.device_count() + 1)
        obs2, _ = self._make_data(seed=1, n=20 * jax.device_count() + 1)
        dense1, dense2 = SampledObs(obs1, weights), SampledObs(obs2, weights)
        lazy1 = _make_lazy(obs1, weights, batch_size=BATCH_SIZE)
        lazy2 = _make_lazy(obs2, weights, batch_size=BATCH_SIZE)

        self.assertTrue(jnp.allclose(lazy1.mean, dense1.mean))
        self.assertTrue(jnp.allclose(lazy1.var, dense1.var))
        self.assertTrue(jnp.allclose(lazy1.get_covar(), dense1.get_covar(), atol=1e-10))
        self.assertTrue(jnp.allclose(lazy1.get_covar(dense2), dense1.get_covar(dense2), atol=1e-10))
        self.assertTrue(jnp.allclose(lazy1.get_covar(lazy2), dense1.get_covar(dense2), atol=1e-10))

    def test_batches_line_up_with_weights(self):
        """
        On more than one device a batch does not consist of consecutive samples,
        so the weights must be split with the same layout as the observations.
        Tagging every sample with its index checks that each batch of weights
        belongs to the same samples as the corresponding batch of observations.
        """
        n = 20 * jax.device_count()
        index = jnp.arange(n, dtype=jnp.float64)
        lazy = _make_lazy(index[:, None], index + 1, batch_size=BATCH_SIZE)
        for batch, weights in zip(lazy.observations, lazy._weights):
            self.assertTrue(jnp.allclose(batch[:, 0] + 1, weights * jnp.sum(index + 1)))

def _num_live_arrays(shape):
    return sum(1 for x in jax.live_arrays() if x.shape == shape and not x.is_deleted())

class TestSampledObsMemory(unittest.TestCase):
    """
    The Jacobian is the largest array of a TDVP step: the covariance with the local
    energy is summed over chunks of samples, and a freshly computed Jacobian is converted
    between its raw, centered and normalized forms in place.
    """
    def setUp(self):
        # Not a multiple of small chunk sizes, and a shape no other array of the tests has
        self.N, self.n = 6 * jax.device_count(), 7
        k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(11), 4)
        self.data = jax.random.normal(k1, (self.N, self.n)) + 1j * jax.random.normal(k2, (self.N, self.n))
        self.other = jax.random.normal(k3, (self.N,)) + 0.5j * jax.random.normal(k4, (self.N,))
        weights = jax.random.uniform(jax.random.PRNGKey(12), (self.N,)) + 0.1
        self.weights = weights / jnp.sum(weights)

    def _reference_covar(self, data, other):
        """sum_i w_i conj(data_i - <data>) (other_i - <other>)"""
        d = data - jnp.tensordot(self.weights, data, axes=(0, 0))
        o = other - jnp.tensordot(self.weights, other, axes=(0, 0))
        return jnp.einsum("i,ij,i->j", self.weights, jnp.conj(d), o)

    def test_weighted_covar_in_chunks(self):
        """
        Real and complex data and other observable, with the whole shard at once, the
        default budget and one sample per chunk; also when the number of samples is not
        divisible by the number of devices (padded by SampledObs).
        """
        old = jVMC_exp.global_defs.MATMUL_CHUNK_BYTES
        try:
            for budget in (0, old, 1):
                jVMC_exp.global_defs.update(matmul_chunk_bytes=budget)
                for data in (self.data, self.data.real):
                    for other in (self.other, self.other.real):
                        for N in (self.N, self.N - 1):
                            with self.subTest(budget=budget, data=str(data.dtype), other=str(other.dtype), N=N):
                                w = self.weights[:N] / jnp.sum(self.weights[:N])
                                obs = SampledObs(data[:N], w)
                                covar = obs.get_covar(SampledObs(other[:N], w))
                                d = data[:N] - jnp.tensordot(w, data[:N], axes=(0, 0))
                                o = other[:N] - jnp.tensordot(w, other[:N], axes=(0, 0))
                                ref = jnp.einsum("i,ij,i->j", w, jnp.conj(d), o)
                                self.assertEqual(covar.shape, (self.n,))
                                self.assertTrue(jnp.allclose(covar, ref, atol=1e-12))
                                if not jnp.iscomplexobj(data) and not jnp.iscomplexobj(other):
                                    self.assertFalse(jnp.iscomplexobj(covar))
        finally:
            jVMC_exp.global_defs.update(matmul_chunk_bytes=old)

    def test_donated_observations_are_converted_in_place(self):
        # a copy that the test gives away, sharded over the samples like a computed Jacobian
        data = jax.device_put(self.data + 0, DEVICE_SHARDING)
        obs = SampledObs(data, self.weights, donate=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            normalized = obs._normalized_obs
        self.assertTrue(data.is_deleted())
        # the conversion reused the buffer (JAX warns about donated buffers it could not use)
        self.assertFalse([w for w in caught if "donated" in str(w.message)])
        mean = jnp.tensordot(self.weights, self.data, axes=(0, 0))
        ref = jnp.sqrt(self.weights)[:, None] * (self.data - mean)
        self.assertTrue(jnp.allclose(normalized, ref, atol=1e-12))

    def test_given_and_handed_out_observations_are_kept(self):
        """Without donate, and for every buffer handed out, a conversion makes a new buffer."""
        obs = SampledObs(self.data, self.weights)
        normalized = obs._normalized_obs
        self.assertFalse(self.data.is_deleted())
        centered = obs._centered_obs
        raw = obs.observations
        _ = obs.var  # back to normalized, internally
        for x in (normalized, centered, raw):
            self.assertFalse(x.is_deleted())
        self.assertTrue(jnp.allclose(raw, self.data, atol=1e-12))
        mean = jnp.tensordot(self.weights, self.data, axes=(0, 0))
        self.assertTrue(jnp.allclose(centered, self.data - mean, atol=1e-12))

    def test_one_copy_through_the_tdvp_conversions(self):
        """
        Covariance with the local energy (raw), variance and normalized Jacobian as the
        optimizer uses them: no copy of the Jacobian is left behind.
        """
        shape = (self.N, self.n)
        gc.collect()
        before = _num_live_arrays(shape)
        obs = SampledObs(jax.device_put(self.data + 0, DEVICE_SHARDING), self.weights, donate=True)
        self.assertEqual(_num_live_arrays(shape), before + 1)
        o_loc = SampledObs(self.other, self.weights)
        covar = obs.get_covar(o_loc)
        _ = obs.var
        _ = obs.get_covar(o_loc)
        normalized = obs._normalized_obs
        self.assertEqual(_num_live_arrays(shape), before + 1)
        self.assertTrue(jnp.allclose(covar, self._reference_covar(self.data, self.other), atol=1e-12))
        mean = jnp.tensordot(self.weights, self.data, axes=(0, 0))
        self.assertTrue(jnp.allclose(normalized, jnp.sqrt(self.weights)[:, None] * (self.data - mean), atol=1e-12))

    def test_covariance_with_itself(self):
        """`other` is converted first: converting it must not overwrite the data of the product."""
        obs = SampledObs(jax.device_put(self.data[:, :1], DEVICE_SHARDING), self.weights, donate=True)
        covar = obs.get_covar(obs)
        ref = self._reference_covar(self.data[:, :1], self.data[:, 0])
        self.assertTrue(jnp.allclose(covar, ref, atol=1e-12))

    def test_transform_keeps_handed_out_buffer(self):
        """An element-wise identity gives back the same buffer: it stays protected."""
        obs = SampledObs(jax.device_put(self.data + 0, DEVICE_SHARDING), self.weights, donate=True)
        raw = obs.observations
        obs.transform(lambda x: x)
        _ = obs._normalized_obs
        self.assertFalse(raw.is_deleted())

if __name__ == "__main__":
    unittest.main()