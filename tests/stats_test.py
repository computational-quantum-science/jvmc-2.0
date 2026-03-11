import unittest

import jax
import jax.numpy as jnp

import jVMC_exp
from jVMC_exp.stats import SampledObs
import jVMC_exp.operator as op


class TestStats(unittest.TestCase):

    # def test_sampled_obs(self):
        
    #     Obs1Loc = jnp.array([1, 2, 3])
    #     Obs2Loc = jnp.array([[1, 4], [2, 5], [3, 7]])

    #     obs1 = SampledObs(Obs1Loc)
    #     obs2 = SampledObs(Obs2Loc)

    #     self.assertAlmostEqual(obs1.mean[0], 2.)
    #     self.assertAlmostEqual(obs1.var[0], 2./3)

    #     self.assertTrue(jnp.allclose(obs2.get_covar(), jnp.array([[2./3, 1],[1., 14./9]])))

    #     self.assertTrue(jnp.allclose(obs2.mean, jnp.array([2, 16./3])))
    #     self.assertTrue(jnp.allclose(obs1.get_covar(obs2), jnp.array([2./3, 1.])))

    #     self.assertTrue(jnp.allclose(obs1.get_covar(obs2), obs1.get_covar_obs(obs2).mean))

    #     self.assertTrue(jnp.allclose(obs1.get_covar_var(obs2), obs1.get_covar_obs(obs2).var))

    #     O = obs2._normalized_obs.reshape((-1, 2))
    #     self.assertTrue(jnp.allclose(obs2.tangent_kernel, jnp.matmul(O, jnp.conj(jnp.transpose(O)))))
        
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

if __name__ == "__main__":
    unittest.main()