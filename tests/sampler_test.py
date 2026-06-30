import unittest
import flax.linen as nn
import jax
import jax.random as random
import jax.numpy as jnp
import numpy as np

import jVMC_exp
import jVMC_exp.nets as nets
from jVMC_exp.vqs import NQS
import jVMC_exp.sampler as sampler
from jVMC_exp.symmetry.lattice_symetries import chain_reflection_symmetry, spin_flip_symmetry, square_translation_symmetry

@jax.jit
def state_to_int(S):
    powers = 2 ** jnp.arange(S.shape[-1])[::-1]
    return jnp.dot(S, powers).astype(jnp.int64)

def _translation_projector(L: int):
    return square_translation_symmetry(L, 1, "spin")

def _translation_reflection_spinflip_projector(L: int):
    return _translation_projector(L) * chain_reflection_symmetry(L, "spin") * spin_flip_symmetry(L, 1, "spin")

class _PeakedGeneratorNet(nn.Module):
    peak_state: tuple[int, ...]

    @nn.compact
    def __call__(self, s):
        offset = self.param("offset", nn.initializers.zeros, ())
        peak = jnp.asarray(self.peak_state, dtype=s.dtype)
        return (offset + jnp.where(jnp.all(s == peak), 0.0, -40.0)).astype(jnp.complex128)

    def sample(self, key):
        del key
        return jnp.asarray(self.peak_state, dtype=jnp.int32)

def _test_sampling(net, test_class: unittest.TestCase, mu=2, log_prob_factor=0.5):
    L = 4
    num_samples = 2 ** 18
    num_chains = 2 ** 16

    weights = jnp.array(
        [0.23898957, 0.12614753, 0.19479055, 0.17325271, 0.14619853, 0.21392751,
            0.19648707, 0.17103704, -0.15457255, 0.10954413, 0.13228065, -0.14935214,
            -0.09963073, 0.17610707, 0.13386381, -0.14836467]
    )
    weights = weights

    # Set up variational wave function
    psi = NQS(net, L, num_samples, seed=1234)
    exact_psi = NQS(net, L, 2 ** L, seed=1234)
    psi.parameters = weights
    exact_psi.parameters = weights

    # Set up exact sampler
    exact_sampler = sampler.ExactSampler(exact_psi, logProbFactor=log_prob_factor)

    # Set up MCMC sampler
    proposer = jVMC_exp.propose.SpinFlip()
    mc_sampler = sampler.MCSampler(
        psi, updateProposer=proposer, key=random.PRNGKey(0), 
        numChains=num_chains, numSamples=num_samples,
        mu=mu, logProbFactor=log_prob_factor
    )

    # Compute exact probabilities
    _, _, pex = exact_sampler.sample()

    # Get samples from MCMC sampler
    samples, _, p = mc_sampler.sample()

    test_class.assertTrue(jnp.array([samples.shape[0],])[None, None, ...] >= num_samples)

    # Compute histogram of sampled configurations
    # samples_int = jax.vmap(state_to_int)(samples)
    samples_int = state_to_int(samples)
    pmc, _ = np.histogram(samples_int, bins=np.arange(0, 17), weights=p)
    pmc = pmc / jnp.sum(pmc)

    # Compare histogram to exact probabilities
    test_class.assertTrue(jnp.max(jnp.abs(pmc - pex)) < 2e-3)

def _test_autoreg_sampling(net, test_class: unittest.TestCase, L=(4,), mu=2, log_prob_factor=0.5):
    num_samples = 2 ** 18
    num_chains = 2 ** 16

    psi = NQS(net, L, num_samples, seed=1234)
    exact_psi = NQS(net, L, 2 ** sum(L), seed=1234)

    # Set up exact sampler
    exact_sampler = sampler.ExactSampler(exact_psi)

    # Set up MCMC sampler
    proposer = jVMC_exp.propose.SpinFlip()
    mc_sampler = sampler.MCSampler(
        psi, updateProposer=proposer, key=random.PRNGKey(0), 
        numChains=num_chains, numSamples=num_samples,
        mu=mu, logProbFactor=log_prob_factor
    )

    psi.parameters = 2 * psi.parameters_flat
    exact_psi.parameters = psi.parameters

    # Compute exact probabilities
    _, _, pex = exact_sampler.sample()
    samples, _, p = mc_sampler.sample()

    test_class.assertTrue(jnp.array([samples.shape[0],])[None, None, ...] >= num_samples)

    # Compute histogram of sampled configurations
    samples_int = state_to_int(samples)
    pmc, _ = np.histogram(samples_int, bins=np.arange(0, 17), weights=p)
    pmc = pmc / jnp.sum(pmc)

    test_class.assertTrue(jnp.max(jnp.abs(pmc - pex.reshape((-1,))[:16])) < 1.1e-3)

class TestMC(unittest.TestCase):

    def test_direct_sampling_randomizes_projected_generator_over_symmetry_orbit(self):
        L = 4
        peak_state = (1, 0, 0, 0)
        orbit = _translation_projector(L)
        net = _PeakedGeneratorNet(peak_state)
        psi = NQS(net, L, 64, seed=1234, orbit=orbit, symmetry_average="sep")
        mc_sampler = sampler.MCSampler(
            psi,
            updateProposer=None,
            key=random.PRNGKey(17),
            numChains=4,
            numSamples=256,
        )

        samples, _, _ = mc_sampler.sample()
        expected = {
            tuple(row.tolist())
            for row in np.asarray(orbit.transformed_states(jnp.asarray(peak_state, dtype=jnp.int32)))
        }
        sampled = {tuple(row.tolist()) for row in np.asarray(samples)}

        self.assertEqual(sampled, expected)

    def test_MCMC_sampling(self):
        rbm = nets.CpxRBM(numHidden=2, bias=False)
        orbit = _translation_reflection_spinflip_projector(4)
        net = orbit * rbm
        
        _test_sampling(net, self)

    def test_MCMC_sampling_with_mu(self):
        rbm = nets.CpxRBM(numHidden=2, bias=False)
        orbit = _translation_projector(4)
        net = orbit * rbm
        
        _test_sampling(net, self, mu=1)

    def test_MCMC_sampling_with_logProbFactor(self):
        rbm = nets.CpxRBM(numHidden=2, bias=False)
        orbit = _translation_projector(4)
        net = orbit * rbm
        
        _test_sampling(net, self, log_prob_factor=1)

    def test_MCMC_sampling_with_two_nets(self):
        rbm1 = nets.RBM(numHidden=2, bias=False)
        rbm2 = nets.RBM(numHidden=2, bias=False)
        model = jVMC_exp.nets.TwoNets((rbm1, rbm2))
        orbit = _translation_projector(4)
        net = orbit * model

        _test_sampling(net, self)

    def test_MCMC_sampling_ratio(self):
        rbm = nets.CpxRBM_ratio(numHidden=2, bias=False)
        orbit = _translation_reflection_spinflip_projector(4)
        net = orbit * rbm
        
        _test_sampling(net, self)


    # def test_autoregressive_sampling(self):
    #     rnn = nets.RNN1DGeneral(L=4, hiddenSize=5, depth=2)
    #     rbm = nets.RBM(numHidden=2, bias=False)
    #     model = jVMC_exp.nets.TwoNets((rnn, rbm))
    #     orbit = _translation_projector(4)
    #     net = orbit * model
        
    #     _test_autoreg_sampling(net, self)

    # def test_autoregressive_sampling_with_symmetries(self):
    #     rnn = nets.RNN1DGeneral(L=4, hiddenSize=5, realValuedOutput=True)
    #     rbm = nets.RBM(numHidden=2, bias=False)
    #     model = jVMC_exp.nets.TwoNets((rnn, rbm))
    #     orbit = _translation_projector(4)
    #     net = orbit * model
        
    #     _test_autoreg_sampling(net, self)

    # def test_autoregressive_sampling_with_lstm(self):
    #     rnn = nets.RNN1DGeneral(L=4, hiddenSize=5, cell="LSTM", realValuedParams=True, realValuedOutput=True, inputDim=2)
    #     rbm = nets.RBM(numHidden=2, bias=False)
    #     model = jVMC_exp.nets.TwoNets((rnn, rbm))
    #     orbit = _translation_projector(4)
    #     net = orbit * model

    #     _test_autoreg_sampling(net, self)

    # def test_autoregressive_sampling_with_gru(self):
    #     rnn = nets.RNN1DGeneral(L=4, hiddenSize=5, cell="GRU", realValuedParams=True, realValuedOutput=True, inputDim=2)
    #     rbm = nets.RBM(numHidden=2, bias=False)

    #     _test_autoreg_sampling((rnn, rbm), self)

#     def test_autoregressive_sampling_with_rnn2d(self):
#         rnn = nets.RNN2DGeneral(L=2, hiddenSize=5, cell="RNN", realValuedParams=True, realValuedOutput=True)
#         model = jVMC_exp.nets.TwoNets((rnn, rnn))
#         orbit = square_translation_symmetry(2, 2, "spin")
#         net = orbit * model

#         _test_autoreg_sampling(net, self, L=(4, 4))

#     def test_autoregressive_sampling_with_rnn2d_symmetric(self):
#         rnn = nets.RNN2DGeneral(L=2, hiddenSize=5, cell="RNN", realValuedParams=True, realValuedOutput=True)
#         model = jVMC_exp.nets.TwoNets((rnn, rnn))
#         orbit = square_translation_symmetry(2, 2, "spin")
#         net = orbit * model

#         _test_autoreg_sampling(net, self, L=(4, 4))

#     def test_autoregressive_sampling_with_lstm2d(self):
#         rnn = nets.RNN2DGeneral(L=2, hiddenSize=5, cell="LSTM", realValuedParams=True, realValuedOutput=True)
#         model = jVMC_exp.nets.TwoNets((rnn, rnn))
#         orbit = square_translation_symmetry(2, 2, "spin")
#         net = orbit * model

#         _test_autoreg_sampling(net, self, L=(4, 4))

class TestExactSampler(unittest.TestCase):
    def test_exact_sampler(self):
        L = 4

        weights = jnp.array(
            [0.23898957, 0.12614753, 0.19479055, 0.17325271, 0.14619853, 0.21392751,
             0.19648707, 0.17103704, -0.15457255, 0.10954413, 0.13228065, -0.14935214,
             -0.09963073, 0.17610707, 0.13386381, -0.14836467]
        )

        # Set up variational wave function
        rbm = nets.CpxRBM(numHidden=2, bias=False)
        psi = NQS(rbm, L, 2**L)
        psi.parameters = weights

        # Set up exact sampler
        exact_sampler = sampler.ExactSampler(psi)  

        # Compute exact probabilities
        s, psi_s, _ = exact_sampler.sample()
        self.assertTrue(jnp.max(jnp.abs((psi(s) - psi_s) / psi_s)) < 1e-14)

if __name__ == "__main__":
    unittest.main()
