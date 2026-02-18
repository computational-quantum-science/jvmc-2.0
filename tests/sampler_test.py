import unittest
import jax
import jax.random as random
import jax.numpy as jnp
import numpy as np

import jVMC
import jVMC.nets as nets
from jVMC.vqs import NQS
import jVMC.sampler as sampler
from jVMC.nets.sym_wrapper import SymNet

@jax.jit
def state_to_int(S):
    powers = 2 ** jnp.arange(S.shape[-1])[::-1]
    return jnp.dot(S, powers).astype(jnp.int64)

def _test_sampling(net, test_class: unittest.TestCase, mu=2, log_prob_factor=0.5, two_nets=False, test_two_samplers=False):
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
    proposer = jVMC.propose.SpinFlip()
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

    if test_two_samplers:
        num_samples = 2 ** 6
        num_chains = 2 ** 4
        psi1 = NQS(net, L, num_samples, seed=1234)
        # Set up another MCMC sampler
        proposer = jVMC.propose.SpinFlip()
        mc_sampler = sampler.MCSampler(
            psi1, updateProposer=proposer, key=random.PRNGKey(0), 
            numChains=num_chains, numSamples=num_samples,
            mu=mu, logProbFactor=log_prob_factor
        )
        s, psi_s, _ = mc_sampler.sample(parameters=psi.parameters)
        psi_s1 = psi(s)
        
        test_class.assertTrue(jnp.max(jnp.abs((psi_s - psi_s1) / psi_s)) < 1e-14)

def _test_autoreg_sampling(net, test_class: unittest.TestCase, L=(4,), mu=2, log_prob_factor=0.5, test_two_samplers=False):
    num_samples = 2 ** 18
    num_chains = 2 ** 16

    psi = NQS(net, L, num_samples, seed=1234)
    exact_psi = NQS(net, L, 2 ** sum(L), seed=1234)

    # Set up exact sampler
    exact_sampler = sampler.ExactSampler(exact_psi)

    # Set up MCMC sampler
    proposer = jVMC.propose.SpinFlip()
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

    if test_two_samplers:
        num_samples = 2 ** 6
        num_chains = 2 ** 4
        psi1 = NQS(net, L, num_samples, seed=1234)
        # Set up another MCMC sampler
        proposer = jVMC.propose.SpinFlip()
        mc_sampler = sampler.MCSampler(
            psi1, updateProposer=proposer, key=random.PRNGKey(0), 
            numChains=num_chains, numSamples=num_samples,
            mu=mu, logProbFactor=log_prob_factor
        )
        s, psi_s, _ = mc_sampler.sample(parameters=psi.parameters)
        psi_s1 = psi(s)
        
        test_class.assertTrue(jnp.max(jnp.abs((psi_s - psi_s1) / psi_s)) < 1e-14)

class TestMC(unittest.TestCase):

    def test_MCMC_sampling(self):
        rbm = nets.CpxRBM(numHidden=2, bias=False)
        orbit = jVMC.util.symmetries.get_orbit_1D(4, "translation", "reflection", "spinflip")
        net = SymNet(net=rbm, orbit=orbit)
        
        _test_sampling(net, self, test_two_samplers=True)

    def test_MCMC_sampling_with_mu(self):
        rbm = nets.CpxRBM(numHidden=2, bias=False)
        orbit = jVMC.util.symmetries.get_orbit_1D(4)
        net = SymNet(net=rbm, orbit=orbit)
        
        _test_sampling(net, self, mu=1)

    def test_MCMC_sampling_with_logProbFactor(self):
        rbm = nets.CpxRBM(numHidden=2, bias=False)
        orbit = jVMC.util.symmetries.get_orbit_1D(4)
        net = SymNet(net=rbm, orbit=orbit)
        
        _test_sampling(net, self, log_prob_factor=1)

    def test_MCMC_sampling_with_two_nets(self):
        rbm1 = nets.RBM(numHidden=2, bias=False)
        rbm2 = nets.RBM(numHidden=2, bias=False)
        model = jVMC.nets.TwoNets((rbm1, rbm2))
        orbit = jVMC.util.symmetries.get_orbit_1D(4)
        net = SymNet(net=model, orbit=orbit)

        _test_sampling(net, self, two_nets=True)

    def test_autoregressive_sampling(self):
        rnn = nets.RNN1DGeneral(L=4, hiddenSize=5, depth=2)
        rbm = nets.RBM(numHidden=2, bias=False)
        model = jVMC.nets.TwoNets((rnn, rbm))
        orbit = jVMC.util.symmetries.get_orbit_1D(4)
        net = SymNet(net=model, orbit=orbit, avgFun=jVMC.nets.sym_wrapper.avgFun_Coefficients_Sep)
        
        _test_autoreg_sampling(net, self, test_two_samplers=True)

    def test_autoregressive_sampling_with_symmetries(self):
        rnn = nets.RNN1DGeneral(L=4, hiddenSize=5, realValuedOutput=True)
        rbm = nets.RBM(numHidden=2, bias=False)
        model = jVMC.nets.TwoNets((rnn, rbm))
        orbit = jVMC.util.symmetries.get_orbit_1D(4, "translation")
        net = SymNet(net=model, orbit=orbit, avgFun=jVMC.nets.sym_wrapper.avgFun_Coefficients_Sep)
        
        _test_autoreg_sampling(net, self)

    def test_autoregressive_sampling_with_lstm(self):
        rnn = nets.RNN1DGeneral(L=4, hiddenSize=5, cell="LSTM", realValuedParams=True, realValuedOutput=True, inputDim=2)
        rbm = nets.RBM(numHidden=2, bias=False)
        model = jVMC.nets.TwoNets((rnn, rbm))
        orbit = jVMC.util.symmetries.get_orbit_1D(4, "translation")
        net = SymNet(net=model, orbit=orbit, avgFun=jVMC.nets.sym_wrapper.avgFun_Coefficients_Sep)

        _test_autoreg_sampling(net, self)

    def test_autoregressive_sampling_with_gru(self):
        rnn = nets.RNN1DGeneral(L=4, hiddenSize=5, cell="GRU", realValuedParams=True, realValuedOutput=True, inputDim=2)
        rbm = nets.RBM(numHidden=2, bias=False)

        _test_autoreg_sampling((rnn, rbm), self)

#     def test_autoregressive_sampling_with_rnn2d(self):
#         rnn = nets.RNN2DGeneral(L=2, hiddenSize=5, cell="RNN", realValuedParams=True, realValuedOutput=True)
#         model = jVMC.nets.TwoNets((rnn, rnn))
#         orbit = jVMC.util.symmetries.get_orbit_2D_square(2)
#         net = SymNet(net=model, orbit=orbit, avgFun=jVMC.nets.sym_wrapper.avgFun_Coefficients_Sep)

#         _test_autoreg_sampling(net, self, L=(4, 4))

#     def test_autoregressive_sampling_with_rnn2d_symmetric(self):
#         rnn = nets.RNN2DGeneral(L=2, hiddenSize=5, cell="RNN", realValuedParams=True, realValuedOutput=True)
#         model = jVMC.nets.TwoNets((rnn, rnn))
#         orbit = jVMC.util.symmetries.get_orbit_2D_square(2, "translation")
#         net = SymNet(net=model, orbit=orbit, avgFun=jVMC.nets.sym_wrapper.avgFun_Coefficients_Sep)

#         _test_autoreg_sampling(net, self, L=(4, 4))

#     def test_autoregressive_sampling_with_lstm2d(self):
#         rnn = nets.RNN2DGeneral(L=2, hiddenSize=5, cell="LSTM", realValuedParams=True, realValuedOutput=True)
#         model = jVMC.nets.TwoNets((rnn, rnn))
#         orbit = jVMC.util.symmetries.get_orbit_2D_square(2, "translation")
#         net = SymNet(net=model, orbit=orbit, avgFun=jVMC.nets.sym_wrapper.avgFun_Coefficients_Sep)

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
        psi = NQS(rbm, L, 2 ** L)

        # Set up exact sampler
        exact_sampler = sampler.ExactSampler(psi)

        p0 = psi.parameters
        psi.parameters = weights

        # Compute exact probabilities
        s, psi_s, _ = exact_sampler.sample()

        self.assertTrue(jnp.max((psi(s) - psi_s) / psi_s) < 1e-14)
        
        s, psi_s, _ = exact_sampler.sample(parameters=p0)

        psi.parameters = p0
        self.assertTrue(jnp.max((psi(s) - psi_s) / psi_s) < 1e-14)

if __name__ == "__main__":
    unittest.main()