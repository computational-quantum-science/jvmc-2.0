import unittest
import jax.random as random
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

import jVMC_exp
import jVMC_exp.operator.discrete as op
import jVMC_exp.sampler
import jVMC_exp.nets as nets
from jVMC_exp.vqs import NQS
from jVMC_exp.global_defs import DT_SAMPLES

L = 4
LDIM = 2
KEY = random.PRNGKey(3)
NUM_SAMPLES = 2 ** 6

class Target(nn.Module):
  """
  Target wave function, returns a vector with the same dimension as the Hilbert space

    Initialization arguments:
        * ``L``: System size
        * ``d``: local Hilbert space dimension
        * ``delta``: small number to avoid log(0)

    """
  L: int
  d: float = 2.00
  delta: float = 1e-15

  @nn.compact
  def __call__(self, s):
    kernel = self.param('kernel', nn.initializers.constant(1), (int(self.d ** self.L)))
    # return amplitude for state s
    idx = ((self.d ** jnp.arange(self.L)).dot(s[::-1])).astype(int) # NOTE that the state is reversed to account for different bit conventions used in openfermion
    
    return jnp.log(abs(kernel[idx] + self.delta)) + 1.j * jnp.angle(kernel[idx])

class TestOperator(unittest.TestCase):

    def test_nonzeros(self):
        s = random.randint(KEY, (NUM_SAMPLES, L), 0, 2, dtype=DT_SAMPLES)

        h = 0
        h += 2. * op.SigmaPlus(0)
        h += 2. * op.SigmaPlus(1)
        h += 2. * op.SigmaPlus(2)

        sp, matEls = h.get_conn_elements(s, NUM_SAMPLES)
        logPsi = jnp.ones(s.shape[:-1])
        logPsiSP = jnp.ones(sp.shape[:-1])
        E_loc = h._get_O_loc(logPsi, logPsiSP, matEls, batch_size=NUM_SAMPLES)

        self.assertTrue(jnp.sum(jnp.abs(E_loc - 2. * jnp.sum(-(s[..., :3] - 1), axis=-1))) < 1e-7)

    def test_op_with_arguments(self):
        s = random.randint(KEY, (NUM_SAMPLES, L), 0, 2, dtype=DT_SAMPLES)
        
        def f(t, **kwargs):
            return 2.0 * t
        h = f * op.SigmaPlus(0) + f * op.SigmaPlus(1) + f * op.SigmaPlus(2)

        for t in [0.5, 2, 13.9]:
            sp, matEls = h.get_conn_elements(s, NUM_SAMPLES, t=t)

            logPsi = jnp.ones(s.shape[:-1])
            logPsiSP = jnp.ones(sp.shape[:-1])

            E_loc = h._get_O_loc(logPsi, logPsiSP, matEls, batch_size=NUM_SAMPLES)

            self.assertTrue(jnp.sum(jnp.abs(E_loc - f(t) * jnp.sum(-(s[..., :3] - 1), axis=-1))) < 1e-7)

    def test_op_2d(self):
        s = random.randint(KEY, (NUM_SAMPLES, L, L), 0, 2, dtype=DT_SAMPLES)

        h = 0.3 * op.SigmaPlus(0) + 1.1 * op.SigmaPlus(1) + 0.15 * op.SigmaPlus(4)

        sp, matEls = h.get_conn_elements(s, NUM_SAMPLES)

        self.assertTrue(sp.shape[2:] == (L, L))

    def test_td_prefactor(self):
        h = op.SigmaZ(0) + op.SigmaZ(1) + 0.1 * (op.SigmaX(0) + op.SigmaX(1))
        h._compile()

    def test_fermionic_operators(self):
        l = 2
        num_samples = 2 ** 7
        rbm = nets.CpxRBM(numHidden=2, bias=True)
        psi = NQS(rbm, l, num_samples)

        sampler = jVMC_exp.sampler.ExactSampler(psi)

        def commutator(i, j):
            return op.Creation(j) * op.Annihilation(i) + op.Annihilation(i) * op.Creation(j)

        observalbes_dict = {
            '0_0': commutator(0, 0),
            '1_1': commutator(1, 1),
            '0_1': commutator(0, 1),
            '1_0': commutator(1, 0)
        }
        out_dict = jVMC_exp.util.measure(observalbes_dict, psi, sampler)

        self.assertTrue(
            jnp.allclose(
                jnp.array([out_dict["0_0"]['mean'], out_dict["1_1"]['mean'], out_dict["0_1"]['mean'], out_dict["1_0"]['mean']]),
                jnp.array([1., 1., 0., 0.]),
                rtol=1e-15
            )
        )
        
        self.assertTrue(
            jnp.allclose(
                jnp.array([out_dict["0_0"]['variance'], out_dict["1_1"]['variance'], out_dict["0_1"]['variance'], out_dict["1_0"]['variance']]),
                jnp.array([0., 0., 0., 0.]),
                rtol=1e-15
            )
        )

    def test_fermionic_operator(self):
        t = - 1.0               # hopping
        mu = - 2.0              # chemical potential
        V = 4.0                 # interaction
        flavour = 2             # number of flavours
        flavourL = flavour * L  # number of spins times sites

        h = 0
        # impurity definitions
        site1UP = 0
        site1DO = flavourL - 1
        # loop over the 1d lattice
        for i in range(0, flavourL // flavour):
            # interaction
            h += V * op.Number(site1UP + i) * op.Number(site1DO - i)
            # chemical potential
            h += mu * (op.Number(site1UP + i) + op.Number(site1DO - i))
            if i == flavourL // flavour-1:
                continue
            # up chain hopping
            h += t * (op.Creation(site1UP + i + 1) * op.Annihilation(site1UP + i))
            h += t * (op.Creation(site1UP + i) * op.Annihilation(site1UP + i + 1))
            # down chain hopping
            h += t * (op.Creation(site1DO - i - 1) * op.Annihilation(site1DO - i))
            h += t * (op.Creation(site1DO - i) * op.Annihilation(site1DO - i - 1))

        chi_model = Target(L=flavourL, d=2)
        chi = NQS(chi_model, flavourL, NUM_SAMPLES)
        chi.parameters = np.loadtxt("data_ref/fermion_ref.txt", dtype=np.complex128)
        chi_sampler = jVMC_exp.sampler.ExactSampler(chi)
        s, logPsi, p = chi_sampler.sample()
        Oloc = jVMC_exp.stats.SampledObs(h.get_O_loc(s, chi, logPsiS=logPsi), p)

        self.assertTrue(jnp.allclose(Oloc.mean, -9.95314531))

if __name__ == "__main__":
    unittest.main()
