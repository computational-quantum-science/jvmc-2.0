import unittest
import jax
import jax.numpy as jnp
from math import isclose
import flax.linen as nn

import jVMC
import jVMC.nets as nets
from jVMC.vqs import NQS
from jVMC.global_defs import DT_SAMPLES
from jVMC.sharding_config import MESH, shard_map, DEVICE_SPEC
import jVMC.global_defs as global_defs
import jVMC.nets.activation_functions as act_funs

def _general_test_grad(model, num_samples, L, test_class: unittest.TestCase):
    psi = NQS(model, L, num_samples)

    s = jnp.zeros((num_samples, L), dtype=DT_SAMPLES)
    s = s.at[0, 1].set(1)
    s = s.at[2, 2].set(1)

    psi0 = psi(s)
    G = psi.gradients(s)
    delta = 1e-6
    params = psi.parameters
    for j in range(G.shape[-1]):
        u = jnp.zeros(G.shape[-1], dtype=global_defs.DT_PARAMS_REAL).at[j].set(1)
        psi.parameters = delta * u + psi.parameters_flat
        psi1 = psi(s)
        psi.parameters = params

        # Finite difference gradients
        Gfd = (psi1 - psi0) / delta
        with test_class.subTest(i=j):
            test_class.assertTrue(jnp.max(jnp.abs(Gfd - G[..., j])) < 1e-2)

class CpxRBM_nonHolomorphic(nn.Module):
    """
    Restricted Boltzmann machine with complex parameters.

    Initialization arguments:
        * ``s``: Computational basis configuration.
        * ``numHidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.

    """
    numHidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):

        layer = nn.Dense(
            self.numHidden, use_bias=self.bias,
            **jVMC.nets.initializers.init_fn_args(
                kernel_init=jVMC.nets.initializers.cplx_init,
                bias_init=jax.nn.initializers.zeros,
                dtype=global_defs.DT_PARAMS_CPX
            )
        )

        out = layer(2 * s.ravel() - 1)
        out = out + jnp.real(out) * 1e-2
        return jnp.sum(act_funs.log_cosh(out))

class Simple_nonHolomorphic(nn.Module):
    """
    Restricted Boltzmann machine with complex parameters.

    Initialization arguments:
        * ``s``: Computational basis configuration.
        * ``numHidden``: Number of hidden units.
        * ``bias``: ``Boolean`` indicating whether to use bias.

    """
    numHidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):
        z = self.param('z', jVMC.nets.initializers.cplx_init, (1,), global_defs.DT_PARAMS_CPX)
    
        return jnp.sum(jnp.conj(z))

class MatrixMultiplication_NonHolomorphic(nn.Module):
    holo: bool = False

    @nn.compact
    def __call__(self, s):
        layer1 = nn.Dense(1, use_bias=False, **jVMC.nets.initializers.init_fn_args(dtype=global_defs.DT_PARAMS_CPX))
        out = layer1(2 * s.ravel() - 1)
        if not self.holo:
            out = out + 1e-1 * jnp.real(out)
    
        return jnp.sum(out)
    
class TestGradients(unittest.TestCase):

    def test_automatic_holomorphicity_recognition(self):
        for k in range(10):
            L = 3
            num_samples = 4

            rbm = nets.CpxRBM(numHidden=2**k, bias=True)
            orbit = jVMC.util.symmetries.get_orbit_1D(L)
            net = nets.sym_wrapper.SymNet(net=rbm, orbit=orbit)
            psiC = NQS(net, L, num_samples)

            self.assertTrue(psiC.holomorphic)

    def test_gradients_cpx(self):
        L = 3
        num_samples = 4

        rbm = nets.CpxRBM(numHidden=2, bias=True)
        orbit = jVMC.util.symmetries.get_orbit_1D(L)
        model = nets.sym_wrapper.SymNet(net=rbm, orbit=orbit)

        _general_test_grad(model, num_samples, L, self)

    def test_gradients_real(self):
        L = 3
        num_samples = 4

        rbmModel1 = nets.RBM(numHidden=2, bias=True)
        rbmModel2 = nets.RBM(numHidden=3, bias=True)

        _general_test_grad((rbmModel1, rbmModel2), num_samples, L, self)

    def test_gradients_nonholomorphic(self):
        L = 3
        num_samples = 4
        model = nets.RNN1DGeneral(L=L)

        _general_test_grad(model, num_samples, L, self)
    
    def test_gradients_complex_nonholomorphic(self):
        L = 3
        num_samples = 4
        model = Simple_nonHolomorphic()

        _general_test_grad(model, num_samples, L, self)

        model = CpxRBM_nonHolomorphic()

        _general_test_grad(model, num_samples, L, self)

        model = MatrixMultiplication_NonHolomorphic(holo=False)

        psi = NQS(model, 1, num_samples)
        s= jnp.zeros((num_samples, 1), dtype=DT_SAMPLES)
        G = psi.gradients(s)
        ref = jnp.array([-1.1+0.j, -1.1+0.j, -1.1+0.j, -1.1+0.j, -0.-1.j, -0.-1.j, -0.-1.j, -0.-1.j])
        self.assertTrue(jnp.allclose(G.T.ravel(), ref))

    def test_gradient_dict(self):
        L = 3
        num_samples = 4
        net = jVMC.nets.CpxRBM(numHidden=8, bias=False)

        psi = jVMC.vqs.NQS(net, L, num_samples, seed=1234)
        s = jnp.zeros((num_samples, L))
        g1 = psi.gradients(s)
        g2 = psi.gradients_dict(s)["Dense_0"]["kernel"]

        self.assertTrue(isclose(jnp.linalg.norm(g1 - g2), 0.0))

class TestEvaluation(unittest.TestCase):

    def test_evaluation_cpx(self):
        L = 3
        num_samples = 4
        model = nets.CpxRBM(numHidden=2, bias=True)
        psiC = NQS(model, L, num_samples)

        s = jnp.zeros((num_samples, L), dtype=DT_SAMPLES)
        s = s.at[0, 1].set(1)
        s = s.at[2, 2].set(1)

        cpxCoeffs = psiC(s)
        f, p = psiC.get_sampler_net()
        realCoeffs = jax.jit(shard_map(jax.vmap(lambda y: f(p, y)), MESH, (DEVICE_SPEC,), (DEVICE_SPEC)))(s)

        self.assertTrue(jnp.linalg.norm(jnp.real(cpxCoeffs) - realCoeffs) < 1e-6)

if __name__ == "__main__":
    unittest.main()
