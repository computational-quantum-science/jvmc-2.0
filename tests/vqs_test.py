import unittest
import jax
import jax.numpy as jnp
from math import isclose
import flax.linen as nn

import jVMC_exp
import jVMC_exp.nets as nets
from jVMC_exp.vqs import NQS
from jVMC_exp.symmetry.lattice_symetries import square_translation_symmetry
from jVMC_exp.global_defs import DT_SAMPLES
from jVMC_exp.sharding_config import MESH, DEVICE_SPEC
import jVMC_exp.global_defs as global_defs
import jVMC_exp.nets.activation_functions as act_funs
from jVMC_exp.precision import set_precision_mode

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
            **jVMC_exp.nets.initializers.init_fn_args(
                kernel_init=jVMC_exp.nets.initializers.cplx_init,
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
        z = self.param('z', jVMC_exp.nets.initializers.cplx_init, (1,), global_defs.DT_PARAMS_CPX)
    
        return jnp.sum(jnp.conj(z))

class MatrixMultiplication_NonHolomorphic(nn.Module):
    holo: bool = False

    @nn.compact
    def __call__(self, s):
        layer1 = nn.Dense(1, use_bias=False, **jVMC_exp.nets.initializers.init_fn_args(dtype=global_defs.DT_PARAMS_CPX))
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
            orbit = square_translation_symmetry(L, 1, "spin")
            psiC = NQS(rbm, L, num_samples, orbit=orbit)

            self.assertTrue(psiC.holomorphic)

    def test_gradients_cpx(self):
        L = 3
        num_samples = 4

        rbm = nets.CpxRBM(numHidden=2, bias=True)
        orbit = square_translation_symmetry(L, 1, "spin")
        model = orbit * rbm

        _general_test_grad(model, num_samples, L, self)

    def test_gradients_real(self):
        L = 3
        num_samples = 4

        rbmModel1 = nets.RBM(numHidden=2, bias=True)
        rbmModel2 = nets.RBM(numHidden=3, bias=True)

        _general_test_grad((rbmModel1, rbmModel2), num_samples, L, self)

    # TODO: RNN is no longer working
    # def test_gradients_nonholomorphic(self): 
    #     L = 3
    #     num_samples = 4
    #     model = nets.RNN1DGeneral(L=L)

    #     _general_test_grad(model, num_samples, L, self)
    
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
        net = jVMC_exp.nets.CpxRBM(numHidden=8, bias=False)

        psi = jVMC_exp.vqs.NQS(net, L, num_samples, seed=1234)
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
        realCoeffs = jax.jit(
            jax.shard_map(
                jax.vmap(lambda y: f(p, y)), 
                mesh=MESH, 
                in_specs=(DEVICE_SPEC,), 
                out_specs=(DEVICE_SPEC)
            )
        )(s)

        self.assertTrue(jnp.linalg.norm(jnp.real(cpxCoeffs) - realCoeffs) < 1e-6)

    def test_mixed_precision_nqs_boundary(self):
        old_dtypes = (
            global_defs.DT_PARAMS_REAL,
            global_defs.DT_PARAMS_CPX,
            global_defs.DT_OPERATORS_REAL,
            global_defs.DT_OPERATORS_CPX,
        )
        try:
            set_precision_mode("mixed_fp32")
            L = 3
            num_samples = 4
            model = nets.CpxRBM(numHidden=2, bias=True)
            psi = NQS(model, L, num_samples, seed=1234, mixed_precision=True)
            param_leaves = jax.tree_util.tree_leaves(psi.parameters["params"])
            self.assertTrue(all(p.dtype == jnp.complex64 for p in param_leaves))

            s = jnp.zeros((num_samples, L), dtype=DT_SAMPLES)
            coeffs = psi(s)
            gradients = psi.gradients(s)
            self.assertEqual(coeffs.dtype, jnp.complex128)
            self.assertEqual(gradients.dtype, jnp.complex128)
            self.assertEqual(psi.parameters_flat.dtype, jnp.float64)

            psi.parameters = psi.parameters_flat + jnp.asarray(1e-4, dtype=jnp.float64)
            param_leaves = jax.tree_util.tree_leaves(psi.parameters["params"])
            self.assertTrue(all(p.dtype == jnp.complex64 for p in param_leaves))

            ratio_psi = NQS(nets.CpxRBM_ratio(numHidden=2, bias=True), L, num_samples, seed=1234, mixed_precision=True)
            sp = s.at[:, 0].set(1)
            ratio = ratio_psi.call_ratio(s, sp)
            self.assertEqual(ratio.dtype, jnp.complex128)

            with self.assertRaises(ValueError):
                NQS(model, L, num_samples, seed=1234, mixed_precision=False)
        finally:
            (
                global_defs.DT_PARAMS_REAL,
                global_defs.DT_PARAMS_CPX,
                global_defs.DT_OPERATORS_REAL,
                global_defs.DT_OPERATORS_CPX,
            ) = old_dtypes

if __name__ == "__main__":
    unittest.main()
