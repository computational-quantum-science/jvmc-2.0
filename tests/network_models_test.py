import unittest

import jax
import jax.numpy as jnp

import jVMC_exp.nets as nets
from jVMC_exp.symmetry.lattice_symetries import square_translation_symmetry
from jVMC_exp.vqs import NQS


def _param_leaves(variables):
    return [x for x in jax.tree_util.tree_leaves(variables["params"]) if hasattr(x, "dtype")]


class TestImportedNetworkModels(unittest.TestCase):

    def test_viteritti_vit_initializes_with_param_dtype(self):
        net = nets.LogViterittiSpatialViTJVMC(
            Lx=2,
            Ly=2,
            patch_size=1,
            layers=1,
            embed_dim=4,
            heads=1,
            param_dtype=jnp.float32,
            compute_dtype=jnp.float32,
        )
        s = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)
        variables = net.init(jax.random.PRNGKey(0), s)
        out = net.apply(variables, s)
        self.assertTrue(jnp.isfinite(jnp.real(out)))
        self.assertTrue(all(leaf.dtype == jnp.float32 for leaf in _param_leaves(variables)))

    def test_hidden_pfaffian_vit_initializes_with_param_dtype(self):
        symmetry = square_translation_symmetry(2, 2, "spinful_fermion")
        net = nets.HiddenPfaffianViT(
            Lx=2,
            Ly=2,
            symmetry=symmetry,
            patch_size=1,
            layers=1,
            embed_dim=4,
            heads=1,
            particles_per_spin=1,
            param_dtype=jnp.float32,
            compute_dtype=jnp.float32,
        )
        s = jnp.asarray([3, 0, 0, 0], dtype=jnp.int32)
        variables = net.init(jax.random.PRNGKey(1), s)
        out = net.apply(variables, s)
        self.assertTrue(jnp.isfinite(jnp.real(out)))
        self.assertTrue(all(leaf.dtype == jnp.float32 for leaf in _param_leaves(variables)))

    def test_rwkv_initializes_samples_and_uses_param_dtype(self):
        net = nets.RWKVCPM(
            L=4,
            LHilDim=2,
            patch_size=1,
            hidden_size=4,
            num_heads=1,
            num_layers=1,
            embedding_size=4,
            param_dtype=jnp.float32,
            compute_dtype=jnp.float32,
        )
        s = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)
        variables = net.init(jax.random.PRNGKey(2), s)
        out = net.apply(variables, s)
        sample = net.apply(variables, jax.random.PRNGKey(3), method=net.sample)
        self.assertTrue(jnp.isfinite(jnp.real(out)))
        self.assertEqual(sample.shape, (4,))
        self.assertTrue(jnp.all((sample == 0) | (sample == 1)))
        self.assertTrue(all(leaf.dtype == jnp.float32 for leaf in _param_leaves(variables)))

    def test_particle_conserving_rwkv_wrapper_initializes(self):
        base_net = nets.RWKVCPM(
            L=4,
            LHilDim=2,
            patch_size=1,
            hidden_size=4,
            num_heads=1,
            num_layers=1,
            embedding_size=4,
            param_dtype=jnp.float32,
            compute_dtype=jnp.float32,
        )
        net = nets.ParticleConservingAutoregressive(base_net, Q=2)
        s = jnp.asarray([1, 1, 0, 0], dtype=jnp.int32)
        variables = net.init(jax.random.PRNGKey(12), s)
        out = net.apply(variables, s)
        sample = net.apply(variables, jax.random.PRNGKey(13), method=net.sample)
        self.assertTrue(jnp.isfinite(jnp.real(out)))
        self.assertEqual(sample.shape, (4,))
        self.assertEqual(int(jnp.sum(sample)), 2)

    def test_particle_conserving_rwkv_fp16_invalid_state_has_zero_amplitude(self):
        base_net = nets.RWKVCPM(
            L=4,
            LHilDim=2,
            patch_size=1,
            hidden_size=4,
            num_heads=1,
            num_layers=1,
            embedding_size=4,
            param_dtype=jnp.float16,
            compute_dtype=jnp.float16,
        )
        net = nets.ParticleConservingAutoregressive(base_net, Q=2)
        valid = jnp.asarray([1, 1, 0, 0], dtype=jnp.int32)
        invalid = jnp.asarray([1, 0, 0, 0], dtype=jnp.int32)
        variables = net.init(jax.random.PRNGKey(15), valid)
        valid_out = net.apply(variables, valid)
        invalid_out = net.apply(variables, invalid)
        self.assertTrue(jnp.isfinite(jnp.real(valid_out)))
        self.assertTrue(jnp.isfinite(jnp.imag(valid_out)))
        self.assertTrue(jnp.isneginf(jnp.real(invalid_out)))
        self.assertTrue(jnp.isfinite(jnp.imag(invalid_out)))

    def test_rwkv2d_initializes_samples_and_uses_param_dtype(self):
        net = nets.LogRWKV2DAutoregressiveJVMC(
            L=4,
            LHilDim=2,
            patch_size=1,
            grid_Lx=2,
            grid_Ly=2,
            hidden_size=4,
            num_heads=1,
            num_layers=1,
            embedding_size=4,
            flag_phase=True,
            param_dtype=jnp.float32,
            compute_dtype=jnp.float32,
        )
        s = jnp.asarray([0, 1, 0, 1], dtype=jnp.int32)
        variables = net.init(jax.random.PRNGKey(4), s)
        out = net.apply(variables, s)
        sample = net.apply(variables, jax.random.PRNGKey(5), method=net.sample)
        self.assertTrue(jnp.isfinite(jnp.real(out)))
        self.assertEqual(sample.shape, (4,))
        self.assertTrue(jnp.all((sample == 0) | (sample == 1)))
        self.assertTrue(all(leaf.dtype == jnp.float32 for leaf in _param_leaves(variables)))

    def test_rwkv2d_fp16_fixed_sector_invalid_state_has_zero_amplitude(self):
        net = nets.LogRWKV2DAutoregressiveJVMC(
            L=4,
            LHilDim=2,
            patch_size=1,
            grid_Lx=2,
            grid_Ly=2,
            hidden_size=4,
            num_heads=1,
            num_layers=1,
            embedding_size=4,
            fixed_n_up=2,
            flag_phase=True,
            param_dtype=jnp.float16,
            compute_dtype=jnp.float16,
        )
        valid = jnp.asarray([0, 0, 1, 1], dtype=jnp.int32)
        invalid = jnp.asarray([0, 0, 0, 1], dtype=jnp.int32)
        variables = net.init(jax.random.PRNGKey(14), valid)
        valid_out = net.apply(variables, valid)
        invalid_out = net.apply(variables, invalid)
        self.assertTrue(jnp.isfinite(jnp.real(valid_out)))
        self.assertTrue(jnp.isfinite(jnp.imag(valid_out)))
        self.assertTrue(jnp.isneginf(jnp.real(invalid_out)))
        self.assertTrue(jnp.isfinite(jnp.imag(invalid_out)))
        self.assertTrue(all(leaf.dtype == jnp.float16 for leaf in _param_leaves(variables)))

    def test_rwkv_generators_work_through_nqs_sampling(self):
        net_1d = nets.RWKVCPM(
            L=4,
            LHilDim=2,
            patch_size=1,
            hidden_size=4,
            num_heads=1,
            num_layers=1,
            embedding_size=4,
            param_dtype=jnp.float32,
            compute_dtype=jnp.float32,
        )
        psi_1d = NQS(net_1d, 4, batchSize=4, seed=6)
        samples_1d = psi_1d.sample(4, jax.random.PRNGKey(7))
        self.assertEqual(samples_1d.shape, (4, 4))

        net_2d = nets.LogRWKV2DAutoregressiveJVMC(
            L=4,
            LHilDim=2,
            patch_size=1,
            grid_Lx=2,
            grid_Ly=2,
            hidden_size=4,
            num_heads=1,
            num_layers=1,
            embedding_size=4,
            param_dtype=jnp.float32,
            compute_dtype=jnp.float32,
        )
        psi_2d = NQS(net_2d, (2, 2), batchSize=4, seed=8)
        samples_2d = psi_2d.sample(4, jax.random.PRNGKey(9))
        self.assertEqual(samples_2d.shape, (4, 4))

    def test_imported_models_work_through_nqs_evaluation(self):
        symmetry = square_translation_symmetry(2, 2, "spinful_fermion")
        cases = [
            (
                nets.LogViterittiSpatialViTJVMC(
                    Lx=2,
                    Ly=2,
                    patch_size=1,
                    layers=1,
                    embed_dim=4,
                    heads=1,
                    param_dtype=jnp.float32,
                    compute_dtype=jnp.float32,
                ),
                (2, 2),
                jnp.asarray([[[0, 1], [0, 1]]], dtype=jnp.int32),
            ),
            (
                nets.HiddenPfaffianViT(
                    Lx=2,
                    Ly=2,
                    symmetry=symmetry,
                    patch_size=1,
                    layers=1,
                    embed_dim=4,
                    heads=1,
                    particles_per_spin=1,
                    param_dtype=jnp.float32,
                    compute_dtype=jnp.float32,
                ),
                (2, 2),
                jnp.asarray([[[3, 0], [0, 0]]], dtype=jnp.int32),
            ),
            (
                nets.RWKVCPM(
                    L=4,
                    LHilDim=2,
                    patch_size=1,
                    hidden_size=4,
                    num_heads=1,
                    num_layers=1,
                    embedding_size=4,
                    param_dtype=jnp.float32,
                    compute_dtype=jnp.float32,
                ),
                4,
                jnp.asarray([[0, 1, 0, 1]], dtype=jnp.int32),
            ),
            (
                nets.LogRWKV2DAutoregressiveJVMC(
                    L=4,
                    LHilDim=2,
                    patch_size=1,
                    grid_Lx=2,
                    grid_Ly=2,
                    hidden_size=4,
                    num_heads=1,
                    num_layers=1,
                    embedding_size=4,
                    param_dtype=jnp.float32,
                    compute_dtype=jnp.float32,
                ),
                (2, 2),
                jnp.asarray([[[0, 1], [0, 1]]], dtype=jnp.int32),
            ),
        ]
        for seed, (net, sample_shape, samples) in enumerate(cases, start=10):
            psi = NQS(net, sample_shape, batchSize=4, seed=seed)
            out = psi(samples)
            self.assertTrue(jnp.all(jnp.isfinite(jnp.real(out))))


if __name__ == "__main__":
    unittest.main()
