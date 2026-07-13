import unittest
import jax
import jax.numpy as jnp
import flax.linen as nn

import jVMC_exp
import jVMC_exp.global_defs as global_defs
import jVMC_exp.nets as nets
import jVMC_exp.nets.activation_functions as act_funs
from jVMC_exp.vqs import NQS
import jVMC_exp.operator.discrete as op
import jVMC_exp.sampler as sampler


class CpxRBMNonHolomorphic(nn.Module):
    numHidden: int = 2
    bias: bool = False

    @nn.compact
    def __call__(self, s):
        layer = nn.Dense(
            self.numHidden,
            use_bias=self.bias,
            **jVMC_exp.nets.initializers.init_fn_args(
                kernel_init=jVMC_exp.nets.initializers.cplx_init,
                bias_init=jax.nn.initializers.zeros,
                param_dtype=global_defs.DT_PARAMS_CPX,
            )
        )
        out = layer(2 * s.ravel() - 1)
        out = out + jnp.real(out) * 1e-2
        return jnp.sum(act_funs.log_cosh(out))


class TestMinSR(unittest.TestCase):
    def test_full_batched_matches_dense_update(self):
        L = 3
        batch_size = 4
        rbm = nets.CpxRBM(numHidden=2, bias=True)
        psi = NQS(rbm, L, batch_size, seed=1234)

        H = 0
        for l in range(L):
            H += -1.0 * op.SigmaZ(l) * op.SigmaZ((l + 1) % L) - 0.7 * op.SigmaX(l)

        exact_sampler = sampler.ExactSampler(psi)
        loss_function = jVMC_exp.objective_function.Observable(H)

        self.assertFalse(jVMC_exp.optimizer.MinSR(exact_sampler, psi).full_batched)

        opt_dense = jVMC_exp.optimizer.MinSR(
            exact_sampler, psi, pinv_tol=1e-10, diagonalShift=1e-3, full_batched=False
        )
        opt_full_batched = jVMC_exp.optimizer.MinSR(
            exact_sampler, psi, pinv_tol=1e-10, diagonalShift=1e-3, full_batched=True
        )

        dense_update = opt_dense(psi.parameters_flat, 0, objective_function=loss_function)
        full_batched_update = opt_full_batched(psi.parameters_flat, 0, objective_function=loss_function)

        self.assertTrue(jnp.allclose(full_batched_update, dense_update, rtol=1e-10, atol=1e-10))

    def test_full_batched_matches_dense_for_complex_nonholomorphic_parameters(self):
        L = 3
        psi = NQS(CpxRBMNonHolomorphic(), L, 4, seed=1234)
        exact_sampler = sampler.ExactSampler(psi)

        H = 0
        for l in range(L):
            H += -1.0 * op.SigmaZ(l) * op.SigmaZ((l + 1) % L) - 0.7 * op.SigmaX(l)

        loss_function = jVMC_exp.objective_function.Observable(H)
        opt_dense = jVMC_exp.optimizer.MinSR(
            exact_sampler, psi, pinv_tol=1e-10, diagonalShift=1e-3, full_batched=False
        )
        opt_full_batched = jVMC_exp.optimizer.MinSR(
            exact_sampler, psi, pinv_tol=1e-10, diagonalShift=1e-3, full_batched=True
        )

        dense_update = opt_dense(psi.parameters_flat, 0, objective_function=loss_function)
        full_batched_update = opt_full_batched(psi.parameters_flat, 0, objective_function=loss_function)

        self.assertTrue(jnp.allclose(full_batched_update, dense_update, rtol=1e-10, atol=1e-10))

class TestGsSearch(unittest.TestCase):
    def test_gs_search_cpx(self):
        L = 4
        J = -1.0
        hxs = [-1.3, -0.3]
        exEs = [-6.10160339, -4.09296160]
        
        batch_size = int(2 ** L)
        learning_rate = 1e-2
        num_steps = 300

        for hx, exE in zip(hxs, exEs):
            # Set up variational wave function
            rbm = nets.CpxRBM(numHidden=3, bias=False)
            psi = NQS(rbm, L, batch_size, seed=1234)

            # Set up hamiltonian for ground state search
            H = 0
            for l in range(L):
                H += J * op.SigmaZ(l) * op.SigmaZ((l + 1) % L) + hx * op.SigmaX(l)

            # Set up exact sampler
            exact_sampler = sampler.ExactSampler(psi)
            
            loss_function = jVMC_exp.objective_function.Observable(H)
            stepper = jVMC_exp.stepper.Euler(timeStep=learning_rate)
            opt = jVMC_exp.optimizer.MinSR(exact_sampler, psi, pinv_tol=1e-6, diagonalShift=1e-3)

            opt.ground_state_search(num_steps, loss_function, stepper)

            E = exact_sampler(H)
            print(jnp.abs((E.mean.item() - exE) / exE))
            self.assertTrue(jnp.max(jnp.abs((E.mean.item() - exE) / exE)) < 1e-3)

if __name__ == "__main__":
    unittest.main()
