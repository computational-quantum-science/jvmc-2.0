import unittest
import jax.random as random
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

import jVMC_exp.operator.discrete as op
from jVMC_exp import global_defs

L = 4
LDIM = 2
KEY = random.PRNGKey(3)
NUM_SAMPLES = 2 ** 6

def _parity_between(state, i, j):
    if i == j:
        return 1
    if i < j:
        mask = ((1 << j) - 1) ^ ((1 << (i + 1)) - 1)
    else:
        mask = ((1 << i) - 1) ^ ((1 << (j + 1)) - 1)
    return 1 - 2 * ((state & mask).bit_count() & 1)

def _hop_reference(H, state, src, dst, amplitude):
    if ((state >> src) & 1) == 0 or ((state >> dst) & 1) == 1:
        return
    sign = _parity_between(state, dst, src)
    new_state = state ^ (1 << src) ^ (1 << dst)
    H[new_state, state] += amplitude * sign

def _spinless_orbital(site, L_chain):
    return L_chain - 1 - site

def _spinful_orbital(site, spin, L_chain):
    if spin == "down":
        return 2 * (L_chain - 1 - site)
    if spin == "up":
        return 2 * (L_chain - 1 - site) + 1
    raise ValueError(f"Unknown spin '{spin}'")

def _spinless_config_from_state(state, L_chain):
    return [((state >> _spinless_orbital(site, L_chain)) & 1) for site in range(L_chain)]

def _spinless_state_from_config(config):
    L_chain = len(config)
    state = 0
    for site, occ in enumerate(config):
        state |= int(occ) << _spinless_orbital(site, L_chain)
    return state

def _spinful_config_from_state(state, L_chain):
    config = []
    for site in range(L_chain):
        up = (state >> _spinful_orbital(site, "up", L_chain)) & 1
        down = (state >> _spinful_orbital(site, "down", L_chain)) & 1
        config.append(up + 2 * down)
    return config

def _spinful_state_from_config(config):
    L_chain = len(config)
    state = 0
    for site, local_state in enumerate(config):
        state |= (int(local_state) & 1) << _spinful_orbital(site, "up", L_chain)
        state |= ((int(local_state) >> 1) & 1) << _spinful_orbital(site, "down", L_chain)
    return state

def _dense_matrix_from_operator(operator, configs, state_from_config):
    configs = np.asarray(configs, dtype=np.int32)
    sp, matEls = operator.get_conn_elements(jnp.asarray(configs, dtype=global_defs.DT_SAMPLES), configs.shape[0])
    sp = np.asarray(sp)
    matEls = np.asarray(matEls)

    dim = configs.shape[0]
    H = np.zeros((dim, dim), dtype=np.complex128)
    for col, _ in enumerate(configs):
        for row_config, mat_el in zip(sp[col], matEls[col]):
            H[state_from_config(row_config), col] += mat_el
    return H

def _reference_spinful_hubbard(L_chain, t, U, mu):
    n_orbitals = 2 * L_chain
    dim = 1 << n_orbitals
    H = np.zeros((dim, dim), dtype=np.float64)

    for state in range(dim):
        for site in range(L_chain):
            nbr = (site + 1) % L_chain
            for spin in ("down", "up"):
                site_orb = _spinful_orbital(site, spin, L_chain)
                nbr_orb = _spinful_orbital(nbr, spin, L_chain)
                _hop_reference(H, state, site_orb, nbr_orb, t)
                _hop_reference(H, state, nbr_orb, site_orb, t)

            up_occ = (state >> _spinful_orbital(site, "up", L_chain)) & 1
            down_occ = (state >> _spinful_orbital(site, "down", L_chain)) & 1
            H[state, state] += U * up_occ * down_occ
            H[state, state] -= mu * (up_occ + down_occ)

    return H

def _reference_spinless_tv(L_chain, t, V, mu):
    dim = 1 << L_chain
    H = np.zeros((dim, dim), dtype=np.float64)

    for state in range(dim):
        for site in range(L_chain):
            nbr = (site + 1) % L_chain
            site_orb = _spinless_orbital(site, L_chain)
            nbr_orb = _spinless_orbital(nbr, L_chain)
            _hop_reference(H, state, site_orb, nbr_orb, t)
            _hop_reference(H, state, nbr_orb, site_orb, t)

            site_occ = (state >> site_orb) & 1
            nbr_occ = (state >> nbr_orb) & 1
            H[state, state] += V * site_occ * nbr_occ
            H[state, state] -= mu * site_occ

    return H

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
        s = random.randint(KEY, (NUM_SAMPLES, L), 0, 2, dtype=global_defs.DT_SAMPLES)

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
        s = random.randint(KEY, (NUM_SAMPLES, L), 0, 2, dtype=global_defs.DT_SAMPLES)
        
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
        s = random.randint(KEY, (NUM_SAMPLES, L, L), 0, 2, dtype=global_defs.DT_SAMPLES)

        h = 0.3 * op.SigmaPlus(0) + 1.1 * op.SigmaPlus(1) + 0.15 * op.SigmaPlus(4)

        sp, matEls = h.get_conn_elements(s, NUM_SAMPLES)

        self.assertTrue(sp.shape[2:] == (L, L))

    def test_td_prefactor(self):
        h = op.SigmaZ(0) + op.SigmaZ(1) + 0.1 * (op.SigmaX(0) + op.SigmaX(1))
        h._compile()

    ########################################################################
    # spinful fermionic operators
    ########################################################################
    def test_spinful_fermionic_onsite_anticommutators(self):
        for spin in ["up", "down"]:
            anti = op.Creation(0, spin) * op.Annihilation(0, spin) + op.Annihilation(0, spin) * op.Creation(0, spin)
            sp, matEls = anti.get_conn_elements(jnp.array([[0], [1], [2], [3]], dtype=global_defs.DT_SAMPLES), 4)
            loc = jnp.sum(matEls, axis=1)
            self.assertTrue(jnp.allclose(loc, jnp.ones(4)))

    def test_spinful_fermionic_onsite_cross_anticommutator(self):
        s = jnp.array([[0], [1], [2], [3]], dtype=global_defs.DT_SAMPLES)

        anti = (
            op.Creation(0, "up") * op.Creation(0, "down")
            + op.Creation(0, "down") * op.Creation(0, "up")
        )

        sp, matEls = anti.get_conn_elements(s, 4)
        loc = jnp.sum(matEls, axis=1)

        self.assertTrue(jnp.allclose(loc, jnp.zeros(4)))

    def test_spinful_fermionic_intersite_anticommutator(self):
        s = jnp.array(
            [
                [0, 0],
                [1, 0],
                [0, 2],
                [3, 1],
            ],
            dtype=global_defs.DT_SAMPLES,
        )

        anti = (
            op.Creation(0, "up") * op.Creation(1, "down")
            + op.Creation(1, "down") * op.Creation(0, "up")
        )

        sp, matEls = anti.get_conn_elements(s, s.shape[0])
        loc = jnp.sum(matEls, axis=1)

        self.assertTrue(jnp.allclose(loc, jnp.zeros(s.shape[0])))

    def test_spinful_hubbard_dense_matrix(self):
        L_chain = 4
        t = 1.0
        U = 4.0
        mu = U / 2

        h = 0
        for site in range(L_chain):
            nbr = (site + 1) % L_chain
            for spin in ("down", "up"):
                h += t * op.Creation(nbr, spin) * op.Annihilation(site, spin)
                h += t * op.Creation(site, spin) * op.Annihilation(nbr, spin)
            h += U * op.Number(site, "up") * op.Number(site, "down")
            h += -mu * (op.Number(site, "up") + op.Number(site, "down"))

        dim = 4 ** L_chain
        configs = [_spinful_config_from_state(state, L_chain) for state in range(dim)]
        H_jvmc = _dense_matrix_from_operator(h, configs, _spinful_state_from_config)
        H_ref = _reference_spinful_hubbard(L_chain, t, U, mu)
        self.assertTrue(np.allclose(H_jvmc, H_ref))

    ########################################################################
    # spinless fermionic operators
    ########################################################################
    def test_spinless_fermionic_onsite_anticommutators(self):
        anti = op.Creation(0) * op.Annihilation(0) + op.Annihilation(0) * op.Creation(0)
        sp, matEls = anti.get_conn_elements(jnp.array([[0], [1]], dtype=global_defs.DT_SAMPLES), 4)
        loc = jnp.sum(matEls, axis=1)
        self.assertTrue(jnp.allclose(loc, jnp.ones(2)))

    def test_spinless_fermionic_intersite_anticommutator(self):
        s = jnp.array(
            [
                [0, 0],
                [1, 0],
            ],
            dtype=global_defs.DT_SAMPLES,
        )

        anti = (
            op.Creation(0) * op.Creation(1)
            + op.Creation(1) * op.Creation(0)
        )

        _, matEls = anti.get_conn_elements(s, s.shape[0])
        loc = jnp.sum(matEls, axis=1)

        self.assertTrue(jnp.allclose(loc, jnp.zeros(s.shape[0])))

    def test_spinless_tv_dense_matrix(self):
        L_chain = 4
        t = 1.0
        V = 4.0
        mu = V / 2

        h = 0
        for site in range(L_chain):
            nbr = (site + 1) % L_chain
            h += t * op.Creation(nbr) * op.Annihilation(site)
            h += t * op.Creation(site) * op.Annihilation(nbr)
            h += V * op.Number(site) * op.Number(nbr)
            h += -mu * op.Number(site)

        dim = 2 ** L_chain
        configs = [_spinless_config_from_state(state, L_chain) for state in range(dim)]
        H_jvmc = _dense_matrix_from_operator(h, configs, _spinless_state_from_config)
        H_ref = _reference_spinless_tv(L_chain, t, V, mu)

        self.assertTrue(np.allclose(H_jvmc, H_ref))

if __name__ == "__main__":
    unittest.main()
