"""Symmetry projectors for NQS orbit sums.

Conventions
-----------
Spinful fermion samples use local dimension 4 site states:
``0=empty``, ``1=up``, ``2=down``, and ``3=up+down``. Fermionic signs use the
jVMC_exp mode ordering ``(N-1, down), (N-1, up), ..., (0, down), (0, up)``.

The built-in two-dimensional site enumerations are ``site_order="direct"``
for row-major order and ``site_order="snake"`` for the column-snake order
``np.arange(Lx * Ly).reshape(Lx, Ly)`` with even columns reversed. Custom
enumerations use ``site_order="custom"`` and a ``site_enum`` array; pass
``site_enum_axes="yx"`` for arrays shaped ``(Ly, Lx)`` and
``site_enum_axes="xy"`` for arrays shaped ``(Lx, Ly)``.

Fermionic particle-hole uses the convention
``C c_i C^-1 = c_i^dagger`` and ``C c_i^dagger C^-1 = c_i`` in the same
jVMC_exp mode ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
import cmath
import math
from typing import Callable, Literal, Sequence
import warnings

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


ParticleType = Literal["spin", "boson", "bosons", "spinless_fermion", "spinful_fermion"]
BuiltinSymmetryAverage = Literal["exp", "log", "sep"]
SymmetryAverage = BuiltinSymmetryAverage | Callable
SiteEnumeration = Sequence[Sequence[int]] | jnp.ndarray | None
SiteOrder = Literal["direct", "snake", "custom"]
SiteEnumerationAxes = Literal["yx", "xy"]
_PARTICLE_TYPES = {"spin", "boson", "spinless_fermion", "spinful_fermion"}


def _normalize_particle_type(particle_type: ParticleType) -> str:
    particle_type = str(particle_type)
    if particle_type == "bosons":
        particle_type = "boson"
    if particle_type == "spins":
        particle_type = "spin"
    if particle_type == "spinless_fermions":
        particle_type = "spinless_fermion"
    if particle_type == "spinful_fermions":
        particle_type = "spinful_fermion"
    if particle_type not in _PARTICLE_TYPES:
        raise ValueError("particle_type must be 'spin', 'boson', 'spinless_fermion', or 'spinful_fermion'.")
    return particle_type


def _validate_lattice_shape(Lx: int, Ly: int) -> tuple[int, int]:
    Lx, Ly = int(Lx), int(Ly)
    if Lx <= 0 or Ly <= 0:
        raise ValueError("Lx and Ly must both be positive. Use Ly=1 for one-dimensional chains.")
    return Lx, Ly

def _site_enum_array(
    Lx: int,
    Ly: int,
    site_order: SiteOrder = "direct",
    site_enum: SiteEnumeration = None,
    site_enum_axes: SiteEnumerationAxes = "yx",
):
    if site_order not in ("direct", "snake", "custom"):
        raise ValueError("site_order must be 'direct', 'snake', or 'custom'.")
    if site_enum_axes not in ("yx", "xy"):
        raise ValueError("site_enum_axes must be 'yx' or 'xy'.")
    if site_order != "custom" and site_enum is not None:
        raise ValueError("site_enum is only accepted with site_order='custom'.")
    if site_order == "direct":
        enum = jnp.arange(Lx * Ly, dtype=jnp.int32).reshape(Ly, Lx)
    elif site_order == "snake":
        enum_xy = jnp.arange(Lx * Ly, dtype=jnp.int32).reshape(Lx, Ly)
        enum_xy = enum_xy.at[::2, :].set(enum_xy[::2, ::-1])
        enum = enum_xy.T
    else:
        if site_enum is None:
            raise ValueError("site_order='custom' requires site_enum.")
        enum = jnp.asarray(site_enum, dtype=jnp.int32)
        if site_enum_axes == "xy":
            if enum.shape != (Lx, Ly):
                raise ValueError("site_enum with site_enum_axes='xy' must have shape (Lx, Ly).")
            enum = enum.T
        else:
            if enum.shape != (Ly, Lx):
                raise ValueError("site_enum with site_enum_axes='yx' must have shape (Ly, Lx).")
    flat = jnp.ravel(enum)
    expected = jnp.arange(Lx * Ly, dtype=jnp.int32)
    if bool(jnp.any(jnp.sort(flat) != expected)):
        raise ValueError("site enumeration must contain each site index from 0 to Lx*Ly-1 exactly once.")
    return enum


def _site_enum_lookup(
    Lx: int,
    Ly: int,
    site_order: SiteOrder = "direct",
    site_enum: SiteEnumeration = None,
    site_enum_axes: SiteEnumerationAxes = "yx",
):
    enum = _site_enum_array(Lx, Ly, site_order, site_enum, site_enum_axes)
    coords = jnp.zeros((Lx * Ly, 2), dtype=jnp.int32)
    for y in range(Ly):
        for x in range(Lx):
            coords = coords.at[int(enum[y, x])].set(jnp.asarray([x, y], dtype=jnp.int32))
    return enum, coords


def _spin_index(spin: int | str) -> int:
    if isinstance(spin, str):
        spin_key = spin.lower()
        if spin_key in {"up", "u", "↑"}:
            return 0
        if spin_key in {"down", "dn", "d", "↓"}:
            return 1
        raise ValueError("spin must be 'up'/'down' or 0/1.")
    spin_idx = int(spin)
    if spin_idx not in (0, 1):
        raise ValueError("spin must be 'up'/'down' or 0/1.")
    return spin_idx


def split_spinful_site_state(s, n_sites: int):
    """Decode true ldim=4 site states into up/down occupation bits.

    Local values are ``0=empty``, ``1=up``, ``2=down``, and ``3=up+down``.
    The physical sample therefore has one integer per lattice site; fermionic
    signs expand this representation internally to the jVMC_exp ordering
    ``(N-1, down), (N-1, up), ..., (0, down), (0, up)``.
    """
    site_state = jnp.asarray(s, dtype=jnp.int32).reshape(int(n_sites)) # flatten and adjust type
    up = jnp.bitwise_and(site_state, 1)
    down = jnp.bitwise_and(jnp.right_shift(site_state, 1), 1)
    return up, down


def _spinful_local_occupancy(s, n_sites: int):
    up, down = split_spinful_site_state(s, n_sites)
    site_order = jnp.arange(int(n_sites) - 1, -1, -1)
    return jnp.stack([down[site_order], up[site_order]], axis=-1).reshape(2 * int(n_sites)) > 0


def _spinless_local_occupancy(s, n_sites: int):
    site_order = jnp.arange(int(n_sites) - 1, -1, -1)
    return jnp.asarray(s, dtype=jnp.int32).reshape(int(n_sites))[site_order] > 0


def _site_perm_to_spinless_orbital_perm(site_perm, n_sites: int):
    site_perm = jnp.asarray(site_perm, dtype=jnp.int32)
    out_sites = jnp.arange(int(n_sites) - 1, -1, -1)
    return int(n_sites) - 1 - site_perm[..., out_sites]


def _site_perm_to_spinful_orbital_perm(site_perm, n_sites: int):
    site_perm = jnp.asarray(site_perm, dtype=jnp.int32)
    out_sites = jnp.arange(int(n_sites) - 1, -1, -1)
    input_sites = site_perm[..., out_sites]
    down_pos = 2 * (int(n_sites) - 1 - input_sites)
    up_pos = down_pos + 1
    return jnp.stack([down_pos, up_pos], axis=-1).reshape(*site_perm.shape[:-1], 2 * int(n_sites))


def _spinful_site_sign_to_orbital_sign(site_sign, n_sites: int):
    site_sign = jnp.asarray(site_sign, dtype=jnp.int32)
    site_order = jnp.arange(int(n_sites) - 1, -1, -1)
    return jnp.repeat(site_sign[..., site_order], 2, axis=-1)


def _fermion_particle_hole_sign_from_occ(occ):
    r"""Fock basis sign for the fermionic particle-hole transform

    P_{ph} c_i P_{ph}^{-1} = c_i^\dagger
    P_{ph} c_i^\dagger P_{ph}^{-1} = c_i
    where P_{ph} acts on the state as 
    P_{ph} |0> = |full>
    and the full reference state
    |full> = c_0^\dagger c_1^\dagger ... c_{M-1}^\dagger |0>
    """
    occ = jnp.asarray(occ, dtype=jnp.int32)
    mode_idx = jnp.arange(occ.shape[-1], dtype=jnp.int32)
    exponent = jnp.sum(occ * mode_idx, axis=-1)
    return (1 - 2 * jnp.bitwise_and(exponent, 1)).astype(jnp.complex128)

def _avg_coefficients_exp(log_amplitudes, weights):
    log_amplitudes = jnp.asarray(log_amplitudes, dtype=jnp.complex128)
    weights = jnp.asarray(weights, dtype=jnp.complex128)
    shift = jnp.max(jnp.real(log_amplitudes))
    return shift + jnp.log(jnp.mean(weights * jnp.exp(log_amplitudes - shift)))


def _avg_coefficients_log(log_amplitudes, weights):
    del weights
    return jnp.mean(jnp.asarray(log_amplitudes, dtype=jnp.complex128))


def _avg_coefficients_sep(log_amplitudes, weights):
    del weights
    log_amplitudes = jnp.asarray(log_amplitudes, dtype=jnp.complex128)
    re = jnp.real(log_amplitudes)
    im = jnp.imag(log_amplitudes)
    return 0.5 * jnp.log(jnp.mean(jnp.exp(2 * re))) + 1j * jnp.angle(jnp.mean(jnp.exp(1j * im)))


def _average_coefficients(log_amplitudes, weights, average: SymmetryAverage):
    if average == "exp":
        return _avg_coefficients_exp(log_amplitudes, weights)
    if average == "log":
        return _avg_coefficients_log(log_amplitudes, weights)
    if average == "sep":
        return _avg_coefficients_sep(log_amplitudes, weights)
    if callable(average):
        return jnp.asarray(average(log_amplitudes, weights), dtype=jnp.complex128)
    raise ValueError("symmetry_average must be 'exp', 'log', 'sep', or a callable.")


def _has_autoregressive_sampler(module) -> bool:
    if bool(getattr(module, "sampler", False)):
        return True
    if callable(getattr(module, "sample", None)):
        return True
    wrapped = getattr(module, "net", None)
    return wrapped is not None and _has_autoregressive_sampler(wrapped)


def _array_tuple(x, dtype):
    return tuple(map(tuple, jnp.asarray(x, dtype=dtype).tolist()))

def _vector_tuple(x, dtype):
    return tuple(jnp.asarray(x, dtype=dtype).tolist())


def _deduplicate_symmetry_arrays(perm, character, local_map=None, local_phase=None, local_sign=None, perm_sign=None):
    n_original = int(perm.shape[0])
    local_sign = jnp.zeros((n_original,), dtype=jnp.int8) if local_sign is None else jnp.asarray(local_sign, dtype=jnp.int8)
    buckets = {}
    order = []
    for idx, row in enumerate(perm.tolist()):
        key = (
            tuple(row),
            None if perm_sign is None else tuple(perm_sign[idx].tolist()),
            None if local_map is None else tuple(local_map[idx].tolist()),
            None if local_phase is None else tuple(local_phase[idx].tolist()),
            int(local_sign[idx]),
        )
        if key not in buckets:
            buckets[key] = {"idx": idx, "character": 0.0 + 0.0j}
            order.append(key)
        buckets[key]["character"] += complex(character[idx])

    keep = jnp.asarray([buckets[key]["idx"] for key in order], dtype=jnp.int32)
    scale = len(order) / n_original
    character_unique = jnp.asarray([scale * buckets[key]["character"] for key in order], dtype=jnp.complex128)
    return keep, character_unique


@dataclass(frozen=True)
class SymmetryProjector:
    """Configuration-space symmetry projector compatible with fermionic signs.

    ``perm[g, i]`` is used as ``s_g = s[perm[g]]``. Spinful fermions use
    true local dimension 4 site states, with values ``0,1,2,3`` for
    empty/up/down/double occupancy. Fermionic signs expand this to the
    jVMC_exp ordering only inside the parity calculation.
    """

    Lx: int
    Ly: int
    particle_type: ParticleType
    perm: tuple[tuple[int, ...], ...]
    character: tuple[complex, ...]
    perm_sign: tuple[tuple[int, ...], ...] | None = None
    local_map: tuple[tuple[int, ...], ...] | None = None
    local_phase: tuple[tuple[complex, ...], ...] | None = None
    local_sign: tuple[int, ...] | None = None
    name: str = "custom"

    def __post_init__(self):
        particle_type = _normalize_particle_type(self.particle_type)
        perm_arr = jnp.asarray(self.perm, dtype=jnp.int32)
        character_arr = jnp.asarray(self.character, dtype=jnp.complex128)
        if perm_arr.ndim != 2:
            raise ValueError("perm must have shape (nsymm, nmodes).")
        if character_arr.shape != (perm_arr.shape[0],):
            raise ValueError("character must have shape (nsymm,).")
        if self.perm_sign is None:
            perm_sign_arr = jnp.ones_like(perm_arr, dtype=jnp.int8)
        else:
            perm_sign_arr = jnp.asarray(self.perm_sign, dtype=jnp.int8)
            if perm_sign_arr.shape != perm_arr.shape:
                raise ValueError("perm_sign must have the same shape as perm.")
        if self.local_map is None:
            local_map_arr = None
        else:
            local_map_arr = jnp.asarray(self.local_map, dtype=jnp.int32)
            if local_map_arr.ndim != 2 or local_map_arr.shape[0] != perm_arr.shape[0]:
                raise ValueError("local_map must have shape (nsymm, local_dim).")
        if self.local_phase is None:
            local_phase_arr = None
        else:
            local_phase_arr = jnp.asarray(self.local_phase, dtype=jnp.complex128)
            if local_phase_arr.ndim != 2 or local_phase_arr.shape[0] != perm_arr.shape[0]:
                raise ValueError("local_phase must have shape (nsymm, local_dim).")
            if local_map_arr is not None and local_phase_arr.shape[1] != local_map_arr.shape[1]:
                raise ValueError("local_phase and local_map must have the same local dimension.")
        if self.local_sign is None:
            local_sign_arr = jnp.zeros((perm_arr.shape[0],), dtype=jnp.int8)
        else:
            local_sign_arr = jnp.asarray(self.local_sign, dtype=jnp.int8)
            if local_sign_arr.shape != (perm_arr.shape[0],):
                raise ValueError("local_sign must have shape (nsymm,).")
            if bool(jnp.any((local_sign_arr != 0) & (local_sign_arr != 1))):
                raise ValueError("local_sign entries must be 0 or 1.")
        object.__setattr__(self, "perm", _array_tuple(perm_arr, jnp.int32))
        object.__setattr__(self, "character", _vector_tuple(character_arr, jnp.complex128))
        object.__setattr__(self, "perm_sign", _array_tuple(perm_sign_arr, jnp.int8))
        if local_map_arr is not None:
            object.__setattr__(self, "local_map", _array_tuple(local_map_arr, jnp.int32))
        if local_phase_arr is not None:
            object.__setattr__(self, "local_phase", _array_tuple(local_phase_arr, jnp.complex128))
        object.__setattr__(self, "local_sign", tuple(jnp.asarray(local_sign_arr, dtype=jnp.int8).tolist()))
        object.__setattr__(self, "particle_type", particle_type)

    @property
    def perm_array(self):
        return jnp.asarray(self.perm, dtype=jnp.int32)

    @property
    def character_array(self):
        return jnp.asarray(self.character, dtype=jnp.complex128)

    @property
    def perm_sign_array(self):
        return jnp.asarray(self.perm_sign, dtype=jnp.int8)

    @property
    def local_map_array(self):
        if self.local_map is None:
            return None
        return jnp.asarray(self.local_map, dtype=jnp.int32)

    @property
    def local_phase_array(self):
        if self.local_phase is None:
            return None
        return jnp.asarray(self.local_phase, dtype=jnp.complex128)

    @property
    def local_sign_array(self):
        return jnp.asarray(self.local_sign, dtype=jnp.int8)

    @property
    def nsymm(self) -> int:
        return int(len(self.perm))

    @property
    def nmodes(self) -> int:
        return int(len(self.perm[0]))

    @property
    def is_fermion(self) -> bool:
        return self.particle_type in {"spinless_fermion", "spinful_fermion"}

    # Useful to determine if  autoregressive networks can be used
    @property
    def has_nontrivial_projector_weights(self) -> bool:
        character = np.asarray(self.character, dtype=np.complex128)
        perm = np.asarray(self.perm, dtype=np.int32)
        perm_sign = np.asarray(self.perm_sign, dtype=np.int8)
        local_sign = np.asarray(self.local_sign, dtype=np.int8)
        local_phase = None if self.local_phase is None else np.asarray(self.local_phase, dtype=np.complex128)
        identity = np.arange(self.nmodes, dtype=np.int32)
        return bool(
            (not np.allclose(character, 1.0 + 0.0j))
            or np.any(perm_sign != 1)
            or np.any(local_sign != 0)
            or (local_phase is not None and not np.allclose(local_phase, 1.0 + 0.0j))
            or (self.is_fermion and np.any(perm != identity[None, :]))
        )

    # Used to determine if autoregressive networks are egligible for symmetrization
    @property
    def has_nontrivial_factors(self) -> bool:
        return self.has_nontrivial_projector_weights

    def _apply_local_map(self, states, idx=None):
        if self.local_map is None:
            return states
        states = jnp.asarray(states, dtype=jnp.int32)
        if idx is None:
            local_map = self.local_map_array
            return jnp.take_along_axis(local_map, states, axis=1)
        return self.local_map_array[idx][states]

    def transformed_states(self, s):
        s = jnp.asarray(s).reshape(self.nmodes)
        return self._apply_local_map(s[self.perm_array])

    def transformed_state_at(self, s, idx):
        s = jnp.asarray(s).reshape(self.nmodes)
        return self._apply_local_map(s[self.perm_array[idx]], idx)

    def random_transformed_state(self, s, key):
        idx = jax.random.randint(key, (), 0, self.nsymm)
        return self.transformed_state_at(s, idx)

    def random_transformed_states(self, samples, key):
        keys = jax.random.split(key, samples.shape[0])
        return jax.vmap(self.random_transformed_state)(samples, keys)

    def _fermion_occupancy_perm_sign(self, s, idx=None):
        if self.particle_type == "spinful_fermion":
            occ = _spinful_local_occupancy(s, self.nmodes)
            site_perm = self.perm_array if idx is None else self.perm_array[idx]
            perm = _site_perm_to_spinful_orbital_perm(site_perm, self.nmodes)
            sign = self.perm_sign_array if idx is None else self.perm_sign_array[idx]
            perm_sign = _spinful_site_sign_to_orbital_sign(sign, self.nmodes)
        else:
            occ = _spinless_local_occupancy(s, self.nmodes)
            perm = self.perm_array if idx is None else self.perm_array[idx]
            perm = _site_perm_to_spinless_orbital_perm(perm, self.nmodes)
            sign = self.perm_sign_array if idx is None else self.perm_sign_array[idx]
            site_order = jnp.arange(self.nmodes - 1, -1, -1)
            perm_sign = sign.astype(jnp.int32)[..., site_order]
        return occ, perm, perm_sign

    def _fermion_local_sign_at(self, s, idx):
        if not self.is_fermion:
            return jnp.asarray(1.0 + 0.0j, dtype=jnp.complex128)
        local_sign = self.local_sign_array[idx]
        pre_local_state = jnp.asarray(s).reshape(self.nmodes)[self.perm_array[idx]]
        if self.particle_type == "spinful_fermion":
            occ = _spinful_local_occupancy(pre_local_state, self.nmodes)
        else:
            occ = _spinless_local_occupancy(pre_local_state, self.nmodes)
        ph_sign = _fermion_particle_hole_sign_from_occ(occ)
        phase = jnp.asarray(1.0 + 0.0j, dtype=jnp.complex128)
        if self.local_phase is not None:
            phase = jnp.prod(self.local_phase_array[idx][pre_local_state])
        return (phase * jnp.where(local_sign == 1, ph_sign, 1.0 + 0.0j)).astype(jnp.complex128)

    def _fermion_local_signs(self, s):
        if not self.is_fermion:
            return jnp.ones((self.nsymm,), dtype=jnp.complex128)
        states = jnp.asarray(s).reshape(self.nmodes)[self.perm_array]
        if self.particle_type == "spinful_fermion":
            occ = jax.vmap(lambda x: _spinful_local_occupancy(x, self.nmodes))(states)
        else:
            occ = jax.vmap(lambda x: _spinless_local_occupancy(x, self.nmodes))(states)
        ph_sign = _fermion_particle_hole_sign_from_occ(occ)
        phase = jnp.ones((self.nsymm,), dtype=jnp.complex128)
        if self.local_phase is not None:
            phase = jnp.prod(jnp.take_along_axis(self.local_phase_array, states, axis=1), axis=1)
        return (phase * jnp.where(self.local_sign_array == 1, ph_sign, 1.0 + 0.0j)).astype(jnp.complex128)

    def fermion_sign_at(self, s, idx):
        if not self.is_fermion:
            return jnp.asarray(1.0 + 0.0j, dtype=jnp.complex128)
        occ, perm, perm_sign = self._fermion_occupancy_perm_sign(s, idx)
        inv_perm = jnp.argsort(perm)
        occ_pair = occ[:, None] & occ[None, :]
        inversions = (inv_perm[:, None] > inv_perm[None, :]) & occ_pair
        inversions = jnp.sum(jnp.triu(inversions.astype(jnp.int32), k=1))
        parity = 1 - 2 * jnp.bitwise_and(inversions, 1)
        boundary = jnp.prod(jnp.where(occ, perm_sign, 1))
        return (parity * boundary).astype(jnp.complex128) * self._fermion_local_sign_at(s, idx)

    def weight_at(self, s, idx):
        return self.character_array[idx] * self.fermion_sign_at(s, idx)

    def fermion_signs(self, s):
        if not self.is_fermion:
            return jnp.ones((self.nsymm,), dtype=jnp.complex128)
        occ, perm, perm_sign = self._fermion_occupancy_perm_sign(s, None)
        inv_perm = jnp.argsort(perm, axis=1)
        occ_pair = occ[None, :, None] & occ[None, None, :]
        inversions = (inv_perm[:, :, None] > inv_perm[:, None, :]) & occ_pair
        inversions = jnp.sum(jnp.triu(inversions.astype(jnp.int32), k=1), axis=(1, 2))
        parity = 1 - 2 * jnp.bitwise_and(inversions, 1)
        boundary = jnp.prod(jnp.where(occ[None, :], perm_sign, 1), axis=1)
        return (parity * boundary).astype(jnp.complex128) * self._fermion_local_signs(s)

    def weights(self, s):
        return self.character_array * self.fermion_signs(s)

    def __mul__(self, other):
        if isinstance(other, SymmetryProjector):
            return compose_projectors(self, other, name=f"{self.name}_x_{other.name}")
        if isinstance(other, nn.Module):
            return ProjectedOrbitNet(base_net=other, symmetry=self)
        return NotImplemented


def square_translation_symmetry(
    Lx: int,
    Ly: int,
    particle_type: ParticleType,
    q: tuple[int, int] = (0, 0),
    site_order: SiteOrder = "direct",
    site_enum: SiteEnumeration = None,
    site_enum_axes: SiteEnumerationAxes = "yx",
) -> SymmetryProjector:
    Lx, Ly = _validate_lattice_shape(Lx, Ly)
    particle_type = _normalize_particle_type(particle_type)
    enum, coords = _site_enum_lookup(Lx, Ly, site_order, site_enum, site_enum_axes)
    qx, qy = int(q[0]) % Lx, int(q[1]) % Ly
    site_perms = []
    characters = []
    for sy in range(Ly):
        for sx in range(Lx):
            site_perm = []
            for sample_idx in range(Lx * Ly):
                x, y = map(int, coords[sample_idx].tolist())
                site_perm.append(int(enum[(y - sy) % Ly, (x - sx) % Lx]))
            site_perms.append(site_perm)
            phase = -2j * math.pi * (qx * sx / Lx + qy * sy / Ly)
            characters.append(complex(cmath.exp(phase)))

    perm = jnp.asarray(site_perms, dtype=jnp.int32)
    return SymmetryProjector(
        Lx=Lx,
        Ly=Ly,
        particle_type=particle_type,
        perm=_array_tuple(perm, jnp.int32),
        character=tuple(characters),
        name="translation",
    )


def square_d4_symmetry(
    Lx: int,
    Ly: int,
    particle_type: ParticleType = "spin",
    site_order: SiteOrder = "direct",
    site_enum: SiteEnumeration = None,
    site_enum_axes: SiteEnumerationAxes = "yx",
) -> SymmetryProjector:
    Lx, Ly = _validate_lattice_shape(Lx, Ly)
    if Lx != Ly:
        raise ValueError("D4 symmetry requires a square lattice.")
    particle_type = _normalize_particle_type(particle_type)
    enum, coords = _site_enum_lookup(Lx, Ly, site_order, site_enum, site_enum_axes)
    transforms = (
        lambda x, y: (x, y),
        lambda x, y: (Lx - 1 - y, x),
        lambda x, y: (Lx - 1 - x, Ly - 1 - y),
        lambda x, y: (y, Ly - 1 - x),
        lambda x, y: (x, Ly - 1 - y),
        lambda x, y: (Lx - 1 - y, Ly - 1 - x),
        lambda x, y: (Lx - 1 - x, y),
        lambda x, y: (y, x),
    )
    site_perms = []
    for transform in transforms:
        site_perm = []
        for sample_idx in range(Lx * Ly):
            x, y = map(int, coords[sample_idx].tolist())
            xp, yp = transform(x, y)
            site_perm.append(int(enum[yp % Ly, xp % Lx]))
        site_perms.append(site_perm)
    site_perm = jnp.asarray(site_perms, dtype=jnp.int32)
    perm = site_perm
    return SymmetryProjector(
        Lx=Lx,
        Ly=Ly,
        particle_type=particle_type,
        perm=_array_tuple(perm, jnp.int32),
        character=tuple([1.0 + 0.0j] * int(perm.shape[0])),
        name="d4",
    )


def _unique_symmetry_projector(
    Lx: int,
    Ly: int,
    particle_type: ParticleType,
    perm,
    character,
    *,
    name: str,
    local_map=None,
    local_phase=None,
    local_sign=None,
) -> SymmetryProjector:
    particle_type = _normalize_particle_type(particle_type)
    perm = jnp.asarray(perm, dtype=jnp.int32)
    character = jnp.asarray(character, dtype=jnp.complex128)
    local_map_arr = None if local_map is None else jnp.asarray(local_map, dtype=jnp.int32)
    local_phase_arr = None if local_phase is None else jnp.asarray(local_phase, dtype=jnp.complex128)
    local_sign_arr = jnp.zeros((perm.shape[0],), dtype=jnp.int8) if local_sign is None else jnp.asarray(local_sign, dtype=jnp.int8)
    keep, character_unique = _deduplicate_symmetry_arrays(
        perm,
        character,
        local_map=local_map_arr,
        local_phase=local_phase_arr,
        local_sign=local_sign_arr,
    )
    return SymmetryProjector(
        Lx=Lx,
        Ly=Ly,
        particle_type=particle_type,
        perm=_array_tuple(perm[keep], jnp.int32),
        character=_vector_tuple(character_unique, jnp.complex128),
        local_map=None if local_map_arr is None else _array_tuple(local_map_arr[keep], jnp.int32),
        local_phase=None if local_phase_arr is None else _array_tuple(local_phase_arr[keep], jnp.complex128),
        local_sign=tuple(jnp.asarray(local_sign_arr[keep], dtype=jnp.int8).tolist()),
        name=name,
    )


def rectangle_d2_symmetry(
    Lx: int,
    Ly: int,
    particle_type: ParticleType = "spin",
    site_order: SiteOrder = "direct",
    site_enum: SiteEnumeration = None,
    site_enum_axes: SiteEnumerationAxes = "yx",
) -> SymmetryProjector:
    """Point-group symmetry of a rectangular periodic lattice.

    The generated operations are identity, 180-degree rotation, x-reflection,
    and y-reflection. Degenerate duplicates are removed, so ``Ly=1`` reduces to
    the one-dimensional reflection group.
    """
    Lx, Ly = _validate_lattice_shape(Lx, Ly)
    particle_type = _normalize_particle_type(particle_type)
    enum, coords = _site_enum_lookup(Lx, Ly, site_order, site_enum, site_enum_axes)
    transforms = (
        lambda x, y: (x, y),
        lambda x, y: (Lx - 1 - x, Ly - 1 - y),
        lambda x, y: (Lx - 1 - x, y),
        lambda x, y: (x, Ly - 1 - y),
    )
    site_perms = []
    for transform in transforms:
        site_perm = []
        for sample_idx in range(Lx * Ly):
            x, y = map(int, coords[sample_idx].tolist())
            xp, yp = transform(x, y)
            site_perm.append(int(enum[yp % Ly, xp % Lx]))
        site_perms.append(site_perm)
    return _unique_symmetry_projector(
        Lx,
        Ly,
        particle_type,
        site_perms,
        [1.0 + 0.0j] * len(site_perms),
        name="d2",
    )


def chain_reflection_symmetry(
    L: int,
    particle_type: ParticleType = "spin",
    site_order: SiteOrder = "direct",
    site_enum: SiteEnumeration = None,
    site_enum_axes: SiteEnumerationAxes = "yx",
) -> SymmetryProjector:
    """Reflection symmetry of a one-dimensional periodic chain."""
    return rectangle_d2_symmetry(
        int(L),
        1,
        particle_type,
        site_order=site_order,
        site_enum=site_enum,
        site_enum_axes=site_enum_axes,
    )


def point_group_symmetry(
    Lx: int,
    Ly: int,
    particle_type: ParticleType = "spin",
    site_order: SiteOrder = "direct",
    site_enum: SiteEnumeration = None,
    site_enum_axes: SiteEnumerationAxes = "yx",
) -> SymmetryProjector:
    """Use D4 for square lattices and D2/reflection for rectangles or chains."""
    Lx, Ly = _validate_lattice_shape(Lx, Ly)
    if Lx == Ly and Ly > 1:
        return square_d4_symmetry(
            Lx,
            Ly,
            particle_type,
            site_order=site_order,
            site_enum=site_enum,
            site_enum_axes=site_enum_axes,
        )
    return rectangle_d2_symmetry(
        Lx,
        Ly,
        particle_type,
        site_order=site_order,
        site_enum=site_enum,
        site_enum_axes=site_enum_axes,
    )


def particle_hole_symmetry(
    Lx: int,
    Ly: int,
    particle_type: ParticleType,
    local_dim: int | None = None,
) -> SymmetryProjector:
    """Global particle-hole transformation.

    Bosons require ``local_dim`` so local occupation ``n`` maps to
    ``local_dim - 1 - n``. Spinless fermions use ``0 <-> 1`` and spinful
    fermions use true ldim=4 site states, ``empty <-> doublon`` and
    ``up <-> down``.
    """
    Lx, Ly = _validate_lattice_shape(Lx, Ly)
    particle_type = _normalize_particle_type(particle_type)
    n_sites = Lx * Ly
    perm = [list(range(n_sites)), list(range(n_sites))]
    character = [1.0 + 0.0j, 1.0 + 0.0j]
    if particle_type == "boson":
        if local_dim is None:
            raise ValueError("particle_hole_symmetry for bosons requires local_dim.")
        local_dim = int(local_dim)
        if local_dim <= 0:
            raise ValueError("local_dim must be positive.")
        ph_map = list(range(local_dim - 1, -1, -1))
    elif particle_type == "spinless_fermion":
        ph_map = [1, 0]
    elif particle_type == "spinful_fermion":
        ph_map = [3, 2, 1, 0]
    else:
        raise ValueError("particle_hole_symmetry is defined for bosons and fermions, not spins.")
    local_sign = [0, 1] if particle_type in {"spinless_fermion", "spinful_fermion"} else [0, 0]
    return SymmetryProjector(
        Lx=Lx,
        Ly=Ly,
        particle_type=particle_type,
        perm=_array_tuple(perm, jnp.int32),
        character=tuple(character),
        local_map=_array_tuple([list(range(len(ph_map))), ph_map], jnp.int32),
        local_sign=tuple(local_sign),
        name="particle_hole",
    )


def spin_flip_symmetry(Lx: int, Ly: int, particle_type: ParticleType = "spin") -> SymmetryProjector:
    """Global spin flip.

    For spins this maps ``0 <-> 1``. For spinful fermions it swaps local
    ``up <-> down`` occupations and assigns a minus sign to each doublon,
    matching ``F c_up F^-1 = c_down`` and ``F c_down F^-1 = c_up``.
    """
    Lx, Ly = _validate_lattice_shape(Lx, Ly)
    particle_type = _normalize_particle_type(particle_type)
    if particle_type == "spin":
        local_map = [[0, 1], [1, 0]]
        local_phase = None
    elif particle_type == "spinful_fermion":
        local_map = [[0, 1, 2, 3], [0, 2, 1, 3]]
        local_phase = [[1.0 + 0.0j] * 4, [1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, -1.0 + 0.0j]]
    else:
        raise ValueError("spin_flip_symmetry is defined for spins and spinful fermions.")
    n_sites = Lx * Ly
    perm = [list(range(n_sites)), list(range(n_sites))]
    return SymmetryProjector(
        Lx=Lx,
        Ly=Ly,
        particle_type=particle_type,
        perm=_array_tuple(perm, jnp.int32),
        character=(1.0 + 0.0j, 1.0 + 0.0j),
        local_map=_array_tuple(local_map, jnp.int32),
        local_phase=None if local_phase is None else _array_tuple(local_phase, jnp.complex128),
        name="spin_flip",
    )


def _local_dimension(symm: SymmetryProjector) -> int | None:
    if symm.local_map is not None:
        return int(symm.local_map_array.shape[1])
    if symm.local_phase is not None:
        return int(symm.local_phase_array.shape[1])
    return None


def _compose_local_maps(left: SymmetryProjector, right: SymmetryProjector):
    left_map = left.local_map_array
    right_map = right.local_map_array
    if left_map is None and right_map is None:
        return None
    if left_map is None:
        return jnp.repeat(right_map[None, :, :], left.nsymm, axis=0).reshape(left.nsymm * right.nsymm, -1)
    if right_map is None:
        return jnp.repeat(left_map[:, None, :], right.nsymm, axis=1).reshape(left.nsymm * right.nsymm, -1)
    if left_map.shape[1] != right_map.shape[1]:
        raise ValueError("Cannot compose local maps with different local dimensions.")
    rows = []
    for left_idx in range(left.nsymm):
        for right_idx in range(right.nsymm):
            rows.append(right_map[right_idx][left_map[left_idx]])
    return jnp.stack(rows, axis=0)


def _compose_local_phases(left: SymmetryProjector, right: SymmetryProjector):
    if left.local_phase is None and right.local_phase is None:
        return None
    left_dim = _local_dimension(left)
    right_dim = _local_dimension(right)
    local_dim = left_dim if left_dim is not None else right_dim
    if right_dim is not None and local_dim != right_dim:
        raise ValueError("Cannot compose local phases with different local dimensions.")

    identity = jnp.arange(local_dim, dtype=jnp.int32)
    left_map = left.local_map_array
    if left_map is None:
        left_map = jnp.repeat(identity[None, :], left.nsymm, axis=0)
    left_phase = left.local_phase_array
    if left_phase is None:
        left_phase = jnp.ones((left.nsymm, local_dim), dtype=jnp.complex128)
    right_phase = right.local_phase_array
    if right_phase is None:
        right_phase = jnp.ones((right.nsymm, local_dim), dtype=jnp.complex128)

    rows = []
    for left_idx in range(left.nsymm):
        for right_idx in range(right.nsymm):
            rows.append(left_phase[left_idx] * right_phase[right_idx][left_map[left_idx]])
    return jnp.stack(rows, axis=0)


def _compose_local_signs(left: SymmetryProjector, right: SymmetryProjector):
    return jnp.bitwise_xor(left.local_sign_array[:, None], right.local_sign_array[None, :]).reshape(-1)


def compose_projectors(left: SymmetryProjector, right: SymmetryProjector, *, name: str = "composed") -> SymmetryProjector:
    if (left.Lx, left.Ly, left.particle_type) != (right.Lx, right.Ly, right.particle_type):
        raise ValueError("Only compatible symmetries can be composed.")
    left_perm = left.perm_array
    right_perm = right.perm_array
    perm = left_perm[:, right_perm].reshape(left.nsymm * right.nsymm, left.nmodes)
    character = (left.character_array[:, None] * right.character_array[None, :]).reshape(-1)
    perm_sign = left.perm_sign_array[:, right_perm] * right.perm_sign_array[None, :, :]
    perm_sign = perm_sign.reshape(left.nsymm * right.nsymm, left.nmodes)
    local_map = _compose_local_maps(left, right)
    local_phase = _compose_local_phases(left, right)
    local_sign = _compose_local_signs(left, right)

    keep, character_unique = _deduplicate_symmetry_arrays(
        perm,
        character,
        local_map=local_map,
        local_phase=local_phase,
        local_sign=local_sign,
        perm_sign=perm_sign,
    )
    return SymmetryProjector(
        Lx=left.Lx,
        Ly=left.Ly,
        particle_type=left.particle_type,
        perm=_array_tuple(perm[keep], jnp.int32),
        character=_vector_tuple(character_unique, jnp.complex128),
        perm_sign=_array_tuple(perm_sign[keep], jnp.int8),
        local_map=None if local_map is None else _array_tuple(local_map[keep], jnp.int32),
        local_phase=None if local_phase is None else _array_tuple(local_phase[keep], jnp.complex128),
        local_sign=tuple(jnp.asarray(local_sign[keep], dtype=jnp.int8).tolist()),
        name=name,
    )


class ProjectedOrbitNet(nn.Module):
    """Apply a ``SymmetryProjector`` orbit projector to a raw log-amplitude network.

    ``symmetry_average`` selects how orbit values are combined:
    ``"exp"`` averages complex amplitudes in log space and includes symmetry
    weights, ``"log"`` averages log coefficients directly, and ``"sep"``
    averages log-probabilities and phases separately. Autoregressive networks
    require ``"sep"`` or a custom callable. The built-in ``"sep"`` average
    assumes the pure-state convention ``logProbFactor=0.5``. Custom callables
    receive ``(log_amplitudes, weights)``.

    ``coefficient_net`` may return one log coefficient per symmetry element
    from the original state. ``coefficient_state_net`` may instead return one
    scalar log coefficient per transformed state and is vectorized over the
    orbit. Use only one of the two coefficient interfaces.
    """

    base_net: nn.Module
    symmetry: SymmetryProjector
    coefficient_net: nn.Module | None = None
    coefficient_state_net: nn.Module | None = None
    symmetry_average: SymmetryAverage = "exp"

    def setup(self):
        if self.symmetry_average not in ("exp", "log", "sep") and not callable(self.symmetry_average):
            raise ValueError("symmetry_average must be 'exp', 'log', 'sep', or a callable.")
        if self.coefficient_net is not None and self.coefficient_state_net is not None:
            raise ValueError("Pass only one of coefficient_net or coefficient_state_net.")
        if self.symmetry_average in ("log", "sep") and self.symmetry.has_nontrivial_projector_weights:
            warnings.warn(
                f"symmetry_average='{self.symmetry_average}' does not apply nontrivial projector weights "
                "as a linear wavefunction projector. Use symmetry_average='exp' for character/sign-weighted "
                "projection, or proceed only if this non-linear orbit average is intentional.",
                UserWarning,
                stacklevel=2,
            )
        if _has_autoregressive_sampler(self.base_net):
            if self.symmetry_average != "sep" and not callable(self.symmetry_average):
                raise ValueError("Autoregressive/sampling networks require symmetry_average='sep' or a custom callable.")
            if self.symmetry.has_nontrivial_projector_weights:
                raise ValueError(
                    "Autoregressive/sampling networks are only compatible with symmetry projectors "
                    "whose bare configuration action has no nontrivial character, boundary sign, "
                    "fermion permutation sign, or particle-hole sign."
                )

    @nn.compact
    def __call__(self, s):
        s = jnp.asarray(s).reshape(self.symmetry.nmodes)
        states = self.symmetry.transformed_states(s)
        base_logs = jax.vmap(self.base_net)(states)
        base_logs = jnp.asarray(base_logs, dtype=jnp.complex128)
        if self.coefficient_net is not None:
            coeff_logs = jnp.asarray(self.coefficient_net(s), dtype=jnp.complex128)
        elif self.coefficient_state_net is not None:
            coeff_logs = jnp.asarray(jax.vmap(self.coefficient_state_net)(states), dtype=jnp.complex128)
        else:
            coeff_logs = jnp.zeros((self.symmetry.nsymm,), dtype=jnp.complex128)
        if coeff_logs.shape != (self.symmetry.nsymm,):
            raise ValueError("coefficient_net and coefficient_state_net must return one log coefficient per symmetry element.")
        return _average_coefficients(
            base_logs + coeff_logs,
            self.symmetry.weights(s),
            self.symmetry_average,
        )

    def sample(self, key):
        sample_key, orbit_key = jax.random.split(key)
        sample = self.base_net.sample(sample_key)
        return self.symmetry.random_transformed_state(sample, orbit_key)
