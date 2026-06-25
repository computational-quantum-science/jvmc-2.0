"""
Symmetry projectors for NQS orbit sums.

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

from typing import Literal, Sequence
import jax.numpy as jnp
import cmath
import math

from .util import ParticleType, normalize_particle_type, array_tuple, vector_tuple, deduplicate_symmetry_arrays
from .projector import SymmetryProjector

SiteEnumeration = Sequence[Sequence[int]] | jnp.ndarray | None
SiteOrder = Literal["direct", "snake", "custom"]
SiteEnumerationAxes = Literal["yx", "xy"]

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
    particle_type = normalize_particle_type(particle_type)
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
        perm=array_tuple(perm, jnp.int32),
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
    particle_type = normalize_particle_type(particle_type)
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
        perm=array_tuple(perm, jnp.int32),
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
    particle_type = normalize_particle_type(particle_type)
    perm = jnp.asarray(perm, dtype=jnp.int32)
    character = jnp.asarray(character, dtype=jnp.complex128)
    local_map_arr = None if local_map is None else jnp.asarray(local_map, dtype=jnp.int32)
    local_phase_arr = None if local_phase is None else jnp.asarray(local_phase, dtype=jnp.complex128)
    local_sign_arr = jnp.zeros((perm.shape[0],), dtype=jnp.int8) if local_sign is None else jnp.asarray(local_sign, dtype=jnp.int8)
    keep, character_unique = deduplicate_symmetry_arrays(
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
        perm=array_tuple(perm[keep], jnp.int32),
        character=vector_tuple(character_unique, jnp.complex128),
        local_map=None if local_map_arr is None else array_tuple(local_map_arr[keep], jnp.int32),
        local_phase=None if local_phase_arr is None else array_tuple(local_phase_arr[keep], jnp.complex128),
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
    particle_type = normalize_particle_type(particle_type)
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
    particle_type = normalize_particle_type(particle_type)
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
        perm=array_tuple(perm, jnp.int32),
        character=tuple(character),
        local_map=array_tuple([list(range(len(ph_map))), ph_map], jnp.int32),
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
    particle_type = normalize_particle_type(particle_type)
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
        perm=array_tuple(perm, jnp.int32),
        character=(1.0 + 0.0j, 1.0 + 0.0j),
        local_map=array_tuple(local_map, jnp.int32),
        local_phase=None if local_phase is None else array_tuple(local_phase, jnp.complex128),
        name="spin_flip",
    )