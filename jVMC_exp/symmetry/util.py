import jax.numpy as jnp
from typing import Literal

ParticleType = Literal["spin", "boson", "bosons", "spinless_fermion", "spinful_fermion"]
_PARTICLE_TYPES = {"spin", "boson", "spinless_fermion", "spinful_fermion"}

def deduplicate_symmetry_arrays(perm, character, local_map=None, local_phase=None, local_sign=None, perm_sign=None):
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

def array_tuple(x, dtype):
    return tuple(map(tuple, jnp.asarray(x, dtype=dtype).tolist()))

def vector_tuple(x, dtype):
    return tuple(jnp.asarray(x, dtype=dtype).tolist())

def normalize_particle_type(particle_type: ParticleType) -> str:
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