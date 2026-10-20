# Port of the randomized eigendecomposition (RSVD) TDVP solver to jVMC 2.0

Date: 2026-10-20. Branch: `rsvd-solver` of `jvmc-2.0` (local clone
`/Users/wladi/Projects/VMC/jvmc-2.0`).

Source of the port: `old_scripts/0encap_folder/crit_tfim/Resnet_nc30_rsvd_scan_cutoff`
(`rsvd.py`, `mytdvp.py`, `run.py`), written against the old `jVMC` package.

## 1. Git housekeeping

| Step | Result |
|---|---|
| Uncommitted changes that were sitting on local `master` (`TargetFractionCutoffSampler`, importance-weight fix in `AbstractOptimizer.__call__`, sampler tests) | committed on the new branch `target-fraction-cutoff-sampler` (commit `5e20970`) |
| GitHub `origin/master` (5 new commits: BatchLayout, block-wise QGT, Gaussian SNR mode, chunked Jacobian products, lazy MinSR scaling) | local `master` fast-forwarded to `e18a5d2` |
| Cluster clone (`er:/local/krinitsin/VMC/jvmc-2.0`) | at the same commit as the old local master; its only working-tree change was a trailing newline, so nothing to pull from there |
| Baseline for the new work | branch `rsvd-solver` = `master` + merge of `target-fraction-cutoff-sampler` (merge commit `2d9b457`). One conflict in `optimizer/base.py`: both sides fixed the importance weights of the later stepper stages the same way; the upstream version was kept. |

The pre-existing local branch `rsvd` (identical to the old master, no commits of
its own) was left untouched.

## 2. Library changes (`jVMC_exp`)

### New: `jVMC_exp/solver/qgt_operator.py` -- `QGTOperator`

The matrix-free left hand side of the TDVP/SR equation,

    A v = q[S_0] v + diag_scale * diag(S_0) v + diag_shift * v,   S_0 = J^dag J,

built from the normalized Jacobian `J` (shape `(num_samples, num_params)`,
sharded over the samples). It is what the optimizer now hands to every
matrix-free solver (`_needs_dense_matrix == False`), replacing the bare closure
`matvec(v)` that `Evolution._get_lhs_lazy` used to return.

* `A(v)` and `A.raw(v)` (bare `S_0 v`) accept a vector `(num_params,)` or a
  block `(num_params, r)`; they are jitted.
* **Sharding.** The product is a `shard_map` over the sample axis
  (`in_specs=(DEVICE_SPEC, REPLICATED_SPEC)`, `out_specs=REPLICATED_SPEC`): each
  device contracts its own samples, the partial sums are `psum`ed. Only the
  `(num_params, r)` blocks are replicated. This is the same layout the old
  `rsvd.py` relied on, made explicit.
* **Chunking.** Inside each device the product is summed over chunks of at most
  `global_defs.MATMUL_CHUNK_BYTES` of the Jacobian (same knob and same
  `optimization_barrier` trick as upstream's `Evolution._add_qgt`), so the
  intermediate `J_chunk @ Q` is bounded in size instead of being
  `(num_samples_local, r)`.
* **Transpose.** `jax.scipy.sparse.linalg.cg` transposes the operator it is given
  and the chunked loop is not transposable by JAX itself, so the product is
  wrapped in `jax.custom_derivatives.linear_call` with the explicit transpose
  `conj(S_0 conj(w))` (`S_0` is Hermitian). The existing `CG` solver therefore
  keeps working, and now also gets the sharded/chunked product.
* **Linearity.** `q = Re` and `q = i Im` are only real-linear, so `q[S_0 v]` equals
  the matrix product `q[S_0] v` for real `v` only. The operator is defined as the
  matrix `q[S_0]`: a complex `v` is split into real and imaginary parts (one
  product of double width). The old closure silently had the "real `v` only"
  semantics.

### New: `jVMC_exp/solver/rsvd.py` -- `randomized_eigh`, `RandomizedPinvSNR`

`randomized_eigh(operator, k, key, n_oversamples, n_iter, which, diagonalization_mode, T_A)`
is the port of `REIGH.randomized_eigh_jac`:

* Halko-style subspace iteration, `2 * n_iter + 2` operator applications on a
  block of `r = min(k + n_oversamples, num_params)` probe vectors, one jitted
  kernel (`_subspace_iteration`) compiled once per shape; the probe block is
  drawn inside the jit, replicated.
* Reduction handling as in the old code: real probe for `q = Re`; for `q = i Im`
  the factor `i` is stripped so that the iterate stays real (real antisymmetric
  `Im(S_0)`), and restored once in the Rayleigh-Ritz matrix. New: a holomorphic
  ansatz (`q = identity`) uses a complex probe, and `q = i Im` with diagonal
  terms (which do not share the factor `i`) falls back to complex arithmetic.
* **Rayleigh-Ritz diagonalization through `jVMC_exp.solver.util.diagonalize`**,
  i.e. with `diagonalization_mode in {"device", "distributed", "host"}`. For
  `"distributed"` (jaxmg `syevd` on `MESH_2D`) the `r x r` matrix is padded to a
  multiple of the number of devices; the padded eigenvalues are pushed to the
  top of the spectrum by `diagonalize` and dropped again. The Ritz vectors
  `Q @ V_small` are formed with the row-sharded `V_small`, then replicated. All
  fallbacks of `diagonalize` (no jaxmg, no GPU, single device,
  `JVMC_USE_DISTRIBUTED` unset) apply unchanged.
* Selection by largest magnitude (`"LM"`, default) or largest algebraic value
  (`"LA"`), returned in **ascending** order like `jnp.linalg.eigh`, because the
  pseudo-inverse solvers read `ev[-1]` as the largest eigenvalue (the bug the old
  `mytdvp.py` comments warn about).

`RandomizedPinvSNR(PinvSNR)` is the solver:

* Same regularization as `PinvSNR` (smooth pinv cutoff, adaptive cutoff until
  `pinv_tol`, optional SNR cutoff in `"empiric"` or `"gaussian"` mode), applied
  in the truncated eigenbasis. Implemented by overriding `Pinv._diagonalize`
  only, so the whole `PinvSNR.__call__` is reused.
* `_needs_dense_matrix = False`: the optimizer builds a `QGTOperator`, never the
  dense `S`. The parameter padding of the dense path (`pad_size`) does not
  apply and is ignored.
* New constructor arguments: `n_components`, `n_oversamples=0`, `n_iter=4`,
  `seed=4321`, `which="LM"`, `compute_true_residual=True`; the `PinvSNR`
  arguments (`snr_tol`, `pinv_tol`, `pinv_cutoff`, `snr_mode`,
  `diagonalization_mode`, `T_A`) are passed through.
* The PRNG key is split with `sharding_config.broadcast_split_key` before every
  solve, so every process of a multi-process run draws the same probes.
* Extra entries in the info/metadata dictionary: `n_components` and
  `true_residual = ||A x - b|| / ||b||` (the old script's `res_rsvd`, one extra
  operator application). `residual` keeps the `PinvSNR` meaning (force discarded
  inside the retained subspace) so that cross validation etc. are unchanged.

Exports: `jVMC_exp.solver.RandomizedPinvSNR`, `jVMC_exp.solver.randomized_eigh`,
`jVMC_exp.solver.QGTOperator`.

### Changed: `jVMC_exp/optimizer/base.py`

* `Evolution.__init__` records `self._lhs_trans_mode` (`"identity" | "real" |
  "imag"`) next to `self._lhs_trans_fn`.
* `Evolution._get_lhs_lazy` returns a `QGTOperator` (with the optimizer's
  `diag_scale` / `diag_shift`) and stores `operator.raw` as `self._qgt` when
  `store_qgt` is set, so `_get_tdvp_error`, cross validation and the script's
  Fubini-Study norm work with `opt.qgt(v)` as before.
* The `LazySampledObs` (batched Jacobian) rejection is unchanged: the
  randomized solver needs the materialized Jacobian (`batched_jacobian=False`).

### New tests: `tests/rsvd_test.py`

Operator against the dense matrix (all three reductions, blocks, diagonal terms,
chunking); `randomized_eigh` against `jnp.linalg.eigh` (exact at full rank for all
reductions, dominant eigenpairs at low rank, `"LA"`, diagonal terms, the padded
distributed path, argument checks); `RandomizedPinvSNR` in `TDVP` (reference
dynamics of `tdvp_test.py` at full rank, step-for-step agreement with `PinvSNR`,
low-rank solve equal to the dense one when the retained spectrum reaches the
cutoff, SNR regularization with an MC sampler in both modes, batched Jacobian
rejected), non-holomorphic ansatz for `make_real=True/False` (real update, equal
to the dense solve), `SR` ground-state search. Run with one CPU device and with
`XLA_FLAGS=--xla_force_host_platform_device_count=4` (sharded paths).

Results on 2026-10-20 (conda env `jvmc`, jax 0.9.1, CPU): `rsvd_test`, `cg_test`,
`tdvp_test`, `sr_test` -- 30 passed on one device; `rsvd_test` + `cg_test` -- 20
passed on four emulated devices. The experiment script ran end to end on CPU with a
tiny configuration (L=4, 184 parameters, rank 40, 2000 samples, 3 steps).

### Not ported (on purpose)

From the old `mytdvp.py`: the L-BFGS refinement of the update (`use_implicit`,
jaxopt), the Hutchinson reconstruction-error estimate, the nullspace/band-rotation
diagnostics (`commit_step`, `band_*`), `clear_caches`, and the per-solve
`solve_log`. They were scan diagnostics around the solver rather than part of it;
the metadata the optimizer already writes (`spectrum`, `residual`,
`true_residual`, `pinv_cutoff`, `effective_rank`, `condition_number`,
`tdvp_error`) covers the quantities the scan plotted, except the band rotation.

## 3. The experiment

* Source folder: `scripts/rsvd/` (`run.py`, `input.json`, `nets/`, `.encap.conf`,
  `sbatch.sh`, `_sbatch.sh`, `_Cluster/`).
* Version folder, as requested: `scripts/0encap_folder/rsvd/test/` (identical
  copy; launch with `encap rerun rsvd -n test ...` from `scripts/`, or with the
  copied `sbatch.sh` for the rank-per-GPU distributed mode).

`run.py` mirrors `scripts/0encap_folder/tfim_dynamics/Ns_3e5/run.py` and differs
in exactly these points:

1. TDVP solver `RandomizedPinvSNR(...)` instead of `PinvSNR(...)`, configured by
   the new `simulation.rsvd` block of `input.json`
   (`n_components=4000` of 12592 real parameters, `n_oversamples=0`, `n_iter=4`,
   `which="LM"`, `seed=4321`, `diagonalization_mode="distributed"`); a run entry
   may override the block. `n_components` is clipped to the number of parameters
   with a warning. The ground-state search for the x-polarized initial state keeps
   the dense `PinvSNR`.
2. `jVMC_exp.sampler.CutoffSampler` instead of `MCSampler`, with the new
   `simulation.eps = 1e-3` (the value of the old 1D script); the bootstrap run that
   initializes the cutoff is capped at 20000 samples. `snr_tol` stays 0, as
   required for importance-weighted samples.
3. `tdvp_norm` uses `opt.qgt` (upstream renamed `_S0` to `qgt`; for the
   matrix-free solver it is the operator `S_0 v`).
4. The data-file name carries `_eps=..._rsvd_nc=..._os=..._it=...`, the `config`
   group of the HDF5 file records the sampler/solver settings, and the per-step
   log line prints `true_res` and the current sampler cutoff.
5. `sbatch.sh`: job name `rsvd_test`, the stale `--dependency` line removed.

`diagonalization_mode="distributed"` only takes effect with
`JVMC_USE_DISTRIBUTED=true` and one rank per GPU (the `sbatch.sh` route,
`--ntasks-per-node=4`); through `encap`'s `.encap.conf` (`ntasks: 1`) it falls
back to the on-device `eigh` of the `4000 x 4000` Rayleigh-Ritz matrix with a
warning.

## 4. Things to keep in mind

* Choose `n_components` so that the smallest retained eigenvalue is already below
  `pinv_cutoff` relative to the largest one; `true_residual` in the metadata
  tells when the truncation discards force. `n_oversamples` is the cheap accuracy
  lever, `n_iter` the expensive one.
* The replicated `(num_params, r)` blocks (probe, `Q`, `Y`, `V`) are what limits
  the memory, as in the old implementation: for `12592 x 4000` doubles that is
  0.4 GB each.
* A batched Jacobian (`Observable(..., batched_jacobian=True)`) is rejected, as
  for every matrix-free solver.
