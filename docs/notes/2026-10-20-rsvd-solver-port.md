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

* **L-BFGS refinement (toggle, port of `use_implicit` of the old `mytdvp.py`).**
  `lbfgs=True` minimizes the true residual `||A x - b|| / ||b||` with
  `jaxopt.LBFGS`, warm-started from the randomized pseudo-inverse update,
  whenever that residual exceeds `lbfgs_tol` (default `1e-7`). As in the old
  code the L-BFGS run (`lbfgs_maxiter=1000` iterations) is restarted while the
  residual still improves by more than `lbfgs_rtol=1e-3` per run; new is the
  hard cap `lbfgs_max_restarts=10`. The optimization variables are real: the
  update itself for a non-holomorphic ansatz (as in the old script), real and
  imaginary part for a holomorphic one (new). The loss is a module-level
  function of the Jacobian, so jaxopt compiles it once per operator type and
  reuses it across steps. Extra info/metadata: `residual_rsvd` (before),
  `true_residual` (after), `n_lbfgs` (runs), `lbfgs_iterations`, `t_lbfgs`,
  `cos_lbfgs`. `jaxopt` is imported lazily; without it `lbfgs=True` raises an
  `ImportError` at the first solve. Note: with `q = i Im` and a diagonal shift
  the equation has no real solution, so the refinement cannot reach a small
  residual there (the shift adds a real component to a purely imaginary system).
* The explicit transpose of the product returns the real part of the
  cotangent for a real input (JAX's convention for a real-to-complex map), which
  is what makes the L-BFGS gradient well typed for real updates.

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
to the dense solve), `SR` ground-state search, and the L-BFGS refinement (recovers
the discarded force at rank 2 of 8 and moves the update towards the dense one, is
skipped below tolerance, works for both non-holomorphic reductions). Run with one CPU device and with
`XLA_FLAGS=--xla_force_host_platform_device_count=4` (sharded paths).

Results on 2026-10-20 (conda env `jvmc`, jax 0.9.1, CPU): `rsvd_test`, `cg_test`,
`tdvp_test`, `sr_test` -- 30 passed on one device; `rsvd_test` + `cg_test` -- 20
passed on four emulated devices. The experiment script ran end to end on CPU with a
tiny configuration (L=4, 184 parameters, rank 40, 2000 samples, 3 steps).
After adding the L-BFGS refinement: 34 passed on one device, 24 on four; the
same tiny script run with `lbfgs=true` brought the true residual from 5e-2 to 3e-3
(5 runs of 200 iterations, tolerance 1e-4 not yet reached at that budget).

### Not ported (on purpose)

From the old `mytdvp.py`: the Hutchinson reconstruction-error estimate, the
nullspace/band-rotation diagnostics (`commit_step`, `band_*`), `clear_caches`,
and the per-solve `solve_log`. They were scan diagnostics around the solver
rather than part of it (the band diagnostics are to be added in the experiment
script); the metadata the optimizer already writes (`spectrum`, `residual`,
`true_residual`, `residual_rsvd`, `pinv_cutoff`, `effective_rank`,
`condition_number`, `tdvp_error`, the `*_lbfgs` entries) covers the quantities
the scan plotted, except the band rotation.

## 3. The experiment

* Source folder: `scripts/rsvd/` (`run.py`, `input.json`, `nets/`, `.encap.conf`,
  `sbatch.sh`, `_sbatch.sh`, `_Cluster/`, the parameter file).
* Version folder, as requested: `scripts/0encap_folder/rsvd/test/` (identical
  copy; `encap rerun rsvd -n test -sl -sl_i 2 -vm <cluster>` from `scripts/`, or
  `sbatch --export=ALL,ENCAP_SLURM_INSTANCE=<grid point> sbatch.sh` in the folder).

### 3.1 What it runs (2026-10-20, second iteration)

The port of `old_scripts/0encap_folder/crit_tfim/Resnet_nc56_rsvd_scan_cutoff`
(jVMC 1) to jvmc-2.0, on two nodes with 8 GPUs:

| | old nc56 run | `scripts/rsvd` |
|---|---|---|
| system | open chain L = 20, H = -sum ZZ - sum X | `system.dimension = 1`, `periodic = false`, `J = 1`, `h = h_ratio * h_crit = 1` |
| ansatz | 1D ResNet, 56 channels, 2 blocks, filter 10, no bias, 94640 real parameters | `nets/resnet1d.py` (same parameter tree) |
| symmetry | reflection + spin flip (`get_orbit_1D`) | `compose_projectors(chain_reflection_symmetry, spin_flip_symmetry)`; jvmc-2.0 averages over the orbit where jVMC 1 summed, a constant shift of log psi |
| initial state | `params_GS_psi2Samp_L=20_numChannels=56_numSamples=40000.npy` | the same file, `simulation.init.params_file`; verified to load unchanged (<X> = 1.000001, Var(H_x)/N = 1e-6, symmetric) |
| if missing | generated once by `Resnet_nc40_*`: 200 SR steps for H_x, rsvd rank 1000 + L-BFGS, diagonal scale 10 | `init.mode = gs_search` with `init.solver = "rsvd"` (rank 1000 + L-BFGS), 200 Euler steps at dt 1e-2, stopped at Var(H_x)/N < 1e-6, saved under `params_file` |
| sampler | `CutoffSampler`, eps 1e-3, 40000 samples, 25 chains, 25 thermalization sweeps | same, 32 chains (divisible by 8 GPUs); the TDVP samples are reused for the observables (jVMC 1 used a separate 20000-sample MC sampler) |
| solver | rsvd, `makeReal = "real"`, pinv cutoff 1e-8 | `RandomizedPinvSNR`, `make_real = True`, `pinv_cutoff = 1e-8`, distributed Ritz diagonalization |
| grid | n_comps 8000 and 16000, n_oversamples 0, n_iter 1, no L-BFGS | `simulation.runs[0/1]` with `rsvd` overrides; one job per entry via `ENCAP_SLURM_INSTANCE` |
| stepper | `AdaptiveHeun(dt = 1e-4, tol = 1e-4)`, Fubini-Study norm | `simulation.stepper = "AdaptiveHeun"`, `rtol = 0`, `atol = 1e-4`, same norm, no max step |
| observables | Z, ZZ1, ZZ2, X, XX1, XX2 per site (divided by L) | X, Z, ZZ_nn, XX_nn, ZZ_d2, XX_d2 per site / per bond (divided by the bond count) |
| output | CSV per step, h5 at the end | h5 + CSV with checkpoints and restarts (unchanged from `Ns_3e5`) |

Script changes for this (all behind `input.json` keys, the 2D setup is still
available with `system.dimension = 2`):

1. `system.dimension`, 1D bonds and sample shape; `net.ResNet.bias`.
2. `simulation.symmetry` gained `"reflection"` and `"reflection_spin_flip"`.
3. `simulation.init.params_file` (explicit file to load / save), with a parameter
   count check against the network; `simulation.init.solver` (`"rsvd"` with
   `init.rsvd`, or `"pinv"`).
4. `simulation.stepper` (`"RK45"` or `"AdaptiveHeun"`), `maxStep: null` allowed.
5. The grid point is read from `ENCAP_SLURM_INSTANCE` before `ENCAP_PROCID`: with
   several ranks per job encap sets `ENCAP_PROCID = SLURM_PROCID`, which differs
   between ranks.
6. `.encap.conf`: `nodes: 2`, `ntasks: 8`, `ntasks-per-node: 4`, `gres: gpu:4`,
   `JVMC_USE_DISTRIBUTED=true` and `JAX_PLATFORMS=cuda` exported before `srun`;
   `sbatch.sh` likewise for 2 nodes, grid point via `ENCAP_SLURM_INSTANCE`.

Memory on 8 GPUs: the Jacobian (40000 x 94640 complex) is 7.6 GB per GPU; the
replicated `(94640, r)` float64 blocks of the solver are 6 GB each at r = 8000
and 12 GB at r = 16000, three of them at once, plus the r x r Ritz matrix
(distributed). r = 16000 therefore needs the 80/96 GB cards.

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
