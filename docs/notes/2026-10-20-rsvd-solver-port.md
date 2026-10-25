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
  `(num_samples_local, r)`. Since 2026-10-22 the result is also accumulated in
  blocks of parameter rows bounded by the same budget: XLA fuses the accumulation
  into the GEMM and autotunes it on copies of all operands, accumulator included,
  which for a single `(94640, 16000)` complex output was a 22.6 GiB allocation
  that ran out of memory on 96 GB cards.
* **Eager, block-wise subspace iteration (2026-10-22, second attempt at r = 16000).**
  The whole iteration used to be one jitted kernel; XLA's buffer assignment for it
  kept several full `(94640, 16000)` temporaries (accumulator and all-reduce
  output of an application next to `Q`, `Y` and the QR workspace): a 72.9 GiB
  peak and a 47.4 GiB temporary that could not be allocated. The stages are now
  separate dispatches (`rsvd._subspace_iteration`): an application is done in
  column blocks of at most `MATMUL_CHUNK_BYTES` of output, each written into a
  preallocated result, the QR donates its input, and buffers are released between
  stages, so at most two full blocks plus the temporaries of one column block (or
  the QR workspace) are alive: about 25 GB for 94640 x 16000 doubles plus the
  Jacobian shard. Doing the blocking inside the single kernel was tried first and
  made XLA's temporary buffer larger. `randomized_eigh` also replicates the small
  eigenvector matrix explicitly before forming the Ritz vectors and skips the
  final column selection when it is the identity (no oversampling, ascending
  spectrum). `JVMC_RSVD_DEBUG=1` synchronizes after every stage and prints the
  device memory of process 0, so an out-of-memory error is reported where it
  happens.
* **Real arithmetic for the reductions.** For `q = Re` and `q = i Im` with a real
  input (the subspace iteration and every non-holomorphic update) the product is
  `A^T(Av) + B^T(Bv)` resp. `A^T(Bv) - B^T(Av)` with `J = A + iB`
  (`_qgt_product_re` / `_qgt_product_im`): a real accumulator, half the memory and
  two thirds of the flops of the complex product followed by taking the part. Both
  carry explicit transposes (symmetric resp. antisymmetric).
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
and 12 GB at r = 16000, about four of them at once (`Q`, `Y`, the reduction
buffer, the QR workspace), plus the r x r Ritz matrix (distributed). More GPUs
do not shrink the replicated blocks, only the Jacobian shard. The first attempt
at r = 16000 on JUPITER (2 x 4 GH200 120 GB) died in the autotuner of the
product with a 22.6 GiB complex accumulator, the second in the 47.4 GiB
temporary of the subspace-iteration executable; see the chunking, real-arithmetic
and column-blocking entries in section 2 for the fixes.

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

### 4.1 Stage timings (2026-10-24)

`RandomizedPinvSNR(stage_timings=True)` synchronizes after every stage of the
randomized decomposition and reports the accumulated wall times in the info
dictionary (hence in the optimizer's `meta_data`): `t_redistribute` (parameters
layout: the all-to-all of the Jacobian), `t_probe`, `t_apply` (all operator
applications), `t_qr`, `t_ritz_matrix`, `t_eigh`, `t_ritz_vectors`, `t_select`
and their sum `t_rsvd`. `JVMC_RSVD_DEBUG=1` switches the same timers on and
additionally prints each stage with the device memory. The limits scan
(`scripts/0encap_folder/rsvd/limits_scan`) writes them into its
`limits_results.jsonl`.

## 5. Parameter-sharded layout (2026-10-23)

Motivation: in the `"samples"` layout above the `(num_params, r)` blocks are
replicated, 12 GB each for 94640 x 16000 doubles, whatever the number of GPUs. The
sample count stays at 40000 while the networks grow, so memory has to scale with
`num_params / n_GPUs`. `RandomizedPinvSNR(layout="parameters", mesh_shape=(P_s, P_p))`
does that; `Evolution`, `SampledObs`, the dense path, MinSR and the steppers are
unchanged (the conversion happens inside the solver). The library default stays
`layout="samples"`; the experiment script defaults to `"parameters"`.

### 5.1 Layout

* **Mesh** (`qgt_operator.parameter_mesh`): `P_s` sample groups x `P_p` parameter
  groups over the devices of `MESH` in the same order (device `i` at
  `(i // P_p, i % P_p)`), axes `("samples", "params")`. Default `(1, n_devices)`:
  pure parameter sharding. `P_s * P_p` must equal the device count.
* **Jacobian** (`ShardedQGTOperator`, built per solve by `from_operator` from the
  optimizer's `QGTOperator`): the sample-sharded normalized Jacobian is
  redistributed once per solve by one `jax.lax.all_to_all` along `"params"` (each
  device splits its columns into `P_p` blocks and concatenates the row blocks it
  receives) into a second copy, `J[s, p]` of shape
  `(num_samples / P_s, padded_num_params / P_p)` on device `(s, p)`. The
  `SampledObs` copy is untouched; the second copy lives only during the solve.
  `num_params` is padded with zero columns to a multiple of `P_p`; the compiled
  redistribution contains only the all-to-all (checked in the HLO).
* **Blocks**: probe, iterates `Y`, `Q`, `A Q` and the Ritz vectors `V` are
  row-sharded, `padded_num_params / P_p` rows per device, replicated over the sample
  groups. Padding rows are zero in the probe and stay zero (zero Jacobian
  columns, zero diagonal entries).
* **Application** (`qgt_operator._column_block_product`): for sample chunks of at
  most `MATMUL_CHUNK_BYTES` (also bounding the `Z` chunk) `Z = J[s, p] Q[p]` +
  `psum` over `"params"`, then `Y[p] += q[J[s, p]^H Z]` locally, accumulated in row
  blocks of at most the same size; at the end a `psum` over `"samples"` (a no-op for
  `P_s = 1`). `Z` is never formed for all samples. `q = Re` / `i Im` with real
  blocks in real arithmetic (`Z = (A Q, B Q)` reduced in one `psum`, then
  `A^T Za + B^T Zb` resp. `A^T Zb - B^T Za`); diagonal terms as in `QGTOperator`.
* **Vectors**: `ShardedQGTOperator.__call__` / `.raw` take and return REPLICATED
  vectors or blocks (`(num_params,)` / `(num_params, r)`): the input is padded and
  split (no communication), the row-sharded result all-gathered by a sharding
  constraint and stripped. Explicit transposes through `linear_call` as in the
  existing operator, so CG (`jax.scipy.sparse.linalg.cg` and `solver.CG`), the
  L-BFGS loss (`_refinement_loss(..., layout=...)`), the true residual,
  `s_norm_fn` and `_get_tdvp_error` work with it unchanged (tested; the optimizer
  itself keeps `opt.qgt = QGTOperator.raw`).
* **Orthonormalization** (`rsvd._cholesky_qr`): shifted CholeskyQR3 (Fukaya et al.,
  SIAM J. Sci. Comput. 42, A477 (2020)). Gram matrix `G = psum_p(Y_p^H Y_p)`
  (replicated `r x r`), Cholesky of `G + s I` with
  `s = 11 (m r + r (r + 1)) u ||G||_F` (`||G||_F >= ||Y||_2^2`, the computable upper
  bound), local triangular solve `Q_p = Y_p L^{-H}` (donating `Y`), then two plain
  CholeskyQR passes. A plain pass whose Cholesky factor is not finite (`Q` still
  conditioned above `u^{-1/2}`, e.g. a numerically rank-deficient block, or
  `kappa(Y) > ~1/(u sqrt(11 m r))`, about 3e10 at the production size) is redone
  shifted, up to 6 passes in total (a warning if two plain passes were not reached).
  `JVMC_RSVD_DEBUG=1` prints the number of passes and `||Q^H Q - I||_F` after every QR.
  Tests: `||Q^H Q - I||_F < 1e-12` for singular values spanning 1 ... 1e-8 (real and
  complex) and for a rank-6 block with 9 columns.
* **Rayleigh-Ritz** (block-sharded since 2026-10-25, see 5.5): the final `Q` is
  padded with zero columns to `r_b`, a multiple of the device count; `B = Q^H (A Q)`
  is formed directly in blocks (`BLOCK_SPEC = P("params", "samples")`: device
  `(s, p)` holds rows `p`, columns `s`, `r_b / P_p x r_b / P_s`) by a ring
  reduce-scatter over the parameter groups (`_ritz_matrix_blocks`); the padding is
  decoupled by `2 ||B||_F + 1` on its diagonal. `"distributed"` hands the blocks to
  `jaxmg.syevd`, whose eigenvectors `W` come back in the same blocks;
  `"device"`/`"host"` gather `B` (symmetrized) and diagonalize it replicated, then
  slice `W` into the blocks. Ritz vectors `V_p = Q_p W` by a ring over the row blocks
  of `W` (`_ritz_vectors_blocks`, one `ppermute` per step), row-sharded; the
  selection of `_select` (which also drops the padded eigenpairs) is done row by
  row. `randomized_eigh` returns `V` padded, `(padded_num_params, k)`, row-sharded
  (stripping it would gather it).
* **Solver** (`RandomizedPinvSNR`): `V^H b` is a `psum` of local `k`-vectors
  (`_project_rows`); `PinvSNR.__call__` is reused unchanged: its
  `jnp.dot(V, pinvEv * V^H b)` partitions as a local GEMV with a row-sharded result
  (checked in the HLO: no collective), which is all-gathered and stripped into the
  replicated update. `info["spectrum"]` etc. are unchanged. `snr_mode="gaussian"`
  works; `snr_mode="empiric"` with `snr_tol != 0` raises `NotImplementedError` in
  this layout (it needs per-sample projections onto `V`). The L-BFGS refinement and
  the true residual use the sharded operator's vector products.

### 5.2 Memory per device

Production case: 94640 real parameters (complex Jacobian, `q = Re`, real blocks),
40000 samples, r = 16000, 8 GPUs, mesh `(1, 8)` (94640 = 8 x 11830, no padding),
`MATMUL_CHUNK_BYTES` = 1 GiB. Derived from the shapes in the code (GB = 1e9 bytes):

| object | shape per device | size |
|---|---|---|
| sample-sharded Jacobian (`SampledObs`, complex128) | 5000 x 94640 | 7.6 GB |
| column-sharded copy (complex128, per solve) | 40000 x 11830 | 7.6 GB |
| one block (`Q`, `Y`, `A Q`, `V`; float64) | 11830 x 16000 | 1.5 GB |
| streamed `Z` chunk (`(A Q, B Q)`, float64) | 2 x 4194 x 16000 | 1.07 GB (+ the same for the all-reduce) |
| Jacobian chunk + its real/imag parts | 4194 x 11830 complex + 2 real | 0.8 + 0.8 GB |
| Gram matrix / Cholesky factor (during a QR, `qr_mode="cholesky"`) | 16000 x 16000 float64 | 2 GB each |
| Rayleigh-Ritz `B`, eigenvectors `W`, `"distributed"` (5.5) | 2000 x 16000 float64 blocks | 0.26 GB each (+ jaxmg workspace outside the pool) |
| the same, `"device"` | 16000 x 16000 float64, replicated | 2 GB each (+ eigh workspace) |

The chunk has `min(2^30 / (11830 * 16), 2^30 / (2 * 16000 * 8)) = 4194` rows. At
most two blocks are alive at once (input and output of an application; the QR
donates its input), so an application peaks at about 7.6 + 7.6 + 2 x 1.5 + 2.1 +
1.6 = 22 GB, the Rayleigh-Ritz stage at about 7.6 + 7.6 + 1.5 (`Q`) + 2 + 2 + the
eigh workspace. The `"samples"` layout needed 7.6 GB plus four 12 GB blocks. In
general, per device: Jacobian copies `2 x 16 N n / P` bytes, blocks `8 n r / P_p`
each, `Z` bounded by the chunk budget, and the `r x r` matrices `8 r^2` each
when replicated (CholeskyQR's Gram matrix and factor, `B` and `W` with the
`"device"` diagonalization) or `8 r^2 / P` when block-sharded (5.5). The redistribution itself briefly
needs the output copy next to the input (plus a padded copy of the local shard
if `num_params` is not divisible by `P_p`). For `P_s > 1` the blocks are
`n / P_p` rows (larger) and the Jacobian copy `N / P_s x n / P_p`.

### 5.3 Communication per application

| collective | `(1, P)` | `(P_s, P_p)` |
|---|---|---|
| `psum` of `Z` over the parameter group | `N x r` values (complex, or two real) in chunks: 10.2 GB for the production case | `N / P_s x r` per group |
| `psum` of `Y` over the sample group | none | `n / P_p x r` |
| redistribution (once per solve) | all-to-all of the Jacobian shard (7.6 GB, 7/8 of it leaves the device) | within each sample group |
| CholeskyQR (per pass, 3 per QR) | all-reduce of the `r x r` Gram matrix, 2 GB | over `P_p` |
| Rayleigh-Ritz matrix (ring reduce-scatter) | `P - 1` steps of one `r / P x r` block, `r^2 / P` in total per device | `P_p - 1` steps of `r / P_p x r / P_s` |
| Ritz vectors (ring) | `P - 1` steps of one `r / P x r` block of `W` | the same, plus a gather of the `V` columns over the sample group |
| vector product (`A(x)`, L-BFGS) | `N` values + all-gather of `n` | |

The all-reduced volume per application (10.2 GB of `Z`) is comparable to the
12 GB `(num_params, r)` all-reduce of the `"samples"` layout, at the same flops
(each GPU does `1/P` of both GEMMs). New per QR are three 2 GB all-reduces,
replacing a redundant Householder QR of the full `(94640, 16000)` block on every
GPU (about 4e13 flops each) by about 3e13 flops of GEMM/TRSM split over the GPUs.

### 5.4 Tests and status

`tests/rsvd_test.py` (new classes `TestShardedQGTOperator`, `TestCholeskyQR`,
`TestRandomizedEighParameterLayout`, `TestRandomizedPinvSNRParameterLayout`), on 1
and 4 emulated CPU devices, mesh shapes `(4, 1)`, `(1, 4)`, `(2, 2)` (on one
device `(1, 1)`): the operator against the dense matrix for the three reductions
(real and complex vectors and blocks, with and without diagonal terms, with a
64-byte chunk budget, `num_params` not divisible by 4), the redistributed copy,
the row-sharded block kernel for all four apply modes, CG, gradients through the
vector product; CholeskyQR3 orthogonality; `randomized_eigh` exact at full rank,
low-rank eigenpairs, `"LA"`, diagonal terms, the distributed padding, agreement
with the sample layout; `RandomizedPinvSNR(layout="parameters")` step for step
equal to dense `PinvSNR` (holomorphic reference evolution), equal to the dense
update for both non-holomorphic reductions (with 30 parameters, i.e. padding),
low-rank solve equal to the dense one, S-norm and TDVP error unchanged (also when
computed with the sharded operator), L-BFGS refinement, Gaussian SNR, empiric SNR
rejected. The experiment script has `simulation.rsvd.layout` (default
`"parameters"`) and `simulation.rsvd.mesh_shape` (`[P_s, P_p]`, `null` = `[1, n_GPUs]`);
`scripts/0encap_folder/rsvd/test/` (running experiment) was not touched.

Results on 2026-10-23 (CPU, jax 0.9.1): `rsvd_test`, `cg_test`, `tdvp_test`,
`sr_test` -- 57 passed on one device and 57 passed on four emulated devices. The
experiment script ran 3 TDVP steps on four emulated devices with a tiny
configuration (L = 4, 2000 samples, rank 30 + 4) for `layout = "parameters"` with
mesh `(1, 4)` (+ L-BFGS) and `(2, 2)`, and for `"samples"`; true residuals
~1e-14 in all three. The layout is not part of the output file name (it changes
the numerics, not the physics), so a run with another layout resumes the same
checkpoint.

Not done yet: a run on GPUs (timings and the measured peak memory at r = 16000).

### 5.5 Block-sharded Rayleigh-Ritz step (2026-10-25)

The limits scans (`scripts/0encap_folder/rsvd/limits_scan`, `limits_scan_16gpu`)
showed that with 5.1-5.4 the `(n, r)` blocks and the Jacobian scale with the GPU
count but the `r x r` objects did not: `B`, `V_small` (replicated before
`Q_p V_small`), CholeskyQR's Gram matrix and factor, and the eigensolver
workspace, 8 GB each per GPU at r = 32000. Now:

* **`B`** is formed block-sharded by a ring reduce-scatter (`_ritz_matrix_blocks`):
  in step `k` each device adds `Q_p[:, t]^H (A Q)_p[:, s]` for the block
  `t = p - k - 1` it received and passes it on; at most two blocks of
  `r_b / P_p x r_b / P_s` are alive, no `r x r` partial sum is formed. No psum over
  the sample groups is needed (`Q`, `A Q` are replicated over them). Not
  symmetrized on this path: `B` is Hermitian up to round-off and the eigensolvers
  read one triangle.
* **Block layout `P("params", "samples")`**: jaxmg (checked in its source,
  `_cusolvermp_layout.py`) shards each matrix dimension over a single mesh axis and
  needs its process grid to cover the whole mesh. Rows over the parameter groups and
  columns over the sample groups satisfies both on any `(P_s, P_p)` mesh, and every
  device's block is a local slice of the row-sharded `B`; for the default `(1, P)`
  it is the plain row sharding over all devices.
* **`syevd`**: `diagonalization_mode="distributed"` calls
  `jaxmg.syevd(B, T_A, mesh, matrix_specs=P("params", "samples"))` on the blocks
  (`B` donated); eigenvalues replicated, `W` in the same blocks, never replicated.
  `T_A` defaults to the gcd of the local block sizes (256 with jaxmg's padding if
  that is below 128). Falls back to `"device"` with a warning where jaxmg cannot be
  used (not installed, no GPU, one device, `JVMC_USE_DISTRIBUTED` unset).
  `"device"`/`"host"` gather `B`, symmetrize and diagonalize replicated as before,
  then slice `W` into the blocks: both paths feed the same ring.
* **Ritz vectors**: `V_p[:, s] = sum_k Q_p[:, k] W[k, s]` with the row blocks of `W`
  travelling around the ring of parameter groups (`_ritz_vectors_blocks`); for
  `P_s > 1` the column blocks of `V` are gathered over the sample group at the end
  (`V` is replicated there anyway). Compiled HLO: only `collective-permute`, plus a
  scalar all-reduce for the padding norm and that gather.
* **`qr_mode`** (`RandomizedPinvSNR(qr_mode=...)`, `randomized_eigh(qr_mode=...)`):
  `"cholesky"` (default) is the shifted CholeskyQR3 of 5.1 with its replicated Gram
  matrix and Cholesky factor; `"distributed"` is `jaxmg.qr` on the row-sharded block
  (`P("params", None)`, `T_A` passed through; the triangular factor comes back
  row-sharded and is dropped; padding rows re-zeroed). It needs a `(1, P)` mesh
  (jaxmg's grid must cover the mesh and the block's columns are not divisible by
  `P_s` in general) and falls back to `"cholesky"` with a warning otherwise, or where
  jaxmg cannot be used. The debug line of every QR names the method used.

Still replicated, by design: the eigenvalues (`r`), the `k`-vectors of the solve
(`V^H b`, the regularized inverse eigenvalues, the SNR), the update `x` (`n`) and
everything the optimizer holds as before; with `qr_mode="cholesky"` the Gram
matrix and Cholesky factor of every QR pass; with `"device"` diagonalization `B`
and `W` while they are diagonalized.

Per device at r = 32000 on 8 GPUs, `(1, 8)`: `B` and `W` blocks 4000 x 32000 x 8 B =
1.0 GB each instead of 8.2 GB replicated; with both options distributed no
`r x r` object is replicated.

**jaxmg memory.** cuSOLVERMp's workspace (for `syevd` and `qr`) is a `cudaMalloc`
outside the JAX memory pool: on 16 GPUs the points (377k, r >= 24000) and (493k,
r >= 24000) died in `syevd` once the pool had grown past ~55 GiB. Runs that use
`"distributed"` diagonalization or QR should set `XLA_PYTHON_CLIENT_MEM_FRACTION`
to about 0.8 rather than 0.95, so that the workspace fits next to the pool.
