# Claude Interaction — Steps Taken

## Interaction 1

Summary of the first working session with Claude on this repository, captured in
commit `da11f3c` *"First iteration with Claude: organizing and documenting code +
initiating uv project"* (2026-07-09).

### Starting point

The repo held a set of research scripts exported from Jupyter notebooks (Weyl-coordinate
Schwarzschild + Morgan-Morgan disk ray tracers, plus a self-contained Kerr script),
plus a `LICENSE`, a one-line `README.md`, and a SLURM `my_job.sbatch`. There was no
package manifest, no environment setup, and no documentation of how the scripts fit
together.

### Steps taken

1. **Initiated a `uv`-managed project.**
   - Added `pyproject.toml` (`black-holes-shadows`, Python `>=3.14`, deps: numpy,
     scipy, matplotlib, numba).
   - Pinned the interpreter with `.python-version` (`3.14`).
   - Generated `uv.lock` so the environment is reproducible via `uv sync`.

2. **Added a `.gitignore`** scoped to how these scripts actually behave:
   - Python cruft (`__pycache__/`, `*.pyc`, `*.pyo`, `*.egg-info/`), the `.venv/`.
   - Generated figures/data (`*.png`, `*.pdf`, `*.svg`) — with an explicit note that
     `np.savetxt` matrix outputs (`Mat`, `Mz`, `Mthet`, `Mphi`, `Mat_nu_disk*`,
     `Mat_constA_*`) and some `plt.savefig` figures have no predictable extension and
     must be checked manually in `git status` before committing.
   - SLURM output (`*.out`, `*.err`, `job_test*`), OS/editor cruft (`.DS_Store`,
     `.vscode/`, `.idea/`), and the machine-specific `my_job.sbatch` (kept local only).

3. **Documented the codebase.**
   - Rewrote `README.md` from a one-line stub into a full description: the two
     spacetimes modelled, `uv` setup instructions, and — most importantly — a
     **Pipeline** section explaining the execution order and role of every script:
     - `generate_matriz.py` tabulates the lambda-potential lookup table (needed
       because lambda(rho, z) only has a closed form on the symmetry axis).
     - The ray tracers (`test_Z_SHADOW.py`, `test2_Z_SHADOW.py`,
       `test_parallel_SHADOW.py` as the production/numba-parallel version, and
       `test_symmetry_lensing.py`) consume that table and emit `Mat`/`Mz`/`Mphi`.
     - The plotting scripts (`symmetry.py`, `simetria_shadow.py`,
       `simetria_shadow_v2.py`, `lensing_image.py`) load the saved matrices into figures.
     - `lensing_r_theta.py` documented as the fully self-contained Kerr family.
   - Added a **Notes on provenance** section flagging the notebook origin (`# In[NN]:`
     cell markers, preserved dead/legacy code) and the helper functions duplicated
     near-verbatim across files (tagged `#Duplicated` in the source).
   - Added inline documentation/organization across the script files themselves
     (`generate_matriz.py`, `lensing_image.py`, `lensing_r_theta.py`,
     `simetria_shadow*.py`, `symmetry.py`, `test*_SHADOW.py`, `test_symmetry_lensing.py`).

4. **Committed** everything as the "First iteration with Claude" commit
   (15 files changed, ~4130 insertions / ~2352 deletions).

### Result

The repo went from a loose bag of notebook exports to a reproducible, documented
`uv` project: `uv sync` sets up the environment, the README explains the ray-tracing
pipeline end to end, and generated data is kept out of version control (with the
manual-review caveat noted above).

## Interaction 2

Compared the two serial ray tracers and removed the redundant one.

### Steps taken

1. **Diffed `test_Z_SHADOW.py` against `test2_Z_SHADOW.py`.** They define the
   exact same set of functions; `test2` is a *de-generalized* copy of `test`:
   - `test_Z_SHADOW.py` keeps the family-standard `nu(…, m)` component selector
     (m=0 BH-only / m=1 disk-only / m=2 sum), and that generality propagates
     through `derNU`, `dlamb`, `dlamb2`, `lamb`, `gtt/grr/gzz/gpp` and `geo`.
   - `test2_Z_SHADOW.py` strips the `m` selector, hard-wiring the BH+disk
     (m=2) case, and inlines `nuD` inside `nu`.
   - Remaining differences are run-configuration only, not logic: `M/MD`
     (`1.0/0.0` vs `0.1/0.9`), matrix file (`Mat_nu_disk0.0` vs `…0.9`),
     integrator `tol` (`1e-4` vs `1e-3`), and active vs commented `np.savetxt`
     / per-pixel timing.

2. **Confirmed nothing needed to be copied from `test2` into `test`.** Every
   definition in `test2` already exists in `test` in a more general form, so
   `test` is effectively a superset — `test2` carried no unique logic.

3. **Deleted `test2_Z_SHADOW.py`** as a redundant specialization. Its disk run
   is reproducible from `test_Z_SHADOW.py` by setting `M=0.1, MD=0.9`, pointing
   at `Mat_nu_disk0.9`, and un-commenting the save lines.

4. **Updated `README.md`** to drop the deleted script from the pipeline list.

### Notes

- Provenance `#Duplicated: … test2_Z_SHADOW.py` references still linger in the
  docstrings/comments of the sibling scripts (`generate_matriz.py`,
  `test_parallel_SHADOW.py`, `test_symmetry_lensing.py`, etc.). These are
  historical notes and were left untouched.
- A latent bug shared by the whole family remains: in `func`, `yf` is only
  assigned inside the `while` loop, so a ray failing the loop condition on the
  first check would hit `return (yf)` unbound. Not addressed here.

## Interaction 3

Compared the rest of the core-sharing ray-tracing family against a single basis
via an AST diff of the *live* functions; verdict was KEEP for all three.

### Method

Chose **`test_parallel_SHADOW.py`** as the comparison basis — every sibling's
docstring names it the canonical/most-complete copy, and it is the only file with
the full tracer set plus true `prange` parallelism. (Distinct from the *pipeline*
base `generate_matriz.py`, which merely runs first.) An `ast`-based tool extracted
each file's top-level `def`s — dead code lives inside triple-quoted strings, so it
is skipped automatically — normalized away docstrings and formatting, and diffed
signatures + bodies against the basis.

### Results (one sub-task per comparison)

1. **`generate_matriz.py` → KEEP.** A strict *subset* of helpers (15 of the basis's
   tracer functions absent) plus one function live *only here*: `lamb_Mat`, the
   quadrature that produces the `Mat_nu_disk*` lookup table every other script
   loads. It is the upstream producer; deleting it breaks the pipeline. Not a ray
   tracer, so not redundant with the basis.

2. **`test_symmetry_lensing.py` → KEEP.** 29/32 functions identical to the basis;
   the meaningful diffs are `func`/`geo` carrying a 5th state component **phi** and
   returning `[rho, z, phi]` / emitting `Mphi` — a gravitational-lensing image, a
   genuinely different output from the binary shadow. Nothing to merge.

3. **`test_Z_SHADOW.py` → KEEP** (decided after the diff). Its physics core differs
   from the basis in exactly one live behaviour: the disk-crossing branch in `func`
   (`rho > b` + z sign flip → `+50.0` z tag). The basis *already contains this exact
   branch commented out* and disables it deliberately to stay a pure BH-shadow
   tracer — so enabling/merging it would change classification for disk-crossing
   pixels, i.e. alter the basis's existing output. Because a merge is not
   output-preserving, `test_Z_SHADOW.py` is a distinct analysis mode
   (disk-crossing-enabled, serial), not a safe-to-fold duplicate. (Contrast
   `test2`, deleted in Interaction 2, which carried *no* unique live behaviour.)

### Notes

- No files were deleted or merged this pass; README pipeline list is unchanged.
- Incidental finding, not fixed (out of scope): `generate_matriz.py`'s `xi2` wraps
  the sqrt argument in `np.abs(...)`, while the basis `xi2` does not — a latent
  numerical inconsistency between the table producer and the interpolation consumers.
- The `geo` "difference" flagged for `test_Z_SHADOW.py` and `generate_matriz.py` was
  only a trailing dead string literal in the basis's `geo`; the live bodies match.

## Interaction 4

Extracted the `#Duplicated` physics core (identified across Interactions 1-3) into
a new shared module, `weyl_core.py`, and updated the four consuming scripts to
import from it instead of carrying their own copies.

### What moved

31 functions (30 physics helpers + a new `load_matrix` loader) moved into
`weyl_core.py`: numerical utilities (`simps`, `derivative`, `run_kut4_mod`),
geometry/potentials (`d1`, `d2`, `xi2`, `nuD`, `nu`, `lambSch`), metric
components and photon momenta (`gtt`/`grr`/`gzz`/`gpp`, `zeta`,
`dthe`/`dr`/`Pphi`/`Pt`/`dphi`/`dt`, `derNU`, `dlamb`, `dlamb2`, `lamb`), and the
non-jitted observer-frame variants (`d1_i`, `d2_i`, `xi2_i`, `nuD_i`, `nu_i`,
`gpp_i`). Each script keeps its own `geo`, `func`, driver, and (where applicable)
`f_paral`/`lamb_Mat` — these are the genuine variant layer (shadow vs. lensing
output, disk-crossing branch, serial vs. parallel, the quadrature-based table
builder) and were deliberately not merged, matching the Interaction 3 findings.

### Two intentional changes made during extraction

1. **`xi2` canonicalized to the `np.abs`-guarded form** (previously only in
   `generate_matriz.py`; the three tracers lacked the guard). User-confirmed
   choice: matches how the lookup table was actually built and avoids NaN for a
   radicand that goes slightly negative near the disk edge.
2. **Decorators unified to plain `@jit(nopython=True)`** for scalar leaf helpers
   (was `parallel=True` in `test_parallel_SHADOW.py`, a no-op there). **Exception:
   `run_kut4_mod` keeps `@jit(nopython=True, parallel=True)`** — it is a
   higher-order function that receives another jitted function (`geo`) as a
   callback, and `test_parallel_SHADOW.py`'s `geo`/`func` still carry
   `parallel=True` (kept as-is, per the variant-layer rule); pairing a
   plain-nopython `run_kut4_mod` with that parallel `geo` callback triggered
   numba's workqueue threading layer to abort ("Concurrent access has been
   detected") — confirmed via direct testing, not a hunch. Restoring
   `run_kut4_mod`'s original `parallel=True` (its historical pairing) fixed it,
   and is a no-op for the two callers whose own `geo`/`func` are plain nopython.

### Verification

- **Static:** wrote an AST-diff tool comparing every function now in
  `weyl_core.py` against its pre-refactor body captured from `git show HEAD:...`.
  29/31 byte-identical; the 2 differences are the intended `xi2` change and one
  incidental dead-code removal in `run_kut4_mod` (a stray triple-quoted legacy
  string statement with zero runtime effect, dropped during the move).
- **Dynamic smoke test:** imported `weyl_core.py` standalone and called every
  jitted function (with a synthesized dummy `Mat_nu` for `lamb` and its
  dependents) to force numba compilation — all succeeded.
- **Driver-core integration test:** since these scripts execute their driver at
  import time, wrote a harness that AST-extracts just `geo`/`func` (+imports)
  from each refactored file, execs them in a fresh namespace, and drives one
  full `func()` call (exercising the whole `run_kut4_mod` -> `geo` -> weyl_core
  call graph) — passed for all three tracers.
- **Pre-existing issues found and ruled out of scope** (confirmed identical on
  the pre-refactor originals via `git stash`/`git show HEAD:...`, so not
  refactor regressions):
  - The real end-to-end driver run (`fsolve` -> `gpp_i` -> `math.log` on a
    0-d array) fails under this repo's currently pinned numpy/scipy versions,
    in both old and new code.
  - `generate_matriz.py`'s `lamb_Mat` (bare `@jit`, calls `scipy.quad`) fails to
    compile under the currently pinned numba version, in both old and new code
    — numba dropped automatic object-mode fallback for un-parameterized `@jit`.
  - `test_parallel_SHADOW.py`'s `f_paral` (the actual `prange` parallel path)
    hits numba's "nested parallel region" limitation when it calls
    `func` -> `run_kut4_mod`(parallel) -> `geo`(parallel) from inside its own
    `prange` loop — reproduced identically on the pre-refactor original, so
    this file's "true parallelism" was already non-functional under the
    current numba/environment combination before this refactor.

### Wrap-up

- Updated `README.md`'s Pipeline and "Notes on provenance" sections to describe
  `weyl_core.py` and point to this entry.

## Interaction 5

Removed the dead notebook-export cruft from the three Weyl-family ray tracers
(`test_parallel_SHADOW.py`, `test_symmetry_lensing.py`, `test_Z_SHADOW.py`),
reversing the earlier "preserve for reference" decision from Interaction 1 now
that the live behaviour of each file has been mapped out (Interactions 3-4).

### What was removed

- Large triple-quoted legacy blocks: earlier pixel-classification attempts,
  dead lensing/diagnostic plotting code, and — in `test_symmetry_lensing.py` —
  the never-executed mirroring/reconstruction block.
- `# In[NN]:` Jupyter cell markers.
- Unused imports left over from the dead blocks: `cmath` and
  `scipy.integrate` in all three files; `matplotlib.pyplot`/`matplotlib.colors`
  in `test_parallel_SHADOW.py` and `test_symmetry_lensing.py` (matplotlib was
  never live in either — only referenced inside the deleted blocks).
  `test_Z_SHADOW.py` keeps its matplotlib imports since its inline
  `plt.show()` figure is live code.
- Dead variables that fed only the deleted blocks: the orphaned `M2`/`Mat2`/
  `Mz2` allocations sitting before the removed blocks in
  `test_parallel_SHADOW.py`; the unused `it` loop counter and a duplicate
  `hder = 10**-6` assignment in `test_symmetry_lensing.py`; the never-populated
  `Mphi` allocation in `test_Z_SHADOW.py` (its `func` only ever returns
  rho/z, never phi).

### What was kept

All live behaviour is untouched: `prange` parallelism and the `f_paral`
driver in `test_parallel_SHADOW.py`; the 5-component state carrying `phi` and
the `Mphi` output in `test_symmetry_lensing.py`; the `+50.0` disk-crossing
branch and the inline matplotlib figure in `test_Z_SHADOW.py`. The `#NOTE:`
docstring caveat about `func`'s possibly-unbound `yf` was left as-is — it
documents live behaviour, not dead code.

### Docs updated

Rewrote each file's module docstring to drop the "commented blocks/cell
markers preserved for provenance" language, and updated `README.md`'s "Notes
on provenance" section to say the cruft has been removed rather than kept in
place (history remains recoverable via git).

### Verification

`python -m py_compile` on all three files, plus a manual grep pass confirming
no references remain to the removed imports (`cmath`, `scipy.integrate as sci`,
`matplotlib`/`plt.` in the two files where it's no longer imported) or to
`Mphi` in `test_Z_SHADOW.py`.

## Interaction 6

Split the flat `weyl_core.py` (from Interaction 4) into a `general_methods/`
package, one file per concern, and added convenience classes around the
metric functions.

### What moved

`weyl_core.py` became `general_methods/`:
- `mathematical_formulas.py` — `simps`, `derivative`, `run_kut4_mod`,
  `load_matrix`, `lamb` (+ the `Mat_nu` module global).
- `physical_quantities.py` — `d1`, `d2`, `xi2` and their non-jitted `_i`
  observer-frame copies.
- `physical_potentials.py` — `nu`, `nuD`, `lambSch`, `derNU`, `dlamb`,
  `dlamb2` and their `_i` copies.
- `spacetime_metrics.py` — `gtt`, `grr`, `gzz`, `gpp`, `gpp_i`, `zeta`,
  `Pphi`, `Pt`, `dthe`, `dr`, `dphi`, `dt`, still as module-level free
  functions, plus two new classes: `Metric` (gtt/grr/gzz/gpp/zeta/Pphi/Pt)
  and `MetricDerivatives` (dthe/dr/dphi/dt), each bound to fixed `(M, MD, b)`
  and exposing `(rho, z, ...)`-only methods.
- `__init__.py` re-exports every submodule's public names and carries
  forward the original module docstring's canonicalization notes (`xi2`
  guard, `run_kut4_mod`'s `parallel=True` requirement).

No functions were deleted. The `_i` observer-frame chain was the only
candidate for "erase non-jit duplicates of jitted functions" and was
deliberately kept — user-confirmed, since `gpp_i` (its only externally-called
member) exists specifically so `scipy.optimize.fsolve` has a non-jitted
target, not as dead duplication.

### Why `gtt`/`grr`/`gzz`/`gpp`/`lamb`/`nu`/`derNU` stayed as free functions

Investigated call sites in the four consumers before wrapping anything in a
class: `gpp`, `lamb`, `nu`, and `derNU` are called by bare name *inside* the
`@jit(nopython=True)` `geo`/`func` bodies of `test_parallel_SHADOW.py`,
`test_Z_SHADOW.py`, and `test_symmetry_lensing.py`. Numba nopython mode
can't call Python instance methods, so those four had to remain module-level
jitted functions — only `gtt`/`grr`/`gzz` (never called this way) were free
to be wrapped without risk. `Metric`/`MetricDerivatives` are therefore thin,
non-jitted wrappers that delegate to the free functions, satisfying the
"class for metrics" ask without touching jit compatibility.

### Consumer script updates

`generate_matriz.py`, `test_parallel_SHADOW.py`, `test_Z_SHADOW.py`,
`test_symmetry_lensing.py`: `import weyl_core` / `from weyl_core import *` /
`weyl_core.load_matrix(...)` renamed to `general_methods` throughout
(including docstring references). `weyl_core.py` deleted.

### Verification

- Imported `general_methods` standalone and called every jitted function
  plus both new classes (`Metric`, `MetricDerivatives`) with a synthesized
  dummy `Mat_nu` — all executed correctly.
- Re-ran the same `fsolve`/`lamb_Mat` code path `generate_matriz.py` uses
  (with a couple of sample points instead of the full 1200x1200 grid) and
  hit the same two pre-existing environment issues already documented in
  Interaction 4 (`fsolve`→`math.log` on a non-scalar array; bare `@jit` +
  `scipy.quad` no longer object-mode-falls-back under the pinned numba) —
  confirmed identical on the pre-refactor `weyl_core.py` via `git stash`, so
  not regressions from this split.
- `ast.parse` syntax check on all four updated consumer scripts.

## Interaction 7

Extracted the duplicated mirroring/classification logic from the three
plotting/post-processing scripts into a new shared module,
`shadow_postprocess.py`, and cleaned up remaining internal dead code left
in `generate_matriz.py` and the three plotting scripts. This does for the
post-processing layer what Interaction 4 did for the ray-tracer physics
core.

### What moved

New module **`shadow_postprocess.py`** with two vectorized, side-effect-free
numpy functions:
- **`mirror_quadrants(A, antisymmetric=False)`** — reflects a ray-traced
  quarter-plane array across both axes into the full 2N x 2M plane
  (top-left = A, top-right/bottom-left/bottom-right mirrored accordingly);
  `antisymmetric=True` negates the z-mirrored bottom half, for quantities
  like `Mz` that flip sign under z -> -z.
- **`classify_shadow(mat, mz, b, capture_r=0.002, mz_escape=49.0,
  unshift_mat=False, use_abs_z=True)`** — classifies each pixel as
  captured (1) / beyond-disk (2) / neither (0), with `unshift_mat` and
  `use_abs_z` flags to reproduce two real differences between the
  scripts' original conventions (a +50 wraparound offset in one family's
  `mat` output; whether the z-escape test uses `abs(mz)` or the raw
  signed value) rather than silently "fixing" them into one behaviour.

`simetria_shadow.py`, `simetria_shadow_v2.py`, and `symmetry.py` — each of
which previously hand-rolled its own nested-loop mirroring and
classification — were updated to call these two helpers instead.

### `symmetry.py` physics-helper deduplication

`symmetry.py` also carried its own full copy-pasted set of physics helpers
(`d1`, `d2`, `xi2`, `nuD`, `nu`, `gpp` and the non-jitted `_i` variants) —
the same duplication `weyl_core.py`/`general_methods` already solved for
the ray tracers and `generate_matriz.py` (Interactions 4 and 6). These were
deleted from `symmetry.py` and replaced with `import general_methods` /
`from general_methods import *`, matching the pattern already used by
`generate_matriz.py`.

### `generate_matriz.py` dead-code removal

Removed two legacy commented-out earlier versions of `lamb_Mat`, a
commented-out alternate `np.savetxt` call, and five now-unused imports
(`math`, `matplotlib.pyplot`, `matplotlib.colors`, `time`, `cmath`).

### Other dead code removed

Assorted disabled/commented-out alternates in the three plotting scripts,
in the same spirit as Interaction 5's cleanup of the ray tracers:
disabled alternate `plt.figure`/colormap/label/tick_params/savefig calls
in `simetria_shadow.py`; a commented circle-patch block plus disabled
alternates in `simetria_shadow_v2.py`; and two large legacy classification
blocks plus commented alternates in `symmetry.py`.

### Verification

No behavior changed anywhere. Verified via numeric-equivalence testing
against pre-refactor logic: the old manual mirroring/classification loops
from each of the three scripts were compared against the new
`mirror_quadrants`/`classify_shadow` calls on synthesized test arrays.

### Docs updated

Updated `README.md`'s Pipeline section to mention `shadow_postprocess.py`
alongside the plotting scripts.

## Interaction 8

Made the Schwarzschild (Weyl-coordinate, `MD=0`) branch of the pipeline
actually runnable end-to-end, fixing the environment/numba blockers flagged
as out-of-scope in Interaction 4 and left as "Suggested next steps" above,
and added a new orchestrator script, `test_run_1.py`.

### Why this pipeline, and why now

Interactions 1-7 refactored the code into shared modules, but no single run
had produced a shadow image since those interactions began — the three
pipeline stages (`generate_matriz.py`, one ray tracer, one plotting script)
each executed their driver at import time with no shared entry point, plus
mismatched filenames/configs between stages, plus the three real blockers
below. Schwarzschild (`MD=0`) was chosen as the first end-to-end target
because `test_Z_SHADOW.py` and `symmetry.py` already shared that config
(`M=1.0, MD=0.0, b=6.0`), and the serial ray tracer sidesteps the
parallel-only nested-`prange` blocker entirely.

### Blockers fixed

1. **`fsolve` → `gpp_i`/`nu_i` → `math.log` on a numpy array.** `fsolve`
   passes its trial point as a length-1 array; `math.log` (called deep inside
   `nu_i` via `d1_i`/`d2_i`) rejects non-scalar input. Fixed by unwrapping to
   a Python scalar at the call site: `lambda R: ... gpp_i(R[0], ...)`.
2. **`float(fsolve(...))` under the pinned numpy (2.4.6).** Separately from
   (1), reading `fsolve`'s (1,)-shaped return value back out via
   `float(R_solution)` now raises `TypeError: only 0-dimensional arrays can
   be converted to Python scalars` — numpy tightened this conversion to
   require true 0-d arrays. Fixed via `float(R_solution[0])`. Found only
   after fixing (1) let the run reach this line; not identified by static
   analysis, only by actually executing the pipeline.
3. **`generate_matriz.py`'s `lamb_Mat`: bare `@jit` around `scipy.quad`.**
   Numba has no object-mode fallback for un-parameterized `@jit` under the
   pinned version, and `scipy.quad` can't compile in nopython mode. The
   decorator was providing no benefit, so it was removed outright.

`test_parallel_SHADOW.py`'s nested-`parallel=True` limitation (the third
blocker from Interaction 4) was left unfixed — deliberately out of scope,
since the serial Schwarzschild path doesn't hit it.

### Driver refactor (plan-approved, user chose "refactor drivers + orchestrate"
over a self-contained duplicate script)

Each stage's import-time driver was wrapped in a callable function,
parameter-driven instead of hardcoded, with the original module-level
behaviour preserved under `if __name__ == "__main__":`:

- **`generate_matriz.py`** → `generate_lambda_matrix(M, MD, b, n, out_path)`,
  returning (and saving) the tabulated matrix. The dead `fsolve`/`gpp_i`
  block (computed `rho0` but never used it) was dropped in the same pass.
- **`test_Z_SHADOW.py`** → `trace_shadow(M, MD, b, n, matrix_path)`,
  returning `(Mat, Mz, alfa, beta)`. Also closed the long-standing
  possibly-unbound-`yf` note from Interaction 2/5 (`func` now initializes
  `yf = [y[0], y[2]]` before the loop) since it was touched anyway.
- **`symmetry.py`** → `render_shadow(Mat, Mz, M, MD, b, out_path)`, taking
  the quarter-plane arrays directly instead of requiring `Mat`/`Mz` on disk
  first, so it composes with `trace_shadow`'s return value. Its `fsolve`
  call was dropped entirely — `rho0` was only ever printed, not used by the
  plot.

No geodesic, classification, or mirroring logic was duplicated; the new
functions call the same `geo`/`func`/`shadow_postprocess` code already
established in Interactions 4-7.

### New `test_run_1.py`

Orchestrates the three functions above at a small/fast resolution (200x200
lambda matrix, 40x40 shadow grid before quadrant-halving) chosen so a first
sanity run completes in well under a minute rather than the multi-hour
1200x1200/80x80 production sizes. Writes the matrix outputs
(`Mat_nu_disk0.0`, `Mat`, `Mz`) to a new `test_run_1_matrices/` directory
(added to `.gitignore`) and the shadow figure to `image_generation/`.

### Verification

Ran `uv run python test_run_1.py` to completion: all three stages executed
without error, produced a 376-pixel captured-shadow region (real count from
`classify_shadow`'s `M2 == 1` mask, not read off the plot), and saved
`image_generation/schwarzschild_shadow.png` showing a clear dark shadow disk
against the lighter escaped-ray background — visually inspected and
confirmed physically sensible.

### Known cosmetic issue, not fixed

`symmetry.py`'s plot uses a 2-color `ListedColormap` (`c_map` has 2 entries)
to render `classify_shadow`'s 3-valued `{0, 1, 2}` output, which — because
`MD=0` still runs the `b`/beyond-disk comparison even with no disk present —
produces faint banding rings inside the rendered shadow. This is a
pre-existing quirk of `symmetry.py`'s original color mapping (predates this
interaction), not a defect in the underlying classification (the printed
pixel count is correct); left as-is rather than silently changed, matching
this repo's established "preserve real behavioural differences, don't
silently fix them" convention (see Interaction 3, `classify_shadow`'s
`unshift_mat`/`use_abs_z` flags in Interaction 7).

## Interaction 9

Branch: `numba-optimization` (created from `main`).

### Problem

In `test_Z_SHADOW.py`, numba never covered the per-pixel work of
`trace_shadow`. The function is plain Python and its double loop over emission
angles ran in the interpreter, merely dispatching into the jitted `func` once
per pixel -- so the loop itself (array construction, `dr`/`dthe` calls, control
flow) stayed uncompiled. The setup also mixes in things numba cannot compile in
nopython mode: the `func_initial` `lambda`, the `scipy.optimize.fsolve`
observer-rho lookup, `general_methods.load_matrix`, and `time`/`print`.

### Steps taken

1. Split `trace_shadow` into three functions:
   - `_solve_observer_rho(M, MD, b, z0)` -- plain Python; isolates the
     numba-incompatible `lambda` + `fsolve` root find, returns `rho0`.
   - `_trace_grid(rho0, z0, M, MD, b, alfa, beta, hder)` --
     `@jit(nopython=True)`; the extracted pixel loop that builds each photon's
     initial state (`dr`/`dthe`) and calls the jitted `func`, populating and
     returning `(Mat, Mz)`. This is the only new jitted piece and lets numba
     compile the whole loop instead of just the per-call `func`.
   - `trace_shadow(...)` -- unchanged signature and `(Mat, Mz, alfa, beta)`
     return; loads the matrix, times, builds the angle grids, calls the two
     helpers, prints.
2. Updated the module docstring to document the split and the JIT-coverage
   motivation.

### Verification

- Structural: `trace_shadow` and `_solve_observer_rho` are plain Python;
  `_trace_grid` is a numba Dispatcher.
- End-to-end smoke test: generated a coarse lambda matrix (`n=120`) and ran
  `trace_shadow(n=20)` -- printed `rho0`≈13.96 and elapsed time, returned finite
  `(10, 10)` `Mat`/`Mz` with 22 captured shadow pixels (unchanged from the
  pre-split loop, which was copied verbatim). Crucially `_trace_grid.signatures`
  had 1 entry afterwards, confirming numba compiled the loop.
Added a `use_disk` toggle so the Schwarzschild (`MD=0`) pipeline can render a
*pure* black-hole shadow — a solid dark disk with no surrounding ring —
directly addressing the cosmetic banding/ring artifact flagged as a known
issue at the end of Interaction 8. Also reorganized the end-to-end run into a
dedicated `test_runs/` area.

### Root cause of the unwanted "disk"

Even with `MD=0` (no disk mass in the metric), the rendered image showed an
accretion-disk-style ring. The ring was **not** coming from the spacetime — it
was produced entirely by the ray *classification*:

- `test_Z_SHADOW.trace_shadow` (the disk-crossing-*enabled* serial tracer used
  by the orchestrator) tags every ray that crosses the equatorial plane beyond
  `rho > b` with a `+50.0` z offset in `func`'s "Pontos do Disco" branch.
- `shadow_postprocess.classify_shadow` then reads that tag and marks those
  pixels as "beyond the disk" (`M2 = 2`), which `render_shadow` paints as the
  extra ring(s).

The repo already contained the pure-shadow variant (`test_parallel_SHADOW.py`,
whose `func` omits that branch), so the capability was never lost — the
orchestrator had simply wired up the disk-crossing tracer. Chosen fix (user
plan-approved: "use_disk flag on the serial path"): thread an opt-out flag
through the existing serial pipeline rather than duplicating a tracer or
refactoring the parallel one's inline driver.

### Changes

- **`test_Z_SHADOW.py`** — `func(...)` and `trace_shadow(...)` gained a
  `use_disk=True` parameter; when `False`, the `rho > b` disk-crossing branch
  is skipped entirely, giving a pure BH-shadow trace (matching
  `test_parallel_SHADOW.py`). `func` is jitted, and the plain-`bool` argument
  compiles cleanly under numba (verified in both `True`/`False` cases).
- **`shadow_postprocess.py`** — `classify_shadow(...)` gained
  `include_beyond_disk=True`; when `False`, the beyond-disk (`M2 = 2`)
  assignment is skipped, leaving only captured (1) / neither (0).
- **`symmetry.py`** — `render_shadow(...)` gained `use_disk=True`, forwarded as
  `include_beyond_disk=use_disk` into `classify_shadow`.
- **Orchestrator moved & renamed.** `test_run_1.py` → new
  `test_runs/generate_Schwarzschild_no_disk/test_run_schwarzschild.py`. It sets
  `USE_DISK = False` and threads it into both `trace_shadow` and
  `render_shadow`. Because the script now lives two directories below the repo
  root (not one), its `REPO_ROOT` sys.path computation was deepened by one more
  `os.path.dirname`, and output paths remain anchored to the script's own
  folder via `SCRIPT_DIR` so matrices/figures follow the script automatically.
- **New `test_runs/Test_Results.md`** — documents this first pure-Schwarzschild
  run (parameters, pipeline stages, and the resulting shadow image).

### Verification

- Unit check on the new flag: `classify_shadow` on a synthesized array yields a
  beyond-disk pixel with `include_beyond_disk=True` and none with `False`.
- numba smoke test: tabulated a small lambda matrix, loaded it, and called the
  jitted `func` with `use_disk` both `False` and `True` — both compiled and ran.
- Import-path check from the new folder depth: `REPO_ROOT` resolves to the repo
  root and all three pipeline modules import successfully.
- End-to-end: user ran the orchestrator and visually confirmed a clean
  Schwarzschild shadow with **no** ring (see `Test_Results.md`).

## Suggested next steps (not yet done)

- Give the `np.savetxt` outputs consistent, `.gitignore`-friendly extensions.
- Consider threading `Mat_nu` as an explicit `lamb` parameter instead of a
  module global set via `load_matrix` (deferred/provisional design choice from
  this interaction).
- `test_parallel_SHADOW.py`'s nested-`parallel=True` limitation is still
  unfixed; running the parallel/production-resolution path needs a dedicated
  numba-nesting fix.
- `symmetry.py`'s 2-color shadow plot doesn't distinguish "beyond disk" from
  "captured" pixels (see the cosmetic issue noted in Interaction 8). For the
  `MD=0` case this is now sidestepped by `use_disk=False` (Interaction 9), which
  drops the beyond-disk class entirely; a 3-color `c_map` would still be worth
  it if that classification is ever needed for a non-`MD=0`, disk-enabled run.
- `README.md`'s Pipeline section still describes the pre-Interaction-8 driver
  scripts; could be updated to mention `test_run_1.py` and the new
  `generate_lambda_matrix`/`trace_shadow`/`render_shadow` function-call
  pattern.
