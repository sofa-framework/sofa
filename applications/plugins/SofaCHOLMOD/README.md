# SofaCHOLMOD

A SOFA plugin providing a direct linear solver based on the **supernodal sparse
Cholesky factorization** of [CHOLMOD](https://github.com/DrTimothyAldenDavis/SuiteSparse)
(from SuiteSparse), wrapped through [Eigen's CholmodSupport](https://eigen.tuxfamily.org/dox/group__CholmodSupport__Module.html).

The plugin is kept separate from the core SOFA solvers so that the dependency on
SuiteSparse / CHOLMOD stays isolated to this optional plugin.

## Component

### `EigenCholmodSupernodalLLT`

A direct solver computing a supernodal `L·Lᵀ` Cholesky factorization of a
symmetric positive-definite system.

Unlike simplicial solvers (`SparseLDLSolver`, `EigenSimplicialLDLT`), the
supernodal factorization groups columns with a similar sparsity pattern into
dense blocks and factorizes them with **dense BLAS3 kernels** (`dgemm`,
`dsyrk`, `dpotrf`). For medium-to-large systems this is dramatically faster —
we measured up to **~8-11x** faster than `SparseLDLSolver` on a 15k-DOF FEM beam
(see [Performance](#performance)).

CHOLMOD manages its own fill-reducing ordering internally (AMD / METIS /
NESDIS). An `OrderingMethod` component is still required in the scene (it is part
of the common linear-solver API), but its choice is **ignored** by this solver.

> **Note:** CHOLMOD only supports **double** precision. The plugin will not build
> if SOFA is configured with `SReal = float`.

#### Templates

| `template` value                     | Block type            |
|--------------------------------------|-----------------------|
| `CompressedRowSparseMatrix`          | scalar (`SReal`)      |
| `CompressedRowSparseMatrixMat3x3`    | `Mat<3,3,SReal>`      |

`CompressedRowSparseMatrixMat3x3` is recommended for 3D mechanics (3-DOF nodes).

#### Data

| Data         | Type  | Default | Description |
|--------------|-------|---------|-------------|
| `numThreads` | `int` | `1`     | Number of threads the underlying BLAS backend may use for the factorization. The default of `1` is a safe choice that avoids thread oversubscription when several solvers run concurrently (see [Threading](#threading)). For a **single large standalone system**, a moderate value (roughly half the physical cores) is faster and should be tuned per machine. A value `<= 0` leaves the BLAS default untouched (controlled by the `OPENBLAS_NUM_THREADS` / `OMP_NUM_THREADS` environment variables). Only effective with OpenBLAS or MKL. |

## Usage

```xml
<RequiredPlugin name="SofaCHOLMOD"/>

<!-- an ordering method is required by the API, but ignored by CHOLMOD -->
<NaturalOrderingMethod/>

<EigenCholmodSupernodalLLT template="CompressedRowSparseMatrixMat3x3" numThreads="4"/>
```

A complete example is provided in
[`examples/FEMBAR_EigenCholmodSupernodalLLT.scn`](examples/FEMBAR_EigenCholmodSupernodalLLT.scn).

## Dependencies

- **SuiteSparse / CHOLMOD** (with its `SuiteSparse_config` and `AMD` modules).
- An **optimized BLAS/LAPACK** implementation backing CHOLMOD (OpenBLAS, Intel
  MKL, Apple Accelerate, …). This is the single most important factor for
  performance — see below.

Installing the dependency:

| Platform | Command |
|----------|---------|
| Ubuntu / Debian | `sudo apt install libsuitesparse-dev libopenblas-dev` |
| macOS (Homebrew) | `brew install suite-sparse` (links Apple Accelerate) |
| vcpkg | `vcpkg install suitesparse` |

The plugin locates CHOLMOD via [`cmake/Modules/FindCHOLMOD.cmake`](../../../cmake/Modules/FindCHOLMOD.cmake),
which first tries the SuiteSparse CMake config (SuiteSparse >= 7) and falls back
to a manual header/library search.

## Building

Enable the plugin when configuring SOFA:

```bash
cmake -DPLUGIN_SOFACHOLMOD=ON <path-to-sofa>
```

## Performance

CHOLMOD's supernodal factorization delegates almost all of its work to dense
BLAS3 kernels, so **its speed is dictated by whichever BLAS is linked behind
CHOLMOD**, not by SOFA. Keep this in mind when benchmarking:

- On **macOS**, SuiteSparse links Apple **Accelerate** (fast), so the speedup is
  always present.
- On **Linux**, the BLAS backing `libcholmod` is selected by
  `update-alternatives`. If it resolves to the reference *netlib* BLAS instead of
  an optimized one (OpenBLAS/MKL), the speedup largely disappears. Check and fix
  with:

  ```bash
  # what BLAS does CHOLMOD use?
  ldd $(ldconfig -p | awk '/libcholmod.so/{print $NF; exit}') | grep -i blas

  # prefer OpenBLAS system-wide
  sudo update-alternatives --config libblas.so.3-x86_64-linux-gnu
  sudo update-alternatives --config liblapack.so.3-x86_64-linux-gnu
  ```

### Threading

An optimized multithreaded BLAS (e.g. OpenBLAS-pthread) uses *all* cores by
default. The right thread count depends entirely on the scene:

- **A single large system** benefits from multithreaded BLAS, but only up to a
  point: there is a sweet spot (roughly half the physical cores), beyond which
  launch/synchronization overhead makes it *slower* again — using all cores is
  usually the worst choice.
- **Many systems solved concurrently** (multiple objects, especially with
  `parallelODESolving="true"`) already saturate the cores through object-level
  parallelism. Adding BLAS threads on top **oversubscribes** them and is
  catastrophic.

`numThreads` therefore **defaults to `1`** — the safe choice that never
oversubscribes. For a single large system, raise it to the sweet spot (tune per
machine). The optimum is machine-dependent, so treat the numbers below as
illustrative (24-core Linux box, OpenBLAS backend).

Single large system — `examples/FEMBAR_EigenCholmodSupernodalLLT.scn`
(10×10×50 grid, ~15k DOF). Effect of `numThreads`:

| `numThreads`      | ms / step | FPS  |
|-------------------|-----------|------|
| 1                 | ~67       | 14.9 |
| 4                 | ~52       | 19.1 |
| **8** (sweet spot)| **~48**   | 20.8 |
| 16                | ~64       | 15.6 |
| 24 (all cores)    | ~96       | 10.4 |

For reference, on the same scene `SparseLDLSolver` is ~762 ms/step, and CHOLMOD
on a non-optimized *reference* BLAS is ~346 ms/step — i.e. having an optimized
BLAS matters far more than the exact thread count.

Many small systems in parallel — `examples/TorusFall.scn` (10 tori,
`parallelODESolving="true"`). Here more BLAS threads only hurt:

| Solver / configuration                          | FPS   | Speedup |
|-------------------------------------------------|-------|---------|
| `EigenCholmodSupernodalLLT`, all cores          | ~20   | 0.5x    |
| `SparseLDLSolver` (`parallelInverseProduct`)    | ~40   | 1.0x    |
| `EigenCholmodSupernodalLLT`, `numThreads="1"`   | ~108  | 2.7x    |

> Rule of thumb: `numThreads="1"` whenever several solvers run concurrently; a
> moderate value (~half the cores) for a single large standalone system.

## Constraint solving (compliance matrix)

For Lagrangian constraints (e.g. contacts), `addJMInvJtLocal` computes the
compliance block `W = J·A⁻¹·Jᵀ`. Since CHOLMOD factorizes `P·A·Pᵀ = L·Lᵀ`, this
is `W = Zᵀ·Z` with `Z = L⁻¹·P·Jᵀ` — one triangular forward-solve (all constraint
columns at once) plus one symmetric product, instead of a full solve per row.

Implementation notes:

- The `Z = L⁻¹·P·Jᵀ` solve uses **`cholmod_solve2`** with workspace buffers held
  on the solver's `CholmodSolverProxy` and **reused across calls**, avoiding a
  per-step allocate/free of the dense right-hand side (relevant for scenes with
  many small systems solved every step).
- `W = Zᵀ·Z` is computed with the BLAS symmetric rank-k update **`dsyrk`** (only
  the triangle that is needed, half the flops of a full product), threaded per
  `numThreads`; it falls back to an Eigen rank update if `dsyrk` is unavailable.

Performance vs `SparseLDLSolver` is size-dependent: CHOLMOD's compliance is
faster for systems above ~1000 DOF per object (and the margin grows with size),
while for very small systems the fixed per-call overhead makes the two solvers
comparable.

## License

LGPL 2.1+ — see the SOFA license.
