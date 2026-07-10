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
| `numThreads` | `int` | `1`     | Number of threads the underlying BLAS backend may use for the factorization. The default of `1` is the fastest setting for the vast majority of SOFA scenes and avoids thread oversubscription (see [Threading](#threading)). Increase it only for a single, very large standalone system. A value `<= 0` leaves the BLAS default untouched (controlled by the `OPENBLAS_NUM_THREADS` / `OMP_NUM_THREADS` environment variables). Only effective with OpenBLAS or MKL. |

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
default. For the medium-sized systems typical in SOFA this **oversubscribes**
threads: the launch/synchronization overhead makes the solver *slower* than with
a single thread, and the problem becomes catastrophic when several solvers are
factorized in parallel (e.g. multiple objects with `parallelODESolving="true"`),
where each solver would otherwise spawn one BLAS pool per core.

For these reasons `numThreads` **defaults to `1`**, which is the fastest setting
in nearly all cases. Increase it only for a single, very large standalone
system.

Illustrative measurements on a 24-core Linux box, OpenBLAS backend:

Single medium system — `examples/FEMBAR_EigenCholmodSupernodalLLT.scn`
(10×10×50 grid, ~15k DOF):

| Solver / configuration                          | ms / step | Speedup |
|-------------------------------------------------|-----------|---------|
| `SparseLDLSolver`                               | ~762      | 1.0x    |
| `EigenCholmodSupernodalLLT`, reference BLAS     | ~346      | 2.2x    |
| `EigenCholmodSupernodalLLT`, OpenBLAS, all cores| ~92       | 8.3x    |
| `EigenCholmodSupernodalLLT`, OpenBLAS, `numThreads="1"` | ~70 | 10.9x |

Many small systems in parallel — `examples/TorusFall.scn` (10 tori,
`parallelODESolving="true"`):

| Solver / configuration                          | FPS   | Speedup |
|-------------------------------------------------|-------|---------|
| `EigenCholmodSupernodalLLT`, OpenBLAS, all cores| ~20   | 0.5x    |
| `SparseLDLSolver` (`parallelInverseProduct`)    | ~40   | 1.0x    |
| `EigenCholmodSupernodalLLT`, `numThreads="1"`   | ~108  | 2.7x    |

> Multithreaded BLAS only pays off for a single, very large system; whenever
> several solvers run concurrently, keep `numThreads="1"`.

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
