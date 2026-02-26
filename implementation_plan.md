# Implementation Plan: Fast Iterative Method (FIM) Eikonal Solver in Rust

This document provides a step-by-step plan for implementing the FIM eikonal solver described in `eikonal_spec.md`. Each stage is self-contained with clear inputs, outputs, and a sub-agent assignment. A master checklist at the end tracks overall progress.

---

## Stage 1: Project Scaffolding & Dependencies

**Sub-agent:** `Bash` (project init)

### Steps
1. Run `cargo init --name eikonal-fim` in the project directory.
2. Replace the generated `Cargo.toml` with the spec's dependency list:
   - `ndarray 0.15`, `ndarray-npy 0.8`, `matfile 0.5` (with `ndarray` feature)
   - `rayon 1.10`, `crossbeam-queue 0.3`
   - `clap 4` (derive feature), `anyhow 1`
   - Dev: `approx 0.5`, `criterion 0.5` (html_reports)
   - Bench target: `solver_bench`, `harness = false`
3. Create the module file tree:
   ```
   src/
     lib.rs          # Public API re-exports
     core.rs         # GridData, TilingScheme traits, CartesianGrid<N>
     update_kernels.rs  # 2D/3D quadratic solvers
     scheduler.rs    # Rayon pool, SegQueue, bitset, termination protocol
     error.rs        # EikonalError enum
     io.rs           # .npy and .mat import/export
     main.rs         # CLI binary (clap)
   benches/
     solver_bench.rs # Criterion benchmarks (stub)
   ```
4. Add module declarations in `lib.rs` and verify `cargo check` passes.

### Exit criteria
- `cargo check` succeeds with no errors.
- All module files exist (can be empty stubs with `// TODO`).

---

## Stage 2: Error Types & Validation Utilities

**Sub-agent:** `api-engineer` — define error types and validation logic

### Steps
1. Define `EikonalError` enum in `src/error.rs`:
   - `InvalidGridShape { axis: usize, size: usize }` — dimension < 2
   - `InvalidTileSize { axis: usize, tile: usize, grid: usize }`
   - `InvalidGridSpacing(f64)` — non-positive or non-finite
   - `InvalidSlowness { index: usize, value: f64 }` — non-positive or non-finite
   - `InvalidSource { coord: Vec<f64>, reason: String }`
   - `InvalidTolerance(f64)`
   - `InvalidVelocity { index: usize, value: f64 }`
   - `ShapeMismatch { expected: Vec<usize>, got: Vec<usize> }`
   - `UnsupportedDtype(String)`
   - `UnsupportedFileFormat(String)`
   - `MatVariableNotFound { expected: String, available: Vec<String> }`
   - `MaxTilePopsExceeded { limit: u64 }`
   - `IoError(std::io::Error)`
   - `Other(String)` — catch-all for anyhow bridging
2. Implement `std::fmt::Display`, `std::error::Error`, and `From<std::io::Error>`.
3. Define `pub type Result<T> = std::result::Result<T, EikonalError>;`
4. Write unit tests for `Display` output.

### Exit criteria
- `cargo test` passes for the error module.

---

## Stage 3: Core Grid & Tiling Abstractions

**Sub-agent:** `api-engineer` — implement traits and concrete grid type

### Steps
1. In `src/core.rs`, define the `GridData<const N: usize>` trait exactly as specified (get_u, get_f, update_u, set_u_init, shape, strides, grid_spacing, num_nodes, flat_to_nd, nd_to_flat).
2. Define the `TilingScheme<const N: usize>` trait (tile_size, tile_counts, num_tiles, tile_id_to_nd, nd_to_tile_id, tile_extent).
3. Implement `CartesianGrid<const N: usize>`:
   - Fields: `shape: [usize; N]`, `strides: [usize; N]`, `h: f64`, `travel_time: Box<[AtomicU64]>`, `slowness: Box<[f64]>`, `tile_size: [usize; N]`, `tile_counts: [usize; N]`, `tile_strides: [usize; N]`.
   - Constructor `new(shape, h, slowness)` — validates inputs, computes strides (row-major), initializes travel_time to INFINITY.
   - `update_u` — CAS loop per spec (compare_exchange_weak, Release/Relaxed).
   - `set_u_init` — unconditional store (Relaxed), used only before solve.
   - `tile_extent` — clamps end to `min(start + tile_size[d], shape[d])`.
   - Implement both `GridData<N>` and `TilingScheme<N>` for `CartesianGrid<N>`.
   - Require `N` to be 2 or 3 at compile time (use `assert!` in constructor or const-generic constraints).
4. Unit tests:
   - Flat ↔ ND index round-trips.
   - Tile ID ↔ ND tile index round-trips.
   - `tile_extent` for full and partial tiles.
   - `update_u` monotonicity (only decreases).
   - CAS correctness under simulated contention (spawn threads that concurrently update the same cell).

### Exit criteria
- All unit tests pass.
- `CartesianGrid::<2>` and `CartesianGrid::<3>` both compile and pass tests.

---

## Stage 4: Update Kernels (2D & 3D Quadratic Solvers)

**Sub-agent:** `api-engineer` — implement numerics

### Steps
1. In `src/update_kernels.rs`, implement:
   - `solve_2d(a: f64, b: f64, f: f64, h: f64) -> f64` — quadratic formula with discriminant check and 1D fallback.
   - `solve_3d(a: f64, b: f64, c: f64, f: f64, h: f64) -> f64` — cascading 3D→2D→1D fallback per spec.
   - `update_node_2d(grid, idx) -> f64` — reads 4 neighbors, extracts min per axis, calls `solve_2d`.
   - `update_node_3d(grid, idx) -> f64` — reads 6 neighbors, extracts min per axis, calls `solve_3d`.
   - Generic wrapper `update_node<const N: usize>(grid, idx) -> f64` dispatching on N.
2. Handle boundary conditions: substitute `f64::INFINITY` for out-of-bounds neighbors.
3. Handle infinity propagation: avoid `INFINITY - INFINITY` via the fallback logic (if both neighbors along an axis are infinite, skip that axis).
4. Unit tests:
   - Known analytical cases: point source on a small grid, check individual node updates.
   - Discriminant-negative fallback: engineer inputs where `2f²h² < (a-b)²`.
   - All-infinite neighbors → result is INFINITY.
   - 3D cascading: cases that exercise each fallback level.

### Exit criteria
- All unit tests pass.
- No NaN produced for any valid input combination.

---

## Stage 5: Scheduler — Active List, Bitset, Termination Protocol

**Sub-agent:** `api-engineer` — implement concurrency machinery

### Steps
1. In `src/scheduler.rs`, implement:
   - **AtomicBitset:** `Vec<AtomicU64>` with `try_set(id) -> bool` (fetch_or, AcqRel) and `clear(id)` (fetch_and, Release).
   - **TileQueue:** Wrapper around `SegQueue<usize>` + `AtomicBitset` providing `push_if_new(tile_id) -> bool` and `pop() -> Option<usize>` (clears bit on pop).
2. Implement `FimSolver<const N: usize, G>`:
   - Fields: `grid: G`, `tolerance: f64`, `max_local_iters: usize`, `tile_size: [usize; N]`, `num_threads: Option<usize>`, `max_tile_pops: Option<u64>`, `progress_callback: Option<Box<dyn Fn(ProgressInfo) + Send>>`.
   - Builder methods: `new()`, `with_tile_size()`, `with_max_local_iters()`, `with_threads()`, `with_max_tile_pops()`.
   - `add_source(coord: [f64; N])` — exact distance seeding within radius 2h, mark containing tiles active.
   - **`solve()`** — the main driver:
     a. Build `rayon::ThreadPool` with configured thread count.
     b. Initialize shared state: `SegQueue`, `AtomicBitset`, `in_flight: AtomicUsize`, `done: AtomicBool`, `tile_pops: AtomicU64`.
     c. `pool.scope(|s| { ... })` spawning `num_threads` tasks.
     d. Each task runs the termination-aware loop (spec pseudocode):
        - Check `done` → break.
        - Increment `in_flight` (AcqRel).
        - Pop tile → process or handle empty.
        - Process tile: run up to `max_local_iters` local sweeps with multi-directional sweep ordering.
        - After local convergence or max iters: for each face, check if boundary nodes changed by > tolerance; if so, activate the neighbor tile via `push_if_new`.
        - Decrement `in_flight` (AcqRel).
        - Increment `tile_pops` if progress callback is set; check max_tile_pops.
     e. Termination: empty queue AND in_flight == 0 → set done.
     f. Progress callback: invoked at most every 500ms (track last call time per thread or use a shared `AtomicU64` timestamp).
3. Implement multi-directional sweep ordering:
   - For local iteration k, direction index = k % 2^N.
   - Each direction is a combination of ±1 per axis.
   - Iterate over tile extent in the corresponding order.
4. Unit tests:
   - AtomicBitset: set/clear/try_set semantics, concurrent access.
   - Single-threaded solve on a tiny 16×16 grid, point source at center, uniform slowness — verify result is close to Euclidean distance.
   - Multi-threaded solve on the same grid — verify same result (within round-off).
   - Termination: verify solver terminates and active list is empty.
   - MaxTilePopsExceeded: set a very low limit, verify error is returned.

### Exit criteria
- All unit tests pass.
- Solver produces correct results on trivial grids in both 2D and 3D.

---

## Stage 6: I/O — NumPy and MAT-file Support

**Sub-agent:** `api-engineer` — implement file I/O

### Steps
1. In `src/io.rs`, implement:
   - **`.npy` import:** Use `ndarray-npy` to read `Array<f64, IxDyn>` (or promote f32→f64). Validate shape matches grid. Return `Vec<f64>`.
   - **`.npy` export:** Convert `Box<[AtomicU64]>` → `Array<f64, IxN>`, write via `ndarray-npy`.
   - **`.mat` import:** Use `matfile` crate. Look up variable by expected name (`slowness` or `velocity`). Validate dtype (f64 or f32→f64). Validate shape. Handle column-major→row-major transpose.
   - **`.mat` export:** Implement minimal Level 5 MAT writer:
     a. 128-byte header (descriptive text + version + endian indicator).
     b. miMATRIX data element containing: array flags (mxDOUBLE_CLASS), dimensions array, array name (`traveltime`), real part (raw f64 bytes in column-major order).
     c. Proper 8-byte alignment padding for each sub-element.
   - **Velocity→slowness conversion:** Element-wise `1.0 / v` with validation (positive, finite).
   - **File format dispatch:** Infer format from extension, reject unsupported extensions.
2. Implement `solver.save(path)` method.
3. Unit tests:
   - Round-trip: write .npy → read back → compare.
   - Round-trip: write .mat → read back with matfile → compare.
   - Shape mismatch detection.
   - f32 promotion.
   - Velocity→slowness conversion with edge cases.

### Exit criteria
- Round-trip tests pass for both formats.
- MAT files are readable by scipy.io.loadmat (manual verification or integration test).

---

## Stage 7: CLI Binary

**Sub-agent:** `api-engineer` — implement CLI with clap

### Steps
1. In `src/main.rs`, implement the CLI using `clap` derive:
   - Parse all flags from the spec (--dim, --size, --source, --spacing, --slowness, --tolerance, --tile-size, --max-local-iters, --output, --threads, --max-tile-pops, --progress).
   - `--source` is repeatable (multiple sources).
   - `--slowness` parsing: dispatch on prefix (`uniform:`, `gradient:`, `slowness-file:`, `velocity-file:`).
   - Gradient mode: compute slowness field `f(y) = 1 / (v0 + g*y)` for 2D, `f(z) = 1 / (v0 + g*z)` for 3D. Validate that velocity is positive everywhere.
2. Wire CLI arguments to `CartesianGrid::new()` → `FimSolver::new()` → `add_source()` → `solve()` → `save()`.
3. Progress callback: when `--progress`, print to stderr: `[{elapsed:.1}s] tiles_processed={} active={} in_flight={}`.
4. Error handling: map `EikonalError` to user-friendly messages via `anyhow`.
5. Dimension dispatch: since `N` is a const generic, use a match on `--dim` to call either the `<2>` or `<3>` code path.

### Exit criteria
- `cargo build` succeeds.
- `cargo run -- --dim 2 --size 64,64 --source 32,32 -o /tmp/test.npy` produces output.
- `cargo run -- --help` prints the expected usage.

---

## Stage 8: Verification Tests

**Sub-agent:** `testing-engineer` — implement integration/verification tests

### Steps
1. Create `tests/verification.rs` with the four spec-mandated tests:

   **Test 1: Point Source (Homogeneous)**
   - 2D, uniform f=1.0, source at center.
   - Run at 128² and 256². Compute L∞ error vs analytical `|x - xs|`.
   - Assert error ratio ≈ 2.0 (±30% tolerance for O(h) convergence).

   **Test 2: Linear Velocity Gradient (2D)**
   - v(y) = 1.0 + 0.5·y, source at grid center.
   - Compute analytical traveltime via arccosh formula.
   - Check L∞ error is O(h) at receivers ≥ 10h from source.

   **Test 3: Checkerboard Slowness**
   - 2D, alternating f=1.0/2.0 blocks of size 8×8 on a 128² grid.
   - Verify: solver terminates, all u values non-negative and finite.

   **Test 4: Multi-Source**
   - 2D, f=1.0, two sources.
   - Verify u(x) ≈ min(|x-s1|, |x-s2|) with O(h) error.

2. Add 3D versions of Test 1 and Test 3 (smaller grids, e.g., 32³ and 64³).

### Exit criteria
- `cargo test --test verification` passes all tests.

---

## Stage 9: Benchmarks

**Sub-agent:** `testing-engineer` — implement criterion benchmarks

### Steps
1. In `benches/solver_bench.rs`, implement using `criterion`:
   - **Single-thread baseline:** 512² homogeneous, 1 thread.
   - **Thread scaling:** 1024² homogeneous, [1, 2, 4, 8, num_cpus] threads.
   - **3D scaling:** 128³ homogeneous, 1 thread and all-cores.
   - **Grid size scaling:** [128², 256², 512², 1024²] homogeneous, all-cores.
2. Each benchmark: construct grid, add source at center, call `solve()`, use `criterion::black_box` on result.

### Exit criteria
- `cargo bench` compiles and runs without errors.
- Benchmark results are printed (performance analysis is manual).

---

## Stage 10: Integration Testing & Polish

**Sub-agent:** `implementation-verifier` — end-to-end validation

### Steps
1. Run the full CLI for a 2D 256² case and verify output file is readable.
2. Run the full CLI for a 3D 64³ case.
3. Test all slowness modes: uniform, gradient, slowness-file, velocity-file.
4. Test .npy and .mat output formats.
5. Test error paths: invalid args, bad files, shape mismatches.
6. Run `cargo clippy` and fix all warnings.
7. Run `cargo test` (all unit + integration tests).
8. Run `cargo bench` to confirm benchmarks work.
9. Ensure public API matches the library example in the spec.

### Exit criteria
- `cargo clippy` clean.
- `cargo test` all green.
- `cargo bench` runs successfully.
- CLI works end-to-end for all modes.

---

## Master Checklist

Update this checklist at the end of each stage:

- [ ] **Stage 1:** Project scaffolding — `cargo check` passes
- [ ] **Stage 2:** Error types — `cargo test` passes for error module
- [ ] **Stage 3:** Core grid & tiling — all unit tests pass
- [ ] **Stage 4:** Update kernels — all unit tests pass, no NaN
- [ ] **Stage 5:** Scheduler — solver works on trivial grids (2D & 3D)
- [ ] **Stage 6:** I/O — round-trip tests pass for .npy and .mat
- [ ] **Stage 7:** CLI — binary builds and runs end-to-end
- [ ] **Stage 8:** Verification tests — all 4 spec tests pass
- [ ] **Stage 9:** Benchmarks — `cargo bench` runs successfully
- [ ] **Stage 10:** Integration & polish — clippy clean, all tests green
