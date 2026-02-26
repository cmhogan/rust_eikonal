# Technical Specification for a Scalable Rust Implementation of the Fast Iterative Method (FIM) for Isotropic Eikonal Equations

The Eikonal equation is a first-order nonlinear partial differential equation of the form $|\nabla u| = f(x)$, where $u$ represents the travel time from a source and $f(x)$ is the slowness (reciprocal of velocity). While the Fast Marching Method (FMM) is $O(N \log N)$, its reliance on a global priority queue limits parallel scalability. The Fast Iterative Method (FIM) is designed specifically for parallel architectures, using a label-correcting approach that manages an "Active Set" of nodes or tiles to achieve high performance on multi-core systems.

### Scope and Limitations

This implementation targets **isotropic** Eikonal equations on **uniform Cartesian grids** in 2D and 3D. Anisotropic slowness fields, non-uniform grid spacing, and unstructured meshes are out of scope.

---

## Part 1: Algorithm & Mathematics

### FIM Mechanics: Active Tile Logic

To optimize for cache locality and minimize synchronization overhead, the Cartesian grid is partitioned into $N$-dimensional blocks called "Tiles" (e.g., $8 \times 8$ in 2D or $4 \times 4 \times 4$ in 3D).

- **Block-Based Updates:** Instead of tracking individual nodes, the solver maintains an Active List of Tile IDs.
- **Cache Efficiency:** When a tile is updated, its values are loaded into the L1/L2 cache. The worker thread performs multiple local iterations within the tile before checking if the change has propagated to neighbors.
- **Convergence:** A tile is "locally converged" when the maximum change between iterations falls below a tolerance $\epsilon$: $\max |u^{(k+1)} - u^{(k)}| < \epsilon$.

#### Tile Size Selection

Tile size controls the trade-off between cache utilization and parallelism granularity:

- **Too small** (e.g., $2 \times 2$): Excessive scheduling overhead, poor cache reuse, too many inter-tile boundary checks.
- **Too large** (e.g., $32 \times 32$): Too few tiles for parallel work distribution; interior nodes do unnecessary computation when only boundary values changed.
- **Recommended defaults:** $8 \times 8$ in 2D, $4 \times 4 \times 4$ in 3D. These fit comfortably in L1 cache (64 or 64 f64 values = 512 bytes) while providing sufficient parallelism for grids of $256^2$ or larger.

The tile size should be configurable at construction time.

#### Partial Tiles

When the grid dimensions are not evenly divisible by the tile size, the last tile along each axis is a **partial tile** containing fewer nodes. For example, a $300 \times 300$ grid with tile size 8 produces $38$ tiles along each axis: 37 full $8 \times 8$ tiles and 1 partial $4 \times 8$ (or $8 \times 4$) tile. The tile count along axis $d$ is $\lceil \text{shape}[d] / \text{tile\_size}[d] \rceil$.

The update kernel must respect the actual extent of each tile by clamping iteration bounds to the grid boundary. Partial tiles participate in the Active List and neighbor activation identically to full tiles.

### Boundary Conditions

Boundary nodes (those at the edge of the computational domain) are treated as having **infinite travel time** ($u = +\infty$) in directions that would reference out-of-bounds neighbors. Concretely, when the upwind stencil for node $(i,j)$ needs $u_{i-1,j}$ but $i=0$, the value $+\infty$ is substituted, ensuring that information only propagates inward from sources and never "reflects" off domain edges.

### Source Initialization

Source points are specified as floating-point coordinates $x_s$ in physical space. Initialization proceeds as follows:

1. **Exact distance seeding:** For all grid nodes within a radius of $r = 2h$ from $x_s$, set $u = f(x_s) \cdot \|x - x_s\|_2$. This provides sub-grid accuracy and avoids the $O(h)$ global error that results from simply setting the nearest grid node to zero. The radius $2h$ ensures that even when $x_s$ lies at the corner of a grid cell (worst case), at least the 4 (2D) or 8 (3D) nearest grid nodes are seeded, providing the upwind stencil with valid finite values in all directions from the start. Larger radii (e.g., $3h$) would seed more nodes but risk masking discretization errors near the source.
2. **Remaining nodes:** Initialize to $u = +\infty$ (`f64::INFINITY`).
3. **Active set seeding:** All tiles containing at least one initialized (non-infinity) node are added to the initial Active List.

For multiple sources, repeat step 1 for each source, taking the minimum $u$ at each node.

### Update Operator: Upwind Discretization

The solver uses a Godunov-type upwind finite difference scheme. For a uniform grid spacing $h$, the value $u$ at node $(i,j)$ is solved by considering the minimum neighbors along each axis.

Let $a = \min(u_{i-1,j}, u_{i+1,j})$, $b = \min(u_{i,j-1}, u_{i,j+1})$, and $c = \min(u_{i,j,k-1}, u_{i,j,k+1})$.

#### 2D Discretization

$$\left( \frac{u-a}{h} \right)^2 + \left( \frac{u-b}{h} \right)^2 = f_{i,j}^2$$

The solution for $u$ is given by the quadratic formula:

$$u = \frac{a+b + \sqrt{2f^2h^2 - (a-b)^2}}{2}$$

If the discriminant $2f^2h^2 - (a-b)^2 < 0$, or if the resulting $u \leq \max(a, b)$, the solver falls back to a 1D update: $u = \min(a, b) + fh$.

#### 3D Discretization

$$\sum_{d \in \{x,y,z\}} \left[ \max \left( \frac{u - u_d}{h}, 0 \right) \right]^2 = f_{i,j,k}^2$$

The local solver uses cascading fallback logic:

1. **3D Update:** Sort neighbors so $a \leq b \leq c$. Attempt to solve $3u^2 - 2(a+b+c)u + (a^2+b^2+c^2 - f^2h^2) = 0$. Accept the larger root if $u > c$.
2. **2D Update:** If the 3D solution is invalid ($u \leq c$ or discriminant negative), solve using the two smallest neighbors ($a, b$).
3. **1D Update:** If conditions fail, $u = a + fh$ (where $a$ is the smallest neighbor).

The update is accepted only if the new value is strictly less than the current value (monotonicity enforcement).

#### Numerical Stability Considerations

The quadratic solver must handle several edge cases:

- **Catastrophic cancellation:** When $a \approx b$ and $f \cdot h$ is small, the discriminant $2f^2h^2 - (a-b)^2$ involves subtracting two nearly equal quantities. Since the discriminant is only used under a square root and added to $(a+b)$, the absolute error is bounded by machine epsilon times $f^2 h^2$, which is negligible relative to $u$. No special handling is required beyond the existing fallback to 1D when the discriminant is negative.
- **Very large slowness ($f \gg 1$):** The product $f \cdot h$ may produce large intermediate values, but since $f$ and $h$ are finite `f64` values and the result $u$ is at most $f \cdot h$ larger than its smallest neighbor, overflow is not a concern for physically meaningful inputs. The input validation requirement that $f$ be finite is sufficient.
- **Very small slowness ($f \to 0$):** This produces near-zero travel times, which are numerically well-behaved. The input validation requirement that $f$ be positive is sufficient.
- **Infinity propagation:** Nodes initialized to `f64::INFINITY` participate in `min()` operations. Since `f64::INFINITY.min(x) == x` for finite `x`, and arithmetic on infinity produces infinity, uninitialized regions naturally remain at infinity until the wavefront reaches them. The update kernel must avoid computing `INFINITY - INFINITY` (which yields `NaN`); this is prevented by the fallback logic, since if both neighbors along an axis are infinite, that axis contributes nothing to the update.

### Global Convergence and Termination

The solver terminates when the Active List is empty **and** no worker threads are actively processing tiles. This is the standard FIM termination criterion.

As a safety measure, the solver also tracks total tile pops across all threads (via a shared `AtomicU64` counter). If this count exceeds `max_tile_pops` (default: `100 * num_tiles`), the solver sets the `done` flag and returns `Err(EikonalError::MaxTilePopsExceeded)`. This prevents infinite loops from pathological inputs or implementation bugs. The default limit is generous — typical solves use $5\text{–}20 \times$ the tile count — and can be increased via `--max-tile-pops` or `with_max_tile_pops()` in the library API.

#### Termination Protocol

A naïve check of "is the queue empty?" is insufficient because a worker thread may be mid-update and about to enqueue new neighbors. The implementation uses an **atomic in-flight counter** to resolve this race:

1. **Before attempting pop:** The worker atomically increments `in_flight: AtomicUsize` (using `Ordering::AcqRel`) **before** calling `queue.pop()`. This prevents a TOCTOU race where another thread observes `queue.is_empty() && in_flight == 0` between a successful pop and its corresponding increment.
2. **Failed pop:** If `queue.pop()` returns `None`, the worker atomically decrements `in_flight` and checks for termination.
3. **After processing:** When the worker finishes updating the tile and has enqueued any activated neighbors, it atomically decrements `in_flight`.
4. **Termination check:** The global loop terminates only when **both** the `SegQueue` is empty **and** `in_flight.load(Ordering::Acquire) == 0`.

In addition to `in_flight`, the implementation uses a shared `done: AtomicBool` flag to coordinate termination across all worker threads spawned by `rayon::scope`. A thread that observes the termination condition sets `done` to `true`. All threads check `done` in their loop condition. If new work is pushed before all threads exit, the pushing thread clears `done` back to `false`.

The main driver loop structure (run by each spawned Rayon task) is:

```
loop {
    if done.load(Acquire) {
        break; // another thread signaled termination
    }
    in_flight.fetch_add(1, AcqRel);
    if let Some(tile_id) = queue.pop() {
        bitset.clear(tile_id);
        // process tile, enqueue neighbors (pushing clears done if set)
        in_flight.fetch_sub(1, AcqRel);
    } else {
        in_flight.fetch_sub(1, AcqRel);
        if in_flight.load(Acquire) == 0 && queue.is_empty() {
            done.store(true, Release); // signal all threads to exit
            break;
        } else {
            // queue empty but work in flight — yield and retry
            std::thread::yield_now();
        }
    }
}
```

When a thread enqueues a new tile (via `try_set` + `SegQueue.push`), it must also clear the done flag to prevent premature termination:

```
if bitset.try_set(neighbor_id) {
    queue.push(neighbor_id);
    done.store(false, Release); // revoke any pending termination signal
}
```

This ensures that even if one thread has signaled termination, other threads that are about to push new work will revoke that signal. The `rayon::scope` call blocks until all spawned tasks return, guaranteeing no work is lost.

`yield_now()` is preferred over `spin_loop()` because within a Rayon thread pool, spinning prevents work-stealing from occurring and wastes CPU cycles during the tail end of computation when few active tiles remain.

#### Non-determinism

Because tiles are processed in non-deterministic order by the thread pool, multi-threaded
solves may produce slightly different floating-point results across runs. The CAS-based
monotonic update ensures convergence is correct, but the order in which updates are applied
can affect rounding at the ULP level. Single-threaded execution is fully deterministic.

**Convergence order:** This first-order upwind scheme is $O(h)$ accurate. Users requiring higher accuracy should refine the grid.

---

## Part 2: Rust Software Architecture

### Dimensionality Agnosticism via Traits

To avoid code duplication between 2D and 3D solvers, the implementation uses Rust traits and const generics. The design separates grid data access from tiling concerns.

```rust
/// Core grid data access. Provides travel time and slowness field
/// access, grid geometry, and index conversion utilities.
pub trait GridData<const N: usize> {
    /// Get the travel time at the given grid index.
    /// Internally performs an atomic load (Relaxed ordering).
    fn get_u(&self, idx: [usize; N]) -> f64;
    /// Get the slowness at the given grid index.
    fn get_f(&self, idx: [usize; N]) -> f64;
    /// Atomically update the travel time at the given grid index,
    /// storing `val` only if it is strictly less than the current value.
    /// Returns `true` if the value was updated, `false` if the existing
    /// value was already ≤ val.
    /// Internally uses a compare-and-exchange loop (see Monotonic Update below).
    fn update_u(&self, idx: [usize; N], val: f64) -> bool;
    /// Unconditionally set the travel time at the given grid index.
    /// Used only during initialization (e.g., filling all nodes with INFINITY
    /// before the solve begins). Must NOT be called concurrently with `update_u`.
    fn set_u_init(&self, idx: [usize; N], val: f64);
    /// Grid dimensions along each axis (e.g., [nx, ny] or [nx, ny, nz]).
    fn shape(&self) -> [usize; N];
    /// Strides for converting N-dimensional indices to flat storage offset.
    fn strides(&self) -> [usize; N];
    /// Grid spacing (uniform along all axes).
    fn grid_spacing(&self) -> f64;
    /// Total number of nodes in the grid.
    fn num_nodes(&self) -> usize;
    /// Convert a flat node index to an N-dimensional grid index.
    fn flat_to_nd(&self, flat: usize) -> [usize; N];
    /// Convert an N-dimensional grid index to a flat index.
    fn nd_to_flat(&self, idx: [usize; N]) -> usize;
}

/// Tiling scheme for partitioning a grid into blocks.
/// Separated from GridData so that alternative tiling strategies
/// (e.g., adaptive tile sizes) can be implemented independently.
pub trait TilingScheme<const N: usize> {
    /// Tile size along each axis (e.g., [8, 8] or [4, 4, 4]).
    fn tile_size(&self) -> [usize; N];
    /// Number of tiles along each axis: ceil(shape[d] / tile_size[d]).
    fn tile_counts(&self) -> [usize; N];
    /// Total number of tiles.
    fn num_tiles(&self) -> usize;
    /// Convert a flat tile ID to the N-dimensional tile index.
    fn tile_id_to_nd(&self, tile_id: usize) -> [usize; N];
    /// Convert an N-dimensional tile index to a flat tile ID.
    fn nd_to_tile_id(&self, tile_idx: [usize; N]) -> usize;
    /// Get the grid-coordinate range [start, end) for a tile along each axis.
    /// For partial tiles at the grid boundary, end is clamped to shape[d].
    fn tile_extent(&self, tile_idx: [usize; N], grid_shape: [usize; N]) -> [(usize, usize); N];
}

pub struct FimSolver<const N: usize, G: GridData<N> + TilingScheme<N> + Sync + Send> {
    grid: G,
    tolerance: f64,
    max_local_iters: usize,
}
// Sync is required because Rayon worker threads share a reference to the grid.
// Send is required because the solver may be moved into a Rayon scope.
```

Note that `update_u` takes `&self` rather than `&mut self`. This is because the underlying storage uses `AtomicU64`, which supports interior mutability and allows concurrent writes from different threads without requiring `&mut` access (see Concurrency Safety below).

#### Monotonic Update via Compare-and-Exchange

The `update_u` method must guarantee that travel time values only decrease. A simple `load` → compute → `store` sequence is unsafe under concurrency: thread A could load a value, thread B could write a smaller value, and then thread A would overwrite it with a larger one. Instead, `update_u` uses an atomic compare-and-exchange (CAS) loop:

```rust
fn update_u(&self, idx: [usize; N], val: f64) -> bool {
    let atom = &self.travel_time[self.nd_to_flat(idx)];
    let mut current = atom.load(Ordering::Relaxed);
    loop {
        if f64::from_bits(current) <= val {
            return false; // existing value is already as good or better
        }
        match atom.compare_exchange_weak(
            current,
            val.to_bits(),
            Ordering::Release,  // success: publish the new value
            Ordering::Relaxed,  // failure: just reload
        ) {
            Ok(_) => return true,
            Err(actual) => current = actual, // retry with updated value
        }
    }
}
```

This ensures monotonicity regardless of thread interleaving. The `compare_exchange_weak` variant is used because it is cheaper on architectures with LL/SC (e.g., ARM) and the CAS is already in a loop. The `Release` ordering on success ensures that the updated value is visible to other threads that subsequently load it.

Source initialization (`add_source`) also uses `update_u` rather than an unconditional store, so that multiple sources are handled correctly (each source seeds the minimum travel time).

### Tile Ownership Convention

Each grid node belongs to exactly **one** tile, determined by integer division of its coordinates by the tile size: node $(i, j)$ belongs to tile $(\lfloor i / T_x \rfloor, \lfloor j / T_y \rfloor)$ (and analogously in 3D). When a tile is being updated, the owning thread may **write** only to nodes belonging to that tile. It may **read** nodes belonging to adjacent tiles (to obtain stencil neighbor values), but never write them. This convention eliminates write-write conflicts between concurrent tile updates.

### Data Structures

- **Grid Storage:** The travel time array is stored as `Box<[AtomicU64]>`, where each entry holds the bit-pattern of an `f64` (via `f64::to_bits` / `f64::from_bits`). This allows concurrent atomic access without `unsafe` pointer casts. The slowness field is read-only after initialization and stored as `Box<[f64]>`. The `ndarray` crate is used only at the I/O boundary for `.npy` and `.mat` import/export.
- **Active Set:** A `crossbeam_queue::SegQueue` provides a lock-free multi-producer multi-consumer queue for distributing Tile IDs. To prevent duplicate tile entries, a custom atomic bitset backed by `Vec<AtomicU64>` tracks "is-queued" state. Each bit corresponds to one tile ID. Operations:
  - `try_set(tile_id) -> bool`: Atomically sets the bit using `fetch_or` with `Ordering::AcqRel`. Returns `true` if the bit was previously unset (i.e., the tile was not already queued).
  - `clear(tile_id)`: Atomically clears the bit using `fetch_and` with `Ordering::Release`.
- **In-flight counter:** An `AtomicUsize` tracks the number of tiles currently being processed by worker threads (see Termination Protocol above).
- **Done flag:** An `AtomicBool` signals termination to all worker threads. Set to `true` when a thread observes an empty queue with zero in-flight tiles. Cleared back to `false` by any thread that enqueues new work, revoking the termination signal (see Termination Protocol above).

### Parallelism: Hybrid Execution

FIM uses a hybrid approach for efficiency:

- **Intra-Tile:** Local Gauss-Seidel updates are performed within a single thread to accelerate convergence. A configurable maximum of local iterations per tile activation (default: 4) prevents excessive work on tiles that converge slowly. Each local iteration uses a different **multi-directional sweep order** to avoid directional bias (see below).
- **Inter-Tile:** Jacobi-style/Asynchronous updates are used globally, where tiles are processed in parallel via Rayon.

#### Rayon Integration

The solver uses `rayon::scope` to spawn worker tasks that each run the pop-process-push loop. This is distinct from `rayon::par_iter`, which is unsuitable because the work queue is dynamic (tiles are enqueued during processing, not known upfront).

The thread count is configured by constructing a custom `rayon::ThreadPoolBuilder` with the requested number of threads. The solve runs within this pool via `pool.scope(|s| { ... })`. When the user does not specify a thread count, Rayon's default (all logical cores) is used.

```rust
let pool = rayon::ThreadPoolBuilder::new()
    .num_threads(num_threads)
    .build()?;

pool.scope(|s| {
    for _ in 0..num_threads {
        s.spawn(|_| {
            // Each spawned task runs the pop-process-push loop
            // (see Termination Protocol)
        });
    }
});
```

Each spawned task runs the termination-aware loop described in the Termination Protocol section. All `num_threads` tasks must observe the `done` flag before the scope exits. The `rayon::scope` call blocks until all spawned tasks return, guaranteeing that no work is lost.

#### Intra-Tile Sweep Ordering

Sweep direction significantly affects convergence speed for eikonal equations. Each local Gauss-Seidel iteration within a tile traverses nodes in a different diagonal direction, cycling through all $2^N$ directions in round-robin order:

- **2D (4 directions):** $(+i,+j)$, $(+i,-j)$, $(-i,+j)$, $(-i,-j)$. For example, direction $(+i,-j)$ means iterating $i$ from low to high and $j$ from high to low.
- **3D (8 directions):** All combinations of $(\pm i, \pm j, \pm k)$.

The sweep direction for local iteration $k$ is direction index $k \mod 2^N$. This ensures that information propagating from any direction reaches interior nodes within a small number of local iterations, which is critical for tiles where the wavefront enters from an arbitrary face.

For partial tiles, the sweep uses the tile's **actual extent** (as returned by `tile_extent`, clamped to the grid boundary), not the full tile size. For example, a $(-i, -j)$ sweep on a partial tile with extent $[0, 4) \times [0, 8)$ iterates $i$ from 3 down to 0 and $j$ from 7 down to 0.

### Concurrency Safety

Tiles share boundary nodes with their neighbors. To avoid data races on shared `u` values:

1. **Exclusive tile ownership:** Each node is owned by exactly one tile (see Tile Ownership Convention). When updating a tile, boundary values from neighboring tiles are **read** but never **written**. Only nodes belonging to the tile being processed are written by that tile's owning thread.
2. **Atomic grid storage:** The travel time array uses `Box<[AtomicU64]>`. Reads use `Ordering::Relaxed` (slightly stale values are acceptable because FIM is a label-correcting algorithm that converges regardless of update order). Writes use a compare-and-exchange loop (`update_u`) with `Ordering::Release` on success, ensuring both monotonicity (values only decrease) and visibility to other threads (see Monotonic Update via Compare-and-Exchange above).
3. **Atomic bitset for activation:** The "is-queued" bitset uses `Ordering::AcqRel` for set operations and `Ordering::Release` for clear operations. The `AcqRel` on set ensures that when a tile is activated, the activating thread's writes to `u` are visible to the thread that will process the tile.
4. **In-flight counter:** Uses `AcqRel` for increment/decrement and `Acquire` for the termination check, ensuring all tile writes are visible before termination is declared.

### Progress Reporting

For large grids (especially 3D), the solver may run for an extended period. An optional progress reporting mechanism allows callers to monitor convergence without affecting default performance.

The solver accepts an optional callback of type `Option<Box<dyn Fn(ProgressInfo) + Send>>` at construction time. When provided, the scheduler invokes it periodically (at most once per wall-clock interval, default 500ms, to avoid overhead). The `ProgressInfo` struct contains:

```rust
pub struct ProgressInfo {
    /// Number of tile activations (pops from the queue) since the solve began.
    pub tiles_processed: u64,
    /// Current number of tiles in the active list.
    pub active_list_size: usize,
    /// Number of tiles currently being processed by worker threads.
    pub in_flight: usize,
    /// Wall-clock elapsed time since solve start.
    pub elapsed: std::time::Duration,
}
```

The CLI binary uses this callback to print a status line to stderr when the `--progress` flag is passed. The library API exposes the callback directly. When `None` is passed (the default), no progress tracking overhead is incurred — the counter for `tiles_processed` is not incremented and the timer is not checked.

### Input Validation

The solver constructor and CLI must validate inputs and return errors (not panic) for invalid configurations:

- Grid dimensions must be positive and at least 2 along each axis.
- Tile size must be positive and no larger than the grid size along each axis.
- Grid spacing `h` must be positive and finite.
- Slowness values must be positive and finite (checked at load time for file-based fields; checked at each node for programmatic fields).
- File-based input arrays must have dtype `f64` (or `f32`, which is promoted), shape matching `--size`, and the expected variable name (for `.mat` files). Unsupported file extensions are rejected.
- For `gradient:<v0>,<g>` mode, the velocity $v_0 + g \cdot y_{\max}$ must be positive, where $y_{\max} = (\text{shape}[d_{\text{depth}}] - 1) \cdot h$ and $d_{\text{depth}}$ is the depth axis ($y$ in 2D, $z$ in 3D). This ensures all computed velocities are positive and the resulting slowness field is finite. Both $v_0$ and $v_0 + g \cdot y_{\max}$ must be positive and finite.
- Source coordinates must lie within the grid domain: $0 \leq x_s[d] \leq (shape[d] - 1) \cdot h$ for each axis $d$.
- Tolerance must be positive and finite.

The library API uses `Result<_, EikonalError>` with a custom error enum. The CLI binary maps these to user-facing error messages via `anyhow`.

---

## Part 3: Implementation Roadmap

### Module Structure

- **`core`**: Contains the `GridData` and `TilingScheme` traits, N-dimensional coordinate/index utilities, and the concrete grid implementation (`CartesianGrid<N>` with `Box<[AtomicU64]>` / `Box<[f64]>` storage).
- **`update_kernels`**: Implements the quadratic solvers for 2D and 3D. SIMD optimization is left to LLVM auto-vectorization (the project targets stable Rust; `std::simd` is not used). Inner loops should be structured to be auto-vectorization-friendly: avoid branches, use contiguous memory access, and keep loop bodies simple.
- **`scheduler`**: Manages the Rayon thread pool, the `SegQueue` Active List, atomic bitset, in-flight counter, and the global iteration loop including the termination protocol.
- **`error`**: Defines `EikonalError` enum and `Result` type alias.
- **`io`**: Implements import and export of grid data in NumPy (`.npy`) and MAT-file v7 (`.mat`) formats. File format is inferred from the file extension. `.npy` I/O uses `ndarray-npy`. `.mat` reading uses the `matfile` crate; `.mat` writing uses a minimal in-house Level 5 MAT writer (see I/O Formats). On import, validates array shape, dtype, and value constraints. On export, converts from `Box<[AtomicU64]>` to `ndarray::Array` at the boundary.
- **`cli`**: Binary entry point using `clap` for command-line usage (see CLI section below).

### CLI Interface

The project provides both a library crate and a binary crate.

```
eikonal-fim [OPTIONS] --dim <2|3> --size <N[,N,N]> --source <x,y[,z]>

Options:
  -d, --dim <DIM>          Dimensionality (2 or 3)
  -s, --size <SIZE>         Grid size, comma-separated (e.g., 256,256 or 128,128,128)
      --source <SOURCE>     Source coordinates, comma-separated (repeatable for multiple sources)
      --spacing <H>         Grid spacing [default: 1.0]
      --slowness <MODE>     Slowness field: "uniform:<val>",
                             "gradient:<v0>,<g>" (linear velocity v0+g*y in 2D,
                             v0+g*z in 3D; slowness = 1/v),
                             "slowness-file:<path>" (load slowness from file),
                             or "velocity-file:<path>" (load velocity from file;
                             converted to slowness via f=1/v)
                             Supported file formats: .npy, .mat (see I/O Formats)
                             [default: uniform:1.0]
  -t, --tolerance <TOL>     Convergence tolerance [default: 1e-6]
      --tile-size <T>       Tile edge length [default: 8 for 2D, 4 for 3D]
      --max-local-iters <N> Max Gauss-Seidel iterations per tile activation [default: 4]
  -o, --output <PATH>       Output file path (.npy or .mat) [default: output.npy]
      --threads <N>         Number of Rayon worker threads [default: all cores]
      --max-tile-pops <N>   Safety limit on total tile pops before aborting
                             [default: 100 * num_tiles]
      --progress            Print convergence progress to stderr every 500ms
```

Note: The short flags `-h` and `-f` from the original draft are removed. `-h` conflicts with `clap`'s built-in `--help`, and `-f` is ambiguous between "file" and "field." Long-form flags are preferred for clarity.

### I/O Formats

The CLI and library support two file formats for both input (slowness/velocity fields) and output (traveltime grids). The format is inferred from the file extension.

#### NumPy `.npy` Format

- **Import/export via:** `ndarray-npy` crate.
- **Dtype:** `f64` (double-precision float). If the file contains an `f32` array, it is promoted to `f64` on load. Other dtypes are rejected with an error.
- **Shape:** Must exactly match the grid dimensions specified by `--size`. For 2D, the array shape is `[nx, ny]`; for 3D, `[nx, ny, nz]`. A shape mismatch is an error.
- **Memory layout:** C-order (row-major), which is the `ndarray-npy` default and matches NumPy's default `order='C'`.

#### MATLAB MAT-file v7 Format

- **Import via:** `matfile` crate (v0.5, pure Rust, read-only).
- **Export via:** Minimal in-house MAT v7 (Level 5) writer. The Level 5 format for a single `f64` array is straightforward: a 128-byte header, a data element tag (type `miMATRIX`), array flags, dimensions, name, and raw `f64` data in column-major order. No compression is used for the output writer; this keeps the implementation simple and the output is compatible with MATLAB `load()` and `scipy.io.loadmat()`.
- **Dtype:** `f64`. If the MAT-file contains an `f32` array, it is promoted to `f64` on load. Other dtypes are rejected.
- **Shape:** Same constraints as `.npy` — must match `--size` exactly. Note: MATLAB uses column-major (Fortran) order. The reader must transpose from column-major to the solver's row-major (C-order) layout on import; the writer must transpose back on export.
- **Variable names:** Fixed by convention:
  - **Input (slowness):** Variable name `slowness` (for `slowness-file:` mode).
  - **Input (velocity):** Variable name `velocity` (for `velocity-file:` mode).
  - **Output:** Variable name `traveltime`.
  If the expected variable name is not found in the file, the loader returns an error listing the available variable names to help the user.
- **Version:** MAT-file v7 (Level 5) only. HDF5-based v7.3 files are not supported and are rejected with an error message suggesting conversion via `save(filename, varname, '-v7')` in MATLAB or `scipy.io.savemat` in Python.

#### Velocity-to-Slowness Conversion

When using `velocity-file:`, the loader reads the velocity array $v$ and converts element-wise to slowness $f = 1/v$. Validation is performed on the velocity values *before* conversion:

- All values must be positive and finite (zero or negative velocity is physically meaningless).
- After conversion, the resulting slowness values are guaranteed to be positive and finite.

#### Output Format

The output file contains the traveltime grid $u$ as an `f64` array with the same shape as the input grid. Nodes that were never reached by the wavefront (if any) retain the value `f64::INFINITY`. For `.mat` output, infinity values are written as IEEE 754 infinity, which MATLAB reads as `Inf`.

### Verification Tests

1. **Point Source (Homogeneous):** Slowness $f=1.0$. Analytical solution is $u(x) = |x - x_s|$ (expanding circle/sphere). Assert $L_\infty$ error is $O(h)$ by comparing two grid refinements (e.g., $128^2$ vs $256^2$): the error ratio should be approximately 2.0.

2. **Linear Velocity Gradient (2D):** For a medium with linear velocity $v(y) = v_0 + g \cdot y$ (velocity increasing with depth $y$), the slowness is $f(y) = 1/v(y)$. The analytical travel time from source $(x_s, y_s)$ to receiver $(x_r, y_r)$ is:

   $$T = \frac{1}{g} \operatorname{arccosh}\!\left(1 + \frac{g^2 \left[(x_r - x_s)^2 + (y_r - y_s)^2\right]}{2\, v(y_s)\, v(y_r)}\right)$$

   Use test parameters $v_0 = 1.0$, $g = 0.5$, source at the grid center. Verify $L_\infty$ error is $O(h)$ at several receiver positions that are at least $10h$ from the source (to avoid near-source discretization artifacts).

3. **Checkerboard Slowness:** Alternating $f=1.0$ and $f=2.0$ in a checkerboard pattern with block size $\geq 4h$. Tests robustness across sharp slowness discontinuities. No analytical solution; verify convergence (Active List drains) and monotonicity ($u$ values are non-negative and finite).

4. **Multi-source:** Two point sources in a homogeneous medium ($f=1.0$); verify that the solution satisfies $u(x) = \min(|x - x_{s1}|, |x - x_{s2}|)$ with $L_\infty$ error $O(h)$.

### Benchmarks

Performance benchmarks use `criterion` and are located in `benches/`. They are **not** run as part of `cargo test`; use `cargo bench` explicitly.

- **Single-thread baseline:** Solve a $512^2$ homogeneous grid with `--threads 1`. Measures raw algorithmic throughput (nodes/second).
- **Thread scaling:** Solve a $1024^2$ homogeneous grid with 1, 2, 4, 8, and all-cores thread counts. Report wall-clock time for each. The benchmark does not assert specific speedup targets, but results should be inspected manually for near-linear scaling up to physical core count.
- **3D scaling:** Solve a $128^3$ homogeneous grid with 1 and all-cores thread counts.
- **Grid size scaling:** Solve $128^2$, $256^2$, $512^2$, $1024^2$ homogeneous grids at all-cores. Verify that wall-clock time scales roughly linearly with node count.

### Cargo.toml Specification

```toml
[package]
name = "eikonal-fim"
version = "0.1.0"
edition = "2021"

[dependencies]
ndarray = "0.15"
ndarray-npy = "0.8"
matfile = { version = "0.5", features = ["ndarray"] }
rayon = "1.10"
crossbeam-queue = "0.3"
clap = { version = "4", features = ["derive"] }
anyhow = "1"

[dev-dependencies]
approx = "0.5"
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "solver_bench"
harness = false
```

### Data Flow Diagram: Lifecycle of an Active Tile

1. **Flagged:** A thread updating Tile $A$ modifies a boundary node. It identifies Tile $B$ (neighbor sharing that boundary) and calls `bitset.try_set(B)`.
2. **Enqueued:** If `try_set` returns `true` (Tile $B$ was not already queued), Tile $B$'s ID is pushed to the `SegQueue` and the `done` flag is cleared (revoking any pending termination signal).
3. **Popped:** A Rayon worker checks the `done` flag, increments `in_flight` (`AcqRel`), and then pops the ID from the queue.
4. **Updated:** The worker loads Tile $B$'s data and performs local Gauss-Seidel iterations (up to `max_local_iters`).
5. **Propagated:** For each face of Tile $B$, if any node on that face changed by more than $\epsilon$, only the **adjacent tile sharing that face** is activated via `try_set` + `SegQueue.push`. For example, if nodes on Tile $B$'s north face changed but its east face did not, only the tile to the north is activated — not the tile to the east. In $N$ dimensions, a tile has at most $2N$ face-neighbors (fewer for tiles at the domain boundary).
6. **Deactivated:** Tile $B$'s bit is cleared (`Release`). `in_flight` is decremented (`AcqRel`). If Tile $B$ needs reactivation later, a neighbor will re-flag it.
7. **Termination:** A thread observing an empty `SegQueue` **and** `in_flight == 0` sets `done` to `true`. All threads exit their loops upon observing `done`. The `rayon::scope` blocks until all threads return.

---

## Part 4: Additional Notes

### Non-Determinism

Due to `Relaxed` atomic reads and the non-deterministic pop order of the lock-free `SegQueue`, the solver is **not** bitwise reproducible across runs with different thread counts (or even across runs with the same thread count). Different processing orders cause different intermediate rounding, leading to differences in the final solution at the level of floating-point round-off (well below the $O(h)$ discretization error). This is inherent to asynchronous parallel iterative methods. Single-threaded execution is deterministic.

### Memory Requirements

The dominant memory consumers are:

| Component | Size per node | Example: $512^2$ | Example: $256^3$ |
|-----------|--------------|-------------------|-------------------|
| Travel time (`AtomicU64`) | 8 bytes | 2 MB | 128 MB |
| Slowness (`f64`) | 8 bytes | 2 MB | 128 MB |
| Bitset (1 bit/tile) | negligible | negligible | negligible |
| **Total** | **~16 bytes/node** | **~4 MB** | **~256 MB** |

For a $1024^3$ grid (~1.07 billion nodes), the memory requirement is approximately **16 GB**. Users should ensure sufficient physical RAM; swapping will destroy performance. The implementation does not support out-of-core computation.

### Library API Example

```rust
use eikonal_fim::{CartesianGrid, FimSolver};

fn main() -> anyhow::Result<()> {
    // Create a 256x256 grid with spacing h=1.0 and uniform slowness f=1.0
    let shape = [256, 256];
    let h = 1.0;
    let slowness = vec![1.0_f64; shape[0] * shape[1]];
    let grid = CartesianGrid::<2>::new(shape, h, slowness)?;

    // Configure the solver
    let mut solver = FimSolver::new(grid, 1e-6)?
        .with_tile_size([8, 8])
        .with_max_local_iters(4)
        .with_threads(4)
        .with_max_tile_pops(200_000);  // optional safety limit

    // Add a point source at the grid center
    solver.add_source([128.0, 128.0])?;

    // Solve (optionally with progress reporting)
    solver.solve(None)?;  // None = no progress callback

    // Export result — format is inferred from extension
    solver.save("output.npy")?;   // NumPy format
    solver.save("output.mat")?;   // MAT-file v7 format

    Ok(())
}
```
