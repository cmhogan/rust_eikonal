// Copyright (c) 2026, Chad Hogan
// All rights reserved.
//
// This source code is licensed under the BSD-3-Clause license found in the
// LICENSE file in the root directory of this source tree.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crossbeam_queue::SegQueue;

use crate::core::{CartesianGrid, GridData, TilingScheme};
use crate::error::{EikonalError, Result};
use crate::update_kernels::{update_node_2d, update_node_3d};

/// Progress information passed to the optional callback.
pub struct ProgressInfo {
    /// Number of tiles processed so far.
    pub tiles_processed: u64,
    /// Current size of the active tile queue.
    pub active_list_size: usize,
    /// Number of worker threads currently processing tiles.
    pub in_flight: usize,
    /// Elapsed time since the solve started.
    pub elapsed: Duration,
}

struct AtomicBitset {
    bits: Box<[AtomicU64]>,
}

impl AtomicBitset {
    fn new(num_bits: usize) -> Self {
        let num_words = num_bits.div_ceil(64);
        let bits: Box<[AtomicU64]> = (0..num_words)
            .map(|_| AtomicU64::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        AtomicBitset { bits }
    }

    fn try_set(&self, id: usize) -> bool {
        let word = id / 64;
        let bit = 1u64 << (id % 64);
        let prev = self.bits[word].fetch_or(bit, Ordering::AcqRel);
        (prev & bit) == 0
    }

    fn clear(&self, id: usize) {
        let word = id / 64;
        let bit = 1u64 << (id % 64);
        self.bits[word].fetch_and(!bit, Ordering::Release);
    }
}

struct TileQueue {
    queue: SegQueue<usize>,
    bitset: AtomicBitset,
}

impl TileQueue {
    fn new(num_tiles: usize) -> Self {
        TileQueue {
            queue: SegQueue::new(),
            bitset: AtomicBitset::new(num_tiles),
        }
    }

    fn push_if_new(&self, tile_id: usize) -> bool {
        if self.bitset.try_set(tile_id) {
            self.queue.push(tile_id);
            true
        } else {
            false
        }
    }

    fn pop(&self) -> Option<usize> {
        if let Some(id) = self.queue.pop() {
            self.bitset.clear(id);
            Some(id)
        } else {
            None
        }
    }

    fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    fn len(&self) -> usize {
        self.queue.len()
    }
}

/// A parallel Fast Iterative Method solver for the eikonal equation.
///
/// Uses a tile-based algorithm to partition the grid and process tiles in parallel,
/// with neighbor activation to propagate updates across tile boundaries.
///
/// Note: multi-threaded solves are not bitwise reproducible across runs due to
/// non-deterministic tile processing order.
pub struct FimSolver<const N: usize> {
    grid: CartesianGrid<N>,
    tolerance: f64,
    max_local_iters: usize,
    num_threads: Option<usize>,
    max_tile_pops: Option<u64>,
    progress_callback: Option<Box<dyn Fn(ProgressInfo) + Send + Sync>>,
    initial_tiles: Vec<usize>,
}

impl<const N: usize> FimSolver<N> {
    /// Create a new FIM solver with the given grid and convergence tolerance.
    ///
    /// # Parameters
    /// - `grid`: The Cartesian grid containing the slowness field
    /// - `tolerance`: Convergence tolerance for local tile iterations (must be positive and finite)
    ///
    /// # Errors
    /// Returns an error if the tolerance is not positive and finite.
    pub fn new(grid: CartesianGrid<N>, tolerance: f64) -> Result<Self> {
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(EikonalError::InvalidTolerance(tolerance));
        }
        Ok(FimSolver {
            grid,
            tolerance,
            max_local_iters: 4,
            num_threads: None,
            max_tile_pops: None,
            progress_callback: None,
            initial_tiles: Vec::new(),
        })
    }

    /// Set a custom tile size (builder method).
    ///
    /// # Errors
    /// Returns an error if any tile size is invalid.
    pub fn with_tile_size(mut self, tile_size: [usize; N]) -> Result<Self> {
        self.grid = self.grid.with_tile_size(tile_size)?;
        Ok(self)
    }

    /// Set the maximum number of local iterations per tile visit (builder method).
    /// Default is 4.
    pub fn with_max_local_iters(mut self, max_local_iters: usize) -> Self {
        self.max_local_iters = max_local_iters;
        self
    }

    /// Set the number of worker threads (builder method).
    /// If not specified, defaults to the number of available CPU cores.
    ///
    /// Setting `threads` to 1 guarantees deterministic, reproducible results.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.num_threads = Some(threads);
        self
    }

    /// Set the maximum number of tile visits before aborting (builder method).
    /// Useful for preventing infinite loops in testing. Default is 100 times the number of tiles.
    pub fn with_max_tile_pops(mut self, max_tile_pops: u64) -> Self {
        self.max_tile_pops = Some(max_tile_pops);
        self
    }

    /// Set a progress callback that will be invoked periodically during solving (builder method).
    /// The callback receives progress information approximately every 500ms.
    pub fn with_progress(mut self, callback: Box<dyn Fn(ProgressInfo) + Send + Sync>) -> Self {
        self.progress_callback = Some(callback);
        self
    }

    /// Get a reference to the grid.
    pub fn grid(&self) -> &CartesianGrid<N> {
        &self.grid
    }

    /// Consume the solver and return the grid with computed travel times.
    pub fn into_grid(self) -> CartesianGrid<N> {
        self.grid
    }

    /// Save the travel time grid to a file. Format is inferred from the extension.
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        crate::io::save_grid(&self.grid, path.as_ref())
    }

    fn get_num_threads(&self) -> usize {
        self.num_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
    }
}

fn seed_source_nodes<const N: usize>(
    grid: &CartesianGrid<N>,
    coord: [f64; N],
    initial_tiles: &mut Vec<usize>,
) -> Result<()> {
    let shape = grid.shape();
    let h = grid.grid_spacing();
    let tile_size = grid.tile_size();

    for d in 0..N {
        let max_coord = (shape[d] - 1) as f64 * h;
        if coord[d] < 0.0 || coord[d] > max_coord {
            return Err(EikonalError::InvalidSource {
                coord: coord.to_vec(),
                reason: format!(
                    "coordinate {} on axis {} is outside domain [0, {}]",
                    coord[d], d, max_coord
                ),
            });
        }
    }

    let radius = 2.0 * h;

    let mut min_idx = [0usize; N];
    let mut max_idx = [0usize; N];
    for d in 0..N {
        let lo = ((coord[d] - radius) / h).floor().max(0.0) as usize;
        let hi = ((coord[d] + radius) / h).ceil().min((shape[d] - 1) as f64) as usize;
        min_idx[d] = lo;
        max_idx[d] = hi;
    }

    let mut nearest = [0usize; N];
    for d in 0..N {
        nearest[d] = (coord[d] / h).round() as usize;
        nearest[d] = nearest[d].min(shape[d] - 1);
    }
    let f_source = grid.get_f(nearest);

    seed_nodes_recursive(
        grid,
        &coord,
        &min_idx,
        &max_idx,
        &tile_size,
        radius,
        f_source,
        h,
        initial_tiles,
        &mut [0usize; N],
        0,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn seed_nodes_recursive<const N: usize>(
    grid: &CartesianGrid<N>,
    coord: &[f64; N],
    min_idx: &[usize; N],
    max_idx: &[usize; N],
    tile_size: &[usize; N],
    radius: f64,
    f_source: f64,
    h: f64,
    initial_tiles: &mut Vec<usize>,
    current_idx: &mut [usize; N],
    dim: usize,
) {
    if dim == N {
        let mut dist_sq = 0.0;
        for d in 0..N {
            let diff = current_idx[d] as f64 * h - coord[d];
            dist_sq += diff * diff;
        }
        let dist = dist_sq.sqrt();
        if dist <= radius {
            let u = f_source * dist;
            let idx = *current_idx;
            grid.update_u(idx, u);
            let mut tile_idx = [0usize; N];
            for d in 0..N {
                tile_idx[d] = idx[d] / tile_size[d];
            }
            let tile_id = grid.nd_to_tile_id(tile_idx);
            if !initial_tiles.contains(&tile_id) {
                initial_tiles.push(tile_id);
            }
        }
        return;
    }

    for v in min_idx[dim]..=max_idx[dim] {
        current_idx[dim] = v;
        seed_nodes_recursive(
            grid,
            coord,
            min_idx,
            max_idx,
            tile_size,
            radius,
            f_source,
            h,
            initial_tiles,
            current_idx,
            dim + 1,
        );
    }
}

/// Helper: check if a node is on a specific face of the tile extent.
fn is_on_face_2d(idx: [usize; 2], extent: &[(usize, usize); 2], axis: usize, is_low: bool) -> bool {
    if is_low {
        idx[axis] == extent[axis].0
    } else {
        idx[axis] == extent[axis].1 - 1
    }
}

fn is_on_face_3d(idx: [usize; 3], extent: &[(usize, usize); 3], axis: usize, is_low: bool) -> bool {
    if is_low {
        idx[axis] == extent[axis].0
    } else {
        idx[axis] == extent[axis].1 - 1
    }
}

impl FimSolver<2> {
    /// Add a point source at the given physical coordinates.
    /// Initializes travel times in a radius around the source and marks affected tiles.
    ///
    /// # Parameters
    /// - `coord`: Physical coordinates [x, y] of the source
    ///
    /// # Errors
    /// Returns an error if the source is outside the grid domain.
    pub fn add_source(&mut self, coord: [f64; 2]) -> Result<()> {
        seed_source_nodes(&self.grid, coord, &mut self.initial_tiles)
    }

    /// Run the parallel solver to compute travel times from all added sources.
    ///
    /// Results may differ at floating-point rounding level between multi-threaded runs.
    ///
    /// # Parameters
    /// - `progress_cb`: Optional callback for progress updates (overrides builder-set callback)
    ///
    /// # Errors
    /// Returns an error if the maximum tile visit limit is exceeded.
    pub fn solve(&self, progress_cb: Option<&(dyn Fn(ProgressInfo) + Sync)>) -> Result<()> {
        let num_tiles = self.grid.num_tiles();
        let tile_queue = TileQueue::new(num_tiles);

        for &tile_id in &self.initial_tiles {
            tile_queue.push_if_new(tile_id);
        }

        let max_tile_pops = self.max_tile_pops.unwrap_or(100 * num_tiles as u64);
        let use_progress = progress_cb.is_some() || self.progress_callback.is_some();
        let num_threads = self.get_num_threads();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| EikonalError::Other(e.to_string()))?;

        let in_flight = AtomicUsize::new(0);
        let done = AtomicBool::new(false);
        let tile_pops = AtomicU64::new(0);
        let start_time = Instant::now();
        let last_progress = AtomicU64::new(0);

        pool.scope(|s| {
            for _ in 0..num_threads {
                s.spawn(|_| loop {
                    if done.load(Ordering::Acquire) {
                        break;
                    }
                    in_flight.fetch_add(1, Ordering::AcqRel);

                    if let Some(tile_id) = tile_queue.pop() {
                        let pops = tile_pops.fetch_add(1, Ordering::Relaxed) + 1;
                        if pops > max_tile_pops {
                            done.store(true, Ordering::Release);
                            in_flight.fetch_sub(1, Ordering::AcqRel);
                            break;
                        }

                        if use_progress {
                            let elapsed_ms = start_time.elapsed().as_millis() as u64;
                            let last = last_progress.load(Ordering::Relaxed);
                            if elapsed_ms >= last + 500
                                && last_progress
                                    .compare_exchange(
                                        last,
                                        elapsed_ms,
                                        Ordering::Relaxed,
                                        Ordering::Relaxed,
                                    )
                                    .is_ok()
                            {
                                let info = ProgressInfo {
                                    tiles_processed: pops,
                                    active_list_size: tile_queue.len(),
                                    in_flight: in_flight.load(Ordering::Relaxed),
                                    elapsed: start_time.elapsed(),
                                };
                                if let Some(cb) = progress_cb {
                                    cb(info);
                                } else if let Some(cb) = &self.progress_callback {
                                    cb(info);
                                }
                            }
                        }

                        self.process_tile_2d(tile_id, &tile_queue, &done);
                        in_flight.fetch_sub(1, Ordering::AcqRel);
                    } else {
                        in_flight.fetch_sub(1, Ordering::AcqRel);
                        if in_flight.load(Ordering::Acquire) == 0 && tile_queue.is_empty() {
                            done.store(true, Ordering::Release);
                            break;
                        } else {
                            std::thread::yield_now();
                        }
                    }
                });
            }
        });

        if tile_pops.load(Ordering::Relaxed) > max_tile_pops {
            return Err(EikonalError::MaxTilePopsExceeded {
                limit: max_tile_pops,
            });
        }

        Ok(())
    }

    fn process_tile_2d(&self, tile_id: usize, tile_queue: &TileQueue, done: &AtomicBool) {
        let shape = self.grid.shape();
        let tile_idx = self.grid.tile_id_to_nd(tile_id);
        let extent = self.grid.tile_extent(tile_idx, shape);
        let tile_counts = self.grid.tile_counts();

        // Track which faces had boundary nodes change
        // [axis0_low, axis0_high, axis1_low, axis1_high]
        let mut face_changed = [false; 4];

        let num_dirs = 4;

        for k in 0..self.max_local_iters {
            let dir = k % num_dirs;
            let mut max_change: f64 = 0.0;

            let rev_i = (dir & 1) != 0;
            let rev_j = (dir & 2) != 0;

            let (i_start, i_end, i_step): (usize, usize, isize) = if rev_i {
                (extent[0].1 - 1, extent[0].0.wrapping_sub(1), -1)
            } else {
                (extent[0].0, extent[0].1, 1)
            };
            let (j_start, j_end, j_step): (usize, usize, isize) = if rev_j {
                (extent[1].1 - 1, extent[1].0.wrapping_sub(1), -1)
            } else {
                (extent[1].0, extent[1].1, 1)
            };

            let mut i = i_start;
            while i != i_end {
                let mut j = j_start;
                while j != j_end {
                    let idx = [i, j];
                    let old = self.grid.get_u(idx);
                    let new_val = update_node_2d(&self.grid, idx);
                    if new_val < old {
                        self.grid.update_u(idx, new_val);
                        let change = old - new_val;
                        if change > max_change {
                            max_change = change;
                        }
                        if change > self.tolerance {
                            // Mark affected faces
                            if is_on_face_2d(idx, &extent, 0, true) {
                                face_changed[0] = true;
                            }
                            if is_on_face_2d(idx, &extent, 0, false) {
                                face_changed[1] = true;
                            }
                            if is_on_face_2d(idx, &extent, 1, true) {
                                face_changed[2] = true;
                            }
                            if is_on_face_2d(idx, &extent, 1, false) {
                                face_changed[3] = true;
                            }
                        }
                    }
                    j = (j as isize + j_step) as usize;
                }
                i = (i as isize + i_step) as usize;
            }

            if max_change < self.tolerance {
                break;
            }
        }

        // Activate neighbor tiles for changed faces
        // Axis 0 low
        if face_changed[0] && tile_idx[0] > 0 {
            let neighbor = [tile_idx[0] - 1, tile_idx[1]];
            if tile_queue.push_if_new(self.grid.nd_to_tile_id(neighbor)) {
                done.store(false, Ordering::Release);
            }
        }
        // Axis 0 high
        if face_changed[1] && tile_idx[0] + 1 < tile_counts[0] {
            let neighbor = [tile_idx[0] + 1, tile_idx[1]];
            if tile_queue.push_if_new(self.grid.nd_to_tile_id(neighbor)) {
                done.store(false, Ordering::Release);
            }
        }
        // Axis 1 low
        if face_changed[2] && tile_idx[1] > 0 {
            let neighbor = [tile_idx[0], tile_idx[1] - 1];
            if tile_queue.push_if_new(self.grid.nd_to_tile_id(neighbor)) {
                done.store(false, Ordering::Release);
            }
        }
        // Axis 1 high
        if face_changed[3] && tile_idx[1] + 1 < tile_counts[1] {
            let neighbor = [tile_idx[0], tile_idx[1] + 1];
            if tile_queue.push_if_new(self.grid.nd_to_tile_id(neighbor)) {
                done.store(false, Ordering::Release);
            }
        }
    }
}

impl FimSolver<3> {
    /// Add a point source at the given physical coordinates.
    /// Initializes travel times in a radius around the source and marks affected tiles.
    ///
    /// # Parameters
    /// - `coord`: Physical coordinates [x, y, z] of the source
    ///
    /// # Errors
    /// Returns an error if the source is outside the grid domain.
    pub fn add_source(&mut self, coord: [f64; 3]) -> Result<()> {
        seed_source_nodes(&self.grid, coord, &mut self.initial_tiles)
    }

    /// Run the parallel solver to compute travel times from all added sources.
    ///
    /// Results may differ at floating-point rounding level between multi-threaded runs.
    ///
    /// # Parameters
    /// - `progress_cb`: Optional callback for progress updates (overrides builder-set callback)
    ///
    /// # Errors
    /// Returns an error if the maximum tile visit limit is exceeded.
    pub fn solve(&self, progress_cb: Option<&(dyn Fn(ProgressInfo) + Sync)>) -> Result<()> {
        let num_tiles = self.grid.num_tiles();
        let tile_queue = TileQueue::new(num_tiles);

        for &tile_id in &self.initial_tiles {
            tile_queue.push_if_new(tile_id);
        }

        let max_tile_pops = self.max_tile_pops.unwrap_or(100 * num_tiles as u64);
        let use_progress = progress_cb.is_some() || self.progress_callback.is_some();
        let num_threads = self.get_num_threads();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| EikonalError::Other(e.to_string()))?;

        let in_flight = AtomicUsize::new(0);
        let done = AtomicBool::new(false);
        let tile_pops = AtomicU64::new(0);
        let start_time = Instant::now();
        let last_progress = AtomicU64::new(0);

        pool.scope(|s| {
            for _ in 0..num_threads {
                s.spawn(|_| loop {
                    if done.load(Ordering::Acquire) {
                        break;
                    }
                    in_flight.fetch_add(1, Ordering::AcqRel);

                    if let Some(tile_id) = tile_queue.pop() {
                        let pops = tile_pops.fetch_add(1, Ordering::Relaxed) + 1;
                        if pops > max_tile_pops {
                            done.store(true, Ordering::Release);
                            in_flight.fetch_sub(1, Ordering::AcqRel);
                            break;
                        }

                        if use_progress {
                            let elapsed_ms = start_time.elapsed().as_millis() as u64;
                            let last = last_progress.load(Ordering::Relaxed);
                            if elapsed_ms >= last + 500
                                && last_progress
                                    .compare_exchange(
                                        last,
                                        elapsed_ms,
                                        Ordering::Relaxed,
                                        Ordering::Relaxed,
                                    )
                                    .is_ok()
                            {
                                let info = ProgressInfo {
                                    tiles_processed: pops,
                                    active_list_size: tile_queue.len(),
                                    in_flight: in_flight.load(Ordering::Relaxed),
                                    elapsed: start_time.elapsed(),
                                };
                                if let Some(cb) = progress_cb {
                                    cb(info);
                                } else if let Some(cb) = &self.progress_callback {
                                    cb(info);
                                }
                            }
                        }

                        self.process_tile_3d(tile_id, &tile_queue, &done);
                        in_flight.fetch_sub(1, Ordering::AcqRel);
                    } else {
                        in_flight.fetch_sub(1, Ordering::AcqRel);
                        if in_flight.load(Ordering::Acquire) == 0 && tile_queue.is_empty() {
                            done.store(true, Ordering::Release);
                            break;
                        } else {
                            std::thread::yield_now();
                        }
                    }
                });
            }
        });

        if tile_pops.load(Ordering::Relaxed) > max_tile_pops {
            return Err(EikonalError::MaxTilePopsExceeded {
                limit: max_tile_pops,
            });
        }

        Ok(())
    }

    fn process_tile_3d(&self, tile_id: usize, tile_queue: &TileQueue, done: &AtomicBool) {
        let shape = self.grid.shape();
        let tile_idx = self.grid.tile_id_to_nd(tile_id);
        let extent = self.grid.tile_extent(tile_idx, shape);
        let tile_counts = self.grid.tile_counts();

        // [axis0_low, axis0_high, axis1_low, axis1_high, axis2_low, axis2_high]
        let mut face_changed = [false; 6];

        let num_dirs = 8;

        for k in 0..self.max_local_iters {
            let dir = k % num_dirs;
            let mut max_change: f64 = 0.0;

            let rev_i = (dir & 1) != 0;
            let rev_j = (dir & 2) != 0;
            let rev_k = (dir & 4) != 0;

            let (i_start, i_end, i_step): (usize, usize, isize) = if rev_i {
                (extent[0].1 - 1, extent[0].0.wrapping_sub(1), -1)
            } else {
                (extent[0].0, extent[0].1, 1)
            };
            let (j_start, j_end, j_step): (usize, usize, isize) = if rev_j {
                (extent[1].1 - 1, extent[1].0.wrapping_sub(1), -1)
            } else {
                (extent[1].0, extent[1].1, 1)
            };
            let (k_start, k_end, k_step): (usize, usize, isize) = if rev_k {
                (extent[2].1 - 1, extent[2].0.wrapping_sub(1), -1)
            } else {
                (extent[2].0, extent[2].1, 1)
            };

            let mut ii = i_start;
            while ii != i_end {
                let mut jj = j_start;
                while jj != j_end {
                    let mut kk = k_start;
                    while kk != k_end {
                        let idx = [ii, jj, kk];
                        let old = self.grid.get_u(idx);
                        let new_val = update_node_3d(&self.grid, idx);
                        if new_val < old {
                            self.grid.update_u(idx, new_val);
                            let change = old - new_val;
                            if change > max_change {
                                max_change = change;
                            }
                            if change > self.tolerance {
                                if is_on_face_3d(idx, &extent, 0, true) {
                                    face_changed[0] = true;
                                }
                                if is_on_face_3d(idx, &extent, 0, false) {
                                    face_changed[1] = true;
                                }
                                if is_on_face_3d(idx, &extent, 1, true) {
                                    face_changed[2] = true;
                                }
                                if is_on_face_3d(idx, &extent, 1, false) {
                                    face_changed[3] = true;
                                }
                                if is_on_face_3d(idx, &extent, 2, true) {
                                    face_changed[4] = true;
                                }
                                if is_on_face_3d(idx, &extent, 2, false) {
                                    face_changed[5] = true;
                                }
                            }
                        }
                        kk = (kk as isize + k_step) as usize;
                    }
                    jj = (jj as isize + j_step) as usize;
                }
                ii = (ii as isize + i_step) as usize;
            }

            if max_change < self.tolerance {
                break;
            }
        }

        // Activate neighbor tiles
        for (face, &changed) in face_changed.iter().enumerate() {
            if !changed {
                continue;
            }
            let axis = face / 2;
            let is_low = (face % 2) == 0;
            let neighbor_coord = if is_low {
                if tile_idx[axis] == 0 {
                    continue;
                }
                tile_idx[axis] - 1
            } else {
                if tile_idx[axis] + 1 >= tile_counts[axis] {
                    continue;
                }
                tile_idx[axis] + 1
            };
            let mut neighbor = tile_idx;
            neighbor[axis] = neighbor_coord;
            if tile_queue.push_if_new(self.grid.nd_to_tile_id(neighbor)) {
                done.store(false, Ordering::Release);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::CartesianGrid;

    #[test]
    fn bitset_set_clear() {
        let bs = AtomicBitset::new(128);
        assert!(bs.try_set(0));
        assert!(!bs.try_set(0));
        bs.clear(0);
        assert!(bs.try_set(0));

        assert!(bs.try_set(63));
        assert!(bs.try_set(64));
        assert!(bs.try_set(127));
    }

    #[test]
    fn bitset_concurrent() {
        use std::sync::Arc;
        let bs = Arc::new(AtomicBitset::new(1024));
        let mut handles = Vec::new();
        let success_count = Arc::new(AtomicUsize::new(0));
        for _ in 0..10 {
            let bs = Arc::clone(&bs);
            let sc = Arc::clone(&success_count);
            handles.push(std::thread::spawn(move || {
                for id in 0..1024 {
                    if bs.try_set(id) {
                        sc.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(success_count.load(Ordering::Relaxed), 1024);
    }

    #[test]
    fn solve_2d_small_grid() {
        let n = 16;
        let slowness = vec![1.0; n * n];
        let grid = CartesianGrid::<2>::new([n, n], 1.0, slowness).unwrap();
        let mut solver = FimSolver::new(grid, 1e-6)
            .unwrap()
            .with_tile_size([4, 4])
            .unwrap()
            .with_threads(1);
        solver.add_source([8.0, 8.0]).unwrap();
        solver.solve(None).unwrap();

        let h = 1.0;
        let mut max_err = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let u = solver.grid().get_u([i, j]);
                let dist = ((i as f64 - 8.0).powi(2) + (j as f64 - 8.0).powi(2)).sqrt();
                if dist > 2.0 * h {
                    let err = (u - dist).abs();
                    if err > max_err {
                        max_err = err;
                    }
                }
            }
        }
        assert!(
            max_err < 2.0 * h,
            "max error {} exceeds 2h={}",
            max_err,
            2.0 * h
        );
    }

    #[test]
    fn solve_2d_multithreaded_same_result() {
        let n = 16;
        let slowness = vec![1.0; n * n];

        let grid1 = CartesianGrid::<2>::new([n, n], 1.0, slowness.clone()).unwrap();
        let mut solver1 = FimSolver::new(grid1, 1e-10)
            .unwrap()
            .with_tile_size([4, 4])
            .unwrap()
            .with_threads(1);
        solver1.add_source([8.0, 8.0]).unwrap();
        solver1.solve(None).unwrap();

        let grid2 = CartesianGrid::<2>::new([n, n], 1.0, slowness).unwrap();
        let mut solver2 = FimSolver::new(grid2, 1e-10)
            .unwrap()
            .with_tile_size([4, 4])
            .unwrap()
            .with_threads(4);
        solver2.add_source([8.0, 8.0]).unwrap();
        solver2.solve(None).unwrap();

        let mut max_diff = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let u1 = solver1.grid().get_u([i, j]);
                let u2 = solver2.grid().get_u([i, j]);
                let diff = (u1 - u2).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
        assert!(
            max_diff < 1e-6,
            "max diff between 1-thread and 4-thread: {}",
            max_diff
        );
    }

    #[test]
    fn solve_3d_small_grid() {
        let n = 8;
        let slowness = vec![1.0; n * n * n];
        let grid = CartesianGrid::<3>::new([n, n, n], 1.0, slowness).unwrap();
        let mut solver = FimSolver::new(grid, 1e-6)
            .unwrap()
            .with_tile_size([4, 4, 4])
            .unwrap()
            .with_threads(1);
        solver.add_source([4.0, 4.0, 4.0]).unwrap();
        solver.solve(None).unwrap();

        let h = 1.0;
        let mut max_err = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let u = solver.grid().get_u([i, j, k]);
                    let dist = ((i as f64 - 4.0).powi(2)
                        + (j as f64 - 4.0).powi(2)
                        + (k as f64 - 4.0).powi(2))
                    .sqrt();
                    if dist > 2.0 * h {
                        let err = (u - dist).abs();
                        if err > max_err {
                            max_err = err;
                        }
                    }
                }
            }
        }
        assert!(
            max_err < 2.0 * h,
            "3D max error {} exceeds 2h={}",
            max_err,
            2.0 * h
        );
    }

    #[test]
    fn max_tile_pops_exceeded() {
        let n = 16;
        let slowness = vec![1.0; n * n];
        let grid = CartesianGrid::<2>::new([n, n], 1.0, slowness).unwrap();
        let mut solver = FimSolver::new(grid, 1e-6)
            .unwrap()
            .with_tile_size([4, 4])
            .unwrap()
            .with_threads(1)
            .with_max_tile_pops(1);
        solver.add_source([8.0, 8.0]).unwrap();
        let result = solver.solve(None);
        assert!(matches!(
            result,
            Err(EikonalError::MaxTilePopsExceeded { .. })
        ));
    }
}
