// Copyright (c) 2026, Chad Hogan
// All rights reserved.
//
// This source code is licensed under the BSD-3-Clause license found in the
// LICENSE file in the root directory of this source tree.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{EikonalError, Result};

/// Core grid data access. Provides travel time and slowness field
/// access, grid geometry, and index conversion utilities.
pub trait GridData<const N: usize> {
    /// Get the travel time value at the given N-dimensional index.
    fn get_u(&self, idx: [usize; N]) -> f64;

    /// Get the slowness value at the given N-dimensional index.
    fn get_f(&self, idx: [usize; N]) -> f64;

    /// Atomically update the travel time if the new value is smaller.
    /// Returns true if the update succeeded, false if the current value was already smaller or equal.
    fn update_u(&self, idx: [usize; N], val: f64) -> bool;

    /// Set the initial travel time value (used for seeding source nodes).
    fn set_u_init(&self, idx: [usize; N], val: f64);

    /// Get the grid shape (number of nodes along each axis).
    fn shape(&self) -> [usize; N];

    /// Get the row-major strides for index computation.
    fn strides(&self) -> [usize; N];

    /// Get the uniform grid spacing.
    fn grid_spacing(&self) -> f64;

    /// Get the total number of nodes in the grid.
    fn num_nodes(&self) -> usize;

    /// Convert a flat index to an N-dimensional index.
    fn flat_to_nd(&self, flat: usize) -> [usize; N];

    /// Convert an N-dimensional index to a flat index.
    fn nd_to_flat(&self, idx: [usize; N]) -> usize;
}

/// Tiling scheme for partitioning a grid into blocks.
pub trait TilingScheme<const N: usize> {
    /// Get the size of each tile (nodes per axis).
    fn tile_size(&self) -> [usize; N];

    /// Get the number of tiles along each axis.
    fn tile_counts(&self) -> [usize; N];

    /// Get the total number of tiles.
    fn num_tiles(&self) -> usize;

    /// Convert a flat tile ID to N-dimensional tile indices.
    fn tile_id_to_nd(&self, tile_id: usize) -> [usize; N];

    /// Convert N-dimensional tile indices to a flat tile ID.
    fn nd_to_tile_id(&self, tile_idx: [usize; N]) -> usize;

    /// Get the node index range for a tile, clamped to the grid boundaries.
    /// Returns an array of (start, end) pairs for each axis.
    fn tile_extent(&self, tile_idx: [usize; N], grid_shape: [usize; N]) -> [(usize, usize); N];
}

/// A Cartesian grid for solving the eikonal equation.
///
/// Stores the grid shape, spacing, slowness field, and travel time solution.
/// The travel time is stored atomically to support parallel updates.
/// The generic parameter `N` is the number of spatial dimensions (2 or 3).
pub struct CartesianGrid<const N: usize> {
    shape: [usize; N],
    strides: [usize; N],
    h: f64,
    travel_time: Box<[AtomicU64]>,
    slowness: Box<[f64]>,
    tile_size: [usize; N],
    tile_counts: [usize; N],
    tile_strides: [usize; N],
}

impl<const N: usize> CartesianGrid<N> {
    /// Create a new Cartesian grid with the given shape, spacing, and slowness field.
    ///
    /// # Parameters
    /// - `shape`: Number of nodes along each axis (each must be >= 2)
    /// - `h`: Uniform grid spacing (must be positive and finite)
    /// - `slowness`: Slowness values in row-major order (must all be positive and finite)
    ///
    /// # Errors
    /// Returns an error if any parameter is invalid or if the slowness vector length
    /// does not match the product of the shape dimensions.
    pub fn new(shape: [usize; N], h: f64, slowness: Vec<f64>) -> Result<Self> {
        assert!(N == 2 || N == 3, "CartesianGrid only supports N=2 or N=3");

        // Validate grid spacing
        if !h.is_finite() || h <= 0.0 {
            return Err(EikonalError::InvalidGridSpacing(h));
        }

        // Validate grid shape
        for (axis, &size) in shape.iter().enumerate() {
            if size < 2 {
                return Err(EikonalError::InvalidGridShape { axis, size });
            }
        }

        // Validate total node count
        let num_nodes: usize = shape.iter().product();
        if slowness.len() != num_nodes {
            return Err(EikonalError::ShapeMismatch {
                expected: shape.to_vec(),
                got: vec![slowness.len()],
            });
        }

        // Validate slowness values
        for (index, &value) in slowness.iter().enumerate() {
            if !value.is_finite() || value <= 0.0 {
                return Err(EikonalError::InvalidSlowness { index, value });
            }
        }

        // Compute row-major strides
        let mut strides = [0usize; N];
        strides[N - 1] = 1;
        for d in (0..N - 1).rev() {
            strides[d] = strides[d + 1] * shape[d + 1];
        }

        // Default tile size: 8 for 2D, 4 for 3D
        let default_tile = if N == 2 { 8 } else { 4 };
        let mut tile_size = [default_tile; N];
        // Clamp tile size to grid size
        for d in 0..N {
            if tile_size[d] > shape[d] {
                tile_size[d] = shape[d];
            }
        }

        let mut tile_counts = [0usize; N];
        for d in 0..N {
            tile_counts[d] = shape[d].div_ceil(tile_size[d]);
        }

        let mut tile_strides = [0usize; N];
        tile_strides[N - 1] = 1;
        for d in (0..N - 1).rev() {
            tile_strides[d] = tile_strides[d + 1] * tile_counts[d + 1];
        }

        // Initialize travel_time to INFINITY
        let travel_time: Box<[AtomicU64]> = (0..num_nodes)
            .map(|_| AtomicU64::new(f64::INFINITY.to_bits()))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Ok(CartesianGrid {
            shape,
            strides,
            h,
            travel_time,
            slowness: slowness.into_boxed_slice(),
            tile_size,
            tile_counts,
            tile_strides,
        })
    }

    /// Set a custom tile size for the grid partitioning (builder method).
    ///
    /// # Parameters
    /// - `tile_size`: Size of each tile along each axis (must be > 0 and <= grid size)
    ///
    /// # Errors
    /// Returns an error if any tile size is zero or exceeds the corresponding grid dimension.
    pub fn with_tile_size(mut self, tile_size: [usize; N]) -> Result<Self> {
        for (axis, (&tile, &grid)) in tile_size.iter().zip(self.shape.iter()).enumerate() {
            if tile == 0 || tile > grid {
                return Err(EikonalError::InvalidTileSize { axis, tile, grid });
            }
        }
        self.tile_size = tile_size;
        for d in 0..N {
            self.tile_counts[d] = self.shape[d].div_ceil(self.tile_size[d]);
        }
        self.tile_strides[N - 1] = 1;
        for d in (0..N - 1).rev() {
            self.tile_strides[d] = self.tile_strides[d + 1] * self.tile_counts[d + 1];
        }
        Ok(self)
    }

    /// Get a reference to the raw travel time storage as atomic u64 values.
    /// This is primarily used for I/O export operations.
    pub fn travel_time_raw(&self) -> &[AtomicU64] {
        &self.travel_time
    }

    /// Get a reference to the slowness field.
    pub fn slowness(&self) -> &[f64] {
        &self.slowness
    }
}

#[allow(clippy::needless_range_loop)]
impl<const N: usize> GridData<N> for CartesianGrid<N> {
    fn get_u(&self, idx: [usize; N]) -> f64 {
        let flat = self.nd_to_flat(idx);
        f64::from_bits(self.travel_time[flat].load(Ordering::Relaxed))
    }

    fn get_f(&self, idx: [usize; N]) -> f64 {
        let flat = self.nd_to_flat(idx);
        self.slowness[flat]
    }

    fn update_u(&self, idx: [usize; N], val: f64) -> bool {
        let atom = &self.travel_time[self.nd_to_flat(idx)];
        let mut current = atom.load(Ordering::Relaxed);
        loop {
            if f64::from_bits(current) <= val {
                return false;
            }
            match atom.compare_exchange_weak(
                current,
                val.to_bits(),
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
    }

    fn set_u_init(&self, idx: [usize; N], val: f64) {
        let flat = self.nd_to_flat(idx);
        self.travel_time[flat].store(val.to_bits(), Ordering::Relaxed);
    }

    fn shape(&self) -> [usize; N] {
        self.shape
    }

    fn strides(&self) -> [usize; N] {
        self.strides
    }

    fn grid_spacing(&self) -> f64 {
        self.h
    }

    fn num_nodes(&self) -> usize {
        self.shape.iter().product()
    }

    fn flat_to_nd(&self, flat: usize) -> [usize; N] {
        let mut idx = [0usize; N];
        let mut remainder = flat;
        for d in 0..N {
            idx[d] = remainder / self.strides[d];
            remainder %= self.strides[d];
        }
        idx
    }

    fn nd_to_flat(&self, idx: [usize; N]) -> usize {
        let mut flat = 0;
        for d in 0..N {
            flat += idx[d] * self.strides[d];
        }
        flat
    }
}

#[allow(clippy::needless_range_loop)]
impl<const N: usize> TilingScheme<N> for CartesianGrid<N> {
    fn tile_size(&self) -> [usize; N] {
        self.tile_size
    }

    fn tile_counts(&self) -> [usize; N] {
        self.tile_counts
    }

    fn num_tiles(&self) -> usize {
        self.tile_counts.iter().product()
    }

    fn tile_id_to_nd(&self, tile_id: usize) -> [usize; N] {
        let mut idx = [0usize; N];
        let mut remainder = tile_id;
        for d in 0..N {
            idx[d] = remainder / self.tile_strides[d];
            remainder %= self.tile_strides[d];
        }
        idx
    }

    fn nd_to_tile_id(&self, tile_idx: [usize; N]) -> usize {
        let mut id = 0;
        for d in 0..N {
            id += tile_idx[d] * self.tile_strides[d];
        }
        id
    }

    fn tile_extent(&self, tile_idx: [usize; N], grid_shape: [usize; N]) -> [(usize, usize); N] {
        let mut extent = [(0usize, 0usize); N];
        for d in 0..N {
            let start = tile_idx[d] * self.tile_size[d];
            let end = (start + self.tile_size[d]).min(grid_shape[d]);
            extent[d] = (start, end);
        }
        extent
    }
}

// SAFETY: CartesianGrid<N> can be safely sent between threads. All fields are Send:
// - `travel_time: Box<[AtomicU64]>` — AtomicU64 is Send
// - `slowness: Box<[f64]>` — Box<[f64]> is Send
// - remaining fields are plain Copy types ([usize; N], f64)
unsafe impl<const N: usize> Send for CartesianGrid<N> {}

// SAFETY: CartesianGrid<N> can be safely shared between threads:
// - `travel_time` is accessed only through AtomicU64 operations (compare_exchange_weak
//   with Release/Acquire ordering in update_u), which are inherently thread-safe
// - `slowness` is never mutated after construction (effectively immutable)
// - all other fields are Copy types that are never mutated after construction
unsafe impl<const N: usize> Sync for CartesianGrid<N> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn flat_nd_roundtrip_2d() {
        let slowness = vec![1.0; 12 * 8];
        let grid = CartesianGrid::<2>::new([12, 8], 1.0, slowness).unwrap();
        for flat in 0..96 {
            let nd = grid.flat_to_nd(flat);
            assert_eq!(grid.nd_to_flat(nd), flat, "flat={} nd={:?}", flat, nd);
        }
    }

    #[test]
    fn flat_nd_roundtrip_3d() {
        let slowness = vec![1.0; 4 * 5 * 6];
        let grid = CartesianGrid::<3>::new([4, 5, 6], 1.0, slowness).unwrap();
        for flat in 0..120 {
            let nd = grid.flat_to_nd(flat);
            assert_eq!(grid.nd_to_flat(nd), flat);
        }
    }

    #[test]
    fn tile_id_nd_roundtrip_2d() {
        let slowness = vec![1.0; 20 * 20];
        let grid = CartesianGrid::<2>::new([20, 20], 1.0, slowness).unwrap();
        let num_tiles = grid.num_tiles();
        for id in 0..num_tiles {
            let nd = grid.tile_id_to_nd(id);
            assert_eq!(grid.nd_to_tile_id(nd), id);
        }
    }

    #[test]
    fn tile_id_nd_roundtrip_3d() {
        let slowness = vec![1.0; 12 * 12 * 12];
        let grid = CartesianGrid::<3>::new([12, 12, 12], 1.0, slowness).unwrap();
        let num_tiles = grid.num_tiles();
        for id in 0..num_tiles {
            let nd = grid.tile_id_to_nd(id);
            assert_eq!(grid.nd_to_tile_id(nd), id);
        }
    }

    #[test]
    fn tile_extent_full_tile() {
        let slowness = vec![1.0; 16 * 16];
        let grid = CartesianGrid::<2>::new([16, 16], 1.0, slowness).unwrap();
        let extent = grid.tile_extent([0, 0], grid.shape());
        assert_eq!(extent, [(0, 8), (0, 8)]);
    }

    #[test]
    fn tile_extent_partial_tile() {
        // 20x20 grid with tile size 8: tiles are 0..8, 8..16, 16..20
        let slowness = vec![1.0; 20 * 20];
        let grid = CartesianGrid::<2>::new([20, 20], 1.0, slowness).unwrap();
        // Last tile along axis 0
        let extent = grid.tile_extent([2, 0], grid.shape());
        assert_eq!(extent, [(16, 20), (0, 8)]);
        // Last tile along both axes
        let extent = grid.tile_extent([2, 2], grid.shape());
        assert_eq!(extent, [(16, 20), (16, 20)]);
    }

    #[test]
    fn update_u_monotonicity() {
        let slowness = vec![1.0; 4 * 4];
        let grid = CartesianGrid::<2>::new([4, 4], 1.0, slowness).unwrap();
        let idx = [1, 1];

        // Start at infinity, decrease
        assert!(grid.update_u(idx, 10.0));
        assert_eq!(grid.get_u(idx), 10.0);

        assert!(grid.update_u(idx, 5.0));
        assert_eq!(grid.get_u(idx), 5.0);

        // Try to increase — should fail
        assert!(!grid.update_u(idx, 7.0));
        assert_eq!(grid.get_u(idx), 5.0);

        // Equal value — should fail
        assert!(!grid.update_u(idx, 5.0));
        assert_eq!(grid.get_u(idx), 5.0);

        // Decrease further
        assert!(grid.update_u(idx, 3.0));
        assert_eq!(grid.get_u(idx), 3.0);
    }

    #[test]
    fn cas_concurrent_monotonicity() {
        let slowness = vec![1.0; 4 * 4];
        let grid = Arc::new(CartesianGrid::<2>::new([4, 4], 1.0, slowness).unwrap());
        let idx = [1, 1];

        // Spawn threads that concurrently try to update the same cell
        let mut handles = Vec::new();
        for i in 0..10 {
            let grid = Arc::clone(&grid);
            handles.push(std::thread::spawn(move || {
                for j in 0..100 {
                    let val = 1000.0 - (i * 100 + j) as f64;
                    grid.update_u(idx, val);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        // The final value should be the minimum written: 1000 - 999 = 1.0
        assert_eq!(grid.get_u(idx), 1.0);
    }

    #[test]
    fn invalid_grid_shape() {
        let result = CartesianGrid::<2>::new([1, 10], 1.0, vec![1.0; 10]);
        assert!(matches!(
            result,
            Err(EikonalError::InvalidGridShape { axis: 0, size: 1 })
        ));
    }

    #[test]
    fn invalid_grid_spacing() {
        let result = CartesianGrid::<2>::new([4, 4], 0.0, vec![1.0; 16]);
        assert!(matches!(result, Err(EikonalError::InvalidGridSpacing(_))));
    }

    #[test]
    fn invalid_slowness_value() {
        let mut slowness = vec![1.0; 16];
        slowness[5] = -1.0;
        let result = CartesianGrid::<2>::new([4, 4], 1.0, slowness);
        assert!(matches!(
            result,
            Err(EikonalError::InvalidSlowness { index: 5, .. })
        ));
    }

    #[test]
    fn shape_mismatch() {
        let result = CartesianGrid::<2>::new([4, 4], 1.0, vec![1.0; 10]);
        assert!(matches!(result, Err(EikonalError::ShapeMismatch { .. })));
    }

    #[test]
    fn with_tile_size_valid() {
        let slowness = vec![1.0; 16 * 16];
        let grid = CartesianGrid::<2>::new([16, 16], 1.0, slowness)
            .unwrap()
            .with_tile_size([4, 4])
            .unwrap();
        assert_eq!(grid.tile_size(), [4, 4]);
        assert_eq!(grid.tile_counts(), [4, 4]);
    }

    #[test]
    fn with_tile_size_invalid() {
        let slowness = vec![1.0; 16 * 16];
        let grid = CartesianGrid::<2>::new([16, 16], 1.0, slowness).unwrap();
        let result = grid.with_tile_size([0, 4]);
        assert!(matches!(
            result,
            Err(EikonalError::InvalidTileSize { axis: 0, .. })
        ));
    }
}
