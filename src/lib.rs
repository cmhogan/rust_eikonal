// Copyright (c) 2026, Chad Hogan
// All rights reserved.
//
// This source code is licensed under the BSD-3-Clause license found in the
// LICENSE file in the root directory of this source tree.

//! A parallel eikonal equation solver using the Fast Iterative Method (FIM).
//!
//! This library computes first-arrival traveltimes on 2D and 3D Cartesian grids
//! by solving the eikonal equation |∇u| = f, where u is the traveltime and f
//! is the slowness field. The solver uses a tile-based parallel algorithm to
//! efficiently compute solutions on multi-core systems.

#![warn(missing_docs)]

/// Core grid data structures and traits.
pub mod core;
/// Error types for the library.
pub mod error;
/// File I/O for loading slowness fields and saving travel times.
pub mod io;
/// Parallel FIM solver implementation.
pub mod scheduler;
/// Eikonal update kernels for 2D and 3D grids.
pub mod update_kernels;

pub use crate::core::CartesianGrid;
pub use crate::error::{EikonalError, Result};
pub use crate::scheduler::{FimSolver, ProgressInfo};
