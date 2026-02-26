// Copyright (c) 2026, Chad Hogan
// All rights reserved.
//
// This source code is licensed under the BSD-3-Clause license found in the
// LICENSE file in the root directory of this source tree.

use std::fmt;

/// Errors that can occur during eikonal solver setup, I/O, or execution.
#[derive(Debug)]
pub enum EikonalError {
    /// Grid shape is invalid (dimension too small).
    InvalidGridShape {
        /// The axis index.
        axis: usize,
        /// The size provided.
        size: usize,
    },
    /// Tile size is invalid (zero or exceeds grid size).
    InvalidTileSize {
        /// The axis index.
        axis: usize,
        /// The tile size provided.
        tile: usize,
        /// The grid size on that axis.
        grid: usize,
    },
    /// Grid spacing is not positive and finite.
    InvalidGridSpacing(f64),
    /// Slowness value is not positive and finite.
    InvalidSlowness {
        /// The flat index of the invalid value.
        index: usize,
        /// The invalid value.
        value: f64,
    },
    /// Source location is invalid (outside domain or other constraint).
    InvalidSource {
        /// The source coordinates.
        coord: Vec<f64>,
        /// Explanation of why it's invalid.
        reason: String,
    },
    /// Solver tolerance is not positive and finite.
    InvalidTolerance(f64),
    /// Velocity value is not positive and finite.
    InvalidVelocity {
        /// The flat index of the invalid value.
        index: usize,
        /// The invalid value.
        value: f64,
    },
    /// Array shape does not match expected shape.
    ShapeMismatch {
        /// The expected shape.
        expected: Vec<usize>,
        /// The actual shape encountered.
        got: Vec<usize>,
    },
    /// Unsupported data type in file.
    UnsupportedDtype(String),
    /// Unsupported file format (unrecognized extension).
    UnsupportedFileFormat(String),
    /// Expected MAT variable not found in file.
    MatVariableNotFound {
        /// The variable name that was requested.
        expected: String,
        /// The variable names that are available.
        available: Vec<String>,
    },
    /// Maximum tile visit limit exceeded (likely indicates non-convergence).
    MaxTilePopsExceeded {
        /// The limit that was set.
        limit: u64,
    },
    /// I/O error occurred.
    IoError(std::io::Error),
    /// Other error with a descriptive message.
    Other(String),
}

impl fmt::Display for EikonalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EikonalError::InvalidGridShape { axis, size } => {
                write!(
                    f,
                    "invalid grid shape: axis {} has size {} (must be >= 2)",
                    axis, size
                )
            }
            EikonalError::InvalidTileSize { axis, tile, grid } => {
                write!(
                    f,
                    "invalid tile size: axis {} tile size {} exceeds grid size {}",
                    axis, tile, grid
                )
            }
            EikonalError::InvalidGridSpacing(h) => {
                write!(
                    f,
                    "invalid grid spacing: {} (must be positive and finite)",
                    h
                )
            }
            EikonalError::InvalidSlowness { index, value } => {
                write!(
                    f,
                    "invalid slowness at index {}: {} (must be positive and finite)",
                    index, value
                )
            }
            EikonalError::InvalidSource { coord, reason } => {
                write!(f, "invalid source at {:?}: {}", coord, reason)
            }
            EikonalError::InvalidTolerance(tol) => {
                write!(
                    f,
                    "invalid tolerance: {} (must be positive and finite)",
                    tol
                )
            }
            EikonalError::InvalidVelocity { index, value } => {
                write!(
                    f,
                    "invalid velocity at index {}: {} (must be positive and finite)",
                    index, value
                )
            }
            EikonalError::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            EikonalError::UnsupportedDtype(dtype) => {
                write!(f, "unsupported dtype: {}", dtype)
            }
            EikonalError::UnsupportedFileFormat(ext) => {
                write!(f, "unsupported file format: {}", ext)
            }
            EikonalError::MatVariableNotFound {
                expected,
                available,
            } => {
                write!(
                    f,
                    "MAT variable '{}' not found; available variables: {:?}",
                    expected, available
                )
            }
            EikonalError::MaxTilePopsExceeded { limit } => {
                write!(f, "max tile pops exceeded: limit was {}", limit)
            }
            EikonalError::IoError(e) => write!(f, "I/O error: {}", e),
            EikonalError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for EikonalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EikonalError::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for EikonalError {
    fn from(e: std::io::Error) -> Self {
        EikonalError::IoError(e)
    }
}

/// Convenience type alias for Results with EikonalError.
pub type Result<T> = std::result::Result<T, EikonalError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_invalid_grid_shape() {
        let e = EikonalError::InvalidGridShape { axis: 0, size: 1 };
        assert_eq!(
            e.to_string(),
            "invalid grid shape: axis 0 has size 1 (must be >= 2)"
        );
    }

    #[test]
    fn display_invalid_tile_size() {
        let e = EikonalError::InvalidTileSize {
            axis: 1,
            tile: 32,
            grid: 16,
        };
        assert_eq!(
            e.to_string(),
            "invalid tile size: axis 1 tile size 32 exceeds grid size 16"
        );
    }

    #[test]
    fn display_invalid_grid_spacing() {
        let e = EikonalError::InvalidGridSpacing(-1.0);
        assert_eq!(
            e.to_string(),
            "invalid grid spacing: -1 (must be positive and finite)"
        );
    }

    #[test]
    fn display_invalid_slowness() {
        let e = EikonalError::InvalidSlowness {
            index: 5,
            value: -0.5,
        };
        assert_eq!(
            e.to_string(),
            "invalid slowness at index 5: -0.5 (must be positive and finite)"
        );
    }

    #[test]
    fn display_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let e = EikonalError::IoError(io_err);
        assert!(e.to_string().contains("file not found"));
    }

    #[test]
    fn from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        let e: EikonalError = io_err.into();
        assert!(matches!(e, EikonalError::IoError(_)));
    }

    #[test]
    fn display_max_tile_pops() {
        let e = EikonalError::MaxTilePopsExceeded { limit: 1000 };
        assert_eq!(e.to_string(), "max tile pops exceeded: limit was 1000");
    }

    #[test]
    fn display_mat_variable_not_found() {
        let e = EikonalError::MatVariableNotFound {
            expected: "slowness".to_string(),
            available: vec!["velocity".to_string(), "grid".to_string()],
        };
        assert!(e.to_string().contains("slowness"));
        assert!(e.to_string().contains("velocity"));
    }
}
