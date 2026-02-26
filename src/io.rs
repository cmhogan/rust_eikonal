// Copyright (c) 2026, Chad Hogan
// All rights reserved.
//
// This source code is licensed under the BSD-3-Clause license found in the
// LICENSE file in the root directory of this source tree.

use std::io::Write;
use std::path::Path;
use std::sync::atomic::Ordering;

use ndarray::{ArrayD, IxDyn, ShapeBuilder};

use crate::core::{CartesianGrid, GridData};
use crate::error::{EikonalError, Result};

/// Load a slowness field from a .npy file.
pub fn load_npy_slowness(path: &Path, expected_shape: &[usize]) -> Result<Vec<f64>> {
    // Try f64 first
    let arr: ArrayD<f64> = match ndarray_npy::read_npy(path) {
        Ok(a) => a,
        Err(_) => {
            // Try f32 and promote
            let arr32: ArrayD<f32> = ndarray_npy::read_npy(path)
                .map_err(|e| EikonalError::UnsupportedDtype(format!("{}", e)))?;
            arr32.mapv(|v| v as f64)
        }
    };

    let got_shape: Vec<usize> = arr.shape().to_vec();
    if got_shape != expected_shape {
        return Err(EikonalError::ShapeMismatch {
            expected: expected_shape.to_vec(),
            got: got_shape,
        });
    }

    // Ensure C-contiguous (row-major) layout before extracting raw data.
    // Fortran-order .npy files would otherwise give column-major data.
    Ok(arr.as_standard_layout().to_owned().into_raw_vec())
}

/// Save travel time data to a .npy file.
pub fn save_npy<const N: usize>(grid: &CartesianGrid<N>, path: &Path) -> Result<()> {
    let shape = grid.shape();
    let num_nodes = grid.num_nodes();

    let data: Vec<f64> = (0..num_nodes)
        .map(|i| f64::from_bits(grid.travel_time_raw()[i].load(Ordering::Relaxed)))
        .collect();

    let shape_dyn: Vec<usize> = shape.to_vec();
    let arr = ArrayD::from_shape_vec(IxDyn(&shape_dyn), data)
        .map_err(|e| EikonalError::Other(format!("shape error: {}", e)))?;

    ndarray_npy::write_npy(path, &arr)
        .map_err(|e| EikonalError::Other(format!("npy write error: {}", e)))?;

    Ok(())
}

/// Load a slowness or velocity field from a .mat file.
pub fn load_mat_field(
    path: &Path,
    variable_name: &str,
    expected_shape: &[usize],
) -> Result<Vec<f64>> {
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    let mat = matfile::MatFile::parse(&mut reader)
        .map_err(|e| EikonalError::Other(format!("MAT parse error: {}", e)))?;

    let available: Vec<String> = mat.arrays().iter().map(|a| a.name().to_string()).collect();

    let array =
        mat.find_by_name(variable_name)
            .ok_or_else(|| EikonalError::MatVariableNotFound {
                expected: variable_name.to_string(),
                available,
            })?;

    let data_f64: Vec<f64> = match array.data() {
        matfile::NumericData::Double { real, imag: _ } => real.clone(),
        matfile::NumericData::Single { real, imag: _ } => real.iter().map(|&v| v as f64).collect(),
        _ => {
            return Err(EikonalError::UnsupportedDtype(
                "MAT file array is not f64 or f32".to_string(),
            ))
        }
    };

    let mat_shape: Vec<usize> = array.size().to_vec();
    let num_elements: usize = expected_shape.iter().product();

    if data_f64.len() != num_elements {
        return Err(EikonalError::ShapeMismatch {
            expected: expected_shape.to_vec(),
            got: mat_shape,
        });
    }

    // MAT files store data in column-major order.
    // The MAT shape might be the same as expected or reversed (col-major convention).
    // We need to re-layout from column-major to row-major.
    let ndim = expected_shape.len();

    // Determine the shape in the MAT file
    let shape_matches = mat_shape == expected_shape;
    let reversed: Vec<usize> = expected_shape.iter().rev().cloned().collect();
    let shape_reversed = mat_shape == reversed;

    if !shape_matches && !shape_reversed {
        return Err(EikonalError::ShapeMismatch {
            expected: expected_shape.to_vec(),
            got: mat_shape,
        });
    }

    // Data is in column-major order with the MAT file's shape.
    // Create an array with the MAT shape in Fortran order, then reshape to our expected shape.
    let arr = ArrayD::from_shape_vec(IxDyn(&mat_shape).f(), data_f64)
        .map_err(|e| EikonalError::Other(format!("shape error: {}", e)))?;

    // If shape was reversed, transpose to get our expected shape
    let result = if shape_reversed && !shape_matches {
        let permutation: Vec<usize> = (0..ndim).rev().collect();
        let transposed = arr.permuted_axes(IxDyn(&permutation));
        transposed.as_standard_layout().to_owned().into_raw_vec()
    } else {
        arr.as_standard_layout().to_owned().into_raw_vec()
    };

    Ok(result)
}

/// Save travel time data to a .mat file (Level 5 format).
///
/// This is a minimal hand-rolled Level 5 MAT file writer for saving numeric arrays.
///
/// # Why Hand-Rolled?
///
/// The `matfile` crate (v0.5) used for reading MAT files does not support writing.
/// Its feature roadmap explicitly lists "Writing .mat files" as planned but not yet
/// implemented. Therefore, this function implements a minimal MAT-File Level 5
/// writer to enable saving solver results in MATLAB-compatible format.
///
/// # Limitations
///
/// - **No compression**: Files are written uncompressed only
/// - **Single array per file**: Only one numeric array can be saved per file
/// - **Level 5 format only**: Does not support newer Level 7/7.3 formats
/// - **Numeric arrays only**: No support for cell arrays, structures, or sparse matrices
/// - **Real data only**: No support for complex numbers
///
/// # Format Details
///
/// This implementation follows the MAT-File Level 5 specification, which uses a
/// binary format consisting of:
/// 1. A 128-byte header identifying the file as MAT-File format
/// 2. A series of data elements, each containing a type/size tag followed by data
///
/// The data is stored in column-major (Fortran) order to match MATLAB's convention.
///
/// # Reference
///
/// MAT-File Format documentation:
/// <https://www.mathworks.com/help/pdf_doc/matlab/matfile_format.pdf>
pub fn save_mat<const N: usize>(
    grid: &CartesianGrid<N>,
    path: &Path,
    var_name: &str,
) -> Result<()> {
    let shape = grid.shape();
    let num_nodes = grid.num_nodes();

    let data: Vec<f64> = (0..num_nodes)
        .map(|i| f64::from_bits(grid.travel_time_raw()[i].load(Ordering::Relaxed)))
        .collect();

    // Convert from row-major to column-major for MAT output
    let shape_vec: Vec<usize> = shape.to_vec();
    let arr = ArrayD::from_shape_vec(IxDyn(&shape_vec), data)
        .map_err(|e| EikonalError::Other(format!("shape error: {}", e)))?;

    // Get data in column-major order by transposing and reading in standard layout
    let t_arr = arr.t();
    let col_major_data: Vec<f64> = t_arr.as_standard_layout().to_owned().into_raw_vec();

    // MAT dimensions are in column-major convention (reversed from our shape)
    let mat_dims: Vec<usize> = shape_vec.iter().rev().cloned().collect();

    write_mat_level5(path, var_name, &mat_dims, &col_major_data)?;
    Ok(())
}

/// Minimal MAT-file Level 5 writer for a single f64 array.
///
/// Writes a binary MAT-File Level 5 format file containing a single numeric array.
/// The format consists of a 128-byte header followed by data elements.
fn write_mat_level5(path: &Path, var_name: &str, dimensions: &[usize], data: &[f64]) -> Result<()> {
    let file = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(file);

    // ========================================================================
    // 1. MAT-File Header (128 bytes total)
    // ========================================================================
    // The header identifies the file as a MAT-File and specifies endianness.
    //
    // Structure:
    //   Bytes 0-115   (116 bytes): Human-readable descriptive text
    //   Bytes 116-123 (8 bytes):   Subsystem data offset (unused, set to 0)
    //   Bytes 124-125 (2 bytes):   Version number (0x0100 for Level 5)
    //   Bytes 126-127 (2 bytes):   Endian indicator ("IM" for little-endian)
    //
    // Bytes 0-115: descriptive text (116 bytes)
    let desc = b"MATLAB 5.0 MAT-file, created by eikonal-fim";
    let mut header_text = [b' '; 116];
    let copy_len = desc.len().min(116);
    header_text[..copy_len].copy_from_slice(&desc[..copy_len]);
    w.write_all(&header_text)?;

    // Bytes 116-123: subsystem data offset (8 bytes, unused)
    // This field is reserved for internal use by MATLAB; we set it to 0.
    w.write_all(&[0u8; 8])?;

    // Bytes 124-125: version (0x0100)
    // Version 0x0100 identifies this as a Level 5 MAT-File.
    w.write_all(&0x0100u16.to_le_bytes())?;

    // Bytes 126-127: endian indicator "IM" (little-endian)
    // "IM" = 0x4D49 in little-endian indicates little-endian byte order.
    // Big-endian files would use "MI" = 0x4D49 in big-endian.
    w.write_all(b"IM")?;

    // ========================================================================
    // 2. Data Element: miMATRIX (type 14)
    // ========================================================================
    // MAT-File Level 5 stores data as a series of "data elements". Each element
    // consists of an 8-byte tag (type + size) followed by the data itself.
    // Data elements must be padded to 8-byte boundaries.
    //
    // A miMATRIX element (type 14) contains an array and consists of:
    //   a) Array Flags sub-element (class type, complexity, etc.)
    //   b) Dimensions Array sub-element
    //   c) Array Name sub-element
    //   d) Real Part sub-element (the actual numeric data)
    //
    // Each sub-element has its own tag (8 bytes) + data + padding to 8-byte boundary.

    // Calculate the total size of each sub-element (including tag and padding)

    // a) Array flags: tag(8) + data(8) = 16 bytes (no padding needed)
    let array_flags_total: u32 = 16;

    // b) Dimensions: tag(8) + data(4*ndim) + padding to 8-byte boundary
    let dims_data_size = (dimensions.len() * 4) as u32;
    let dims_padded = dims_data_size.div_ceil(8) * 8; // Round up to multiple of 8
    let dims_total = 8 + dims_padded;

    // c) Name: tag(8) + data + padding to 8-byte boundary
    let name_bytes = var_name.as_bytes();
    let name_data_size = name_bytes.len() as u32;
    let name_padded = name_data_size.div_ceil(8) * 8; // Round up to multiple of 8
    let name_total = 8 + name_padded;

    // d) Real data: tag(8) + data + padding to 8-byte boundary
    let real_data_size = (data.len() * 8) as u32; // 8 bytes per f64
    let real_padded = real_data_size.div_ceil(8) * 8; // Round up to multiple of 8
    let real_total = 8 + real_padded;

    // Total size of all sub-elements (this goes in the miMATRIX tag)
    let matrix_data_size = array_flags_total + dims_total + name_total + real_total;

    // Write the miMATRIX tag (8 bytes: type + size)
    // Tag format: [4-byte type][4-byte size]
    w.write_all(&14u32.to_le_bytes())?; // Data type: miMATRIX = 14
    w.write_all(&matrix_data_size.to_le_bytes())?; // Total size of all sub-elements

    // ------------------------------------------------------------------------
    // a) Array Flags Sub-element
    // ------------------------------------------------------------------------
    // Specifies the array class (double, single, int32, etc.) and properties
    // (complex, global, logical).
    //
    // Format: [tag: type=miUINT32, size=8][2 uint32 values]
    //   - First uint32: flags byte (bits 0-7) + class byte (bits 8-15)
    //   - Second uint32: reserved (set to 0)
    //
    // Class codes: mxDOUBLE_CLASS=6, mxSINGLE_CLASS=7, mxINT32_CLASS=12, etc.
    w.write_all(&6u32.to_le_bytes())?; // Tag type: miUINT32 = 6
    w.write_all(&8u32.to_le_bytes())?; // Tag size: 8 bytes of data
    w.write_all(&6u32.to_le_bytes())?; // Data: mxDOUBLE_CLASS = 6 (no flags)
    w.write_all(&0u32.to_le_bytes())?; // Data: reserved field

    // ------------------------------------------------------------------------
    // b) Dimensions Array Sub-element
    // ------------------------------------------------------------------------
    // Specifies the array dimensions as a sequence of int32 values.
    // For a 3D array of size [10, 20, 30], this would be [10, 20, 30].
    //
    // Format: [tag: type=miINT32, size][int32 array][padding]
    // Padding: Each sub-element must be padded to an 8-byte boundary.
    w.write_all(&5u32.to_le_bytes())?; // Tag type: miINT32 = 5
    w.write_all(&dims_data_size.to_le_bytes())?; // Tag size: number of bytes
    for &d in dimensions {
        w.write_all(&(d as i32).to_le_bytes())?; // Data: each dimension as int32
    }
    let dims_pad = (dims_padded - dims_data_size) as usize;
    if dims_pad > 0 {
        w.write_all(&vec![0u8; dims_pad])?; // Padding to 8-byte boundary
    }

    // ------------------------------------------------------------------------
    // c) Array Name Sub-element
    // ------------------------------------------------------------------------
    // The variable name as an ASCII string (no null terminator required).
    // MATLAB variable names must be valid identifiers.
    //
    // Format: [tag: type=miINT8, size][ASCII string][padding]
    // Padding: Must pad to 8-byte boundary.
    w.write_all(&1u32.to_le_bytes())?; // Tag type: miINT8 = 1 (byte array)
    w.write_all(&name_data_size.to_le_bytes())?; // Tag size: number of bytes
    w.write_all(name_bytes)?; // Data: ASCII variable name
    let name_pad = (name_padded - name_data_size) as usize;
    if name_pad > 0 {
        w.write_all(&vec![0u8; name_pad])?; // Padding to 8-byte boundary
    }

    // ------------------------------------------------------------------------
    // d) Real Part Sub-element
    // ------------------------------------------------------------------------
    // The actual numeric data as an array of doubles (f64).
    // Data is stored in column-major (Fortran) order to match MATLAB convention.
    // For complex arrays, this would be followed by an "Imaginary Part" sub-element.
    //
    // Format: [tag: type=miDOUBLE, size][f64 array][padding]
    // Padding: Must pad to 8-byte boundary (though f64 array is already aligned).
    w.write_all(&9u32.to_le_bytes())?; // Tag type: miDOUBLE = 9
    w.write_all(&real_data_size.to_le_bytes())?; // Tag size: number of bytes
    for &val in data {
        w.write_all(&val.to_le_bytes())?; // Data: each element as f64
    }
    let real_pad = (real_padded - real_data_size) as usize;
    if real_pad > 0 {
        w.write_all(&vec![0u8; real_pad])?; // Padding to 8-byte boundary
    }

    w.flush()?;
    Ok(())
}

/// Convert velocity field to slowness (element-wise 1/v).
pub fn velocity_to_slowness(velocity: &[f64]) -> Result<Vec<f64>> {
    let mut slowness = Vec::with_capacity(velocity.len());
    for (index, &v) in velocity.iter().enumerate() {
        if !v.is_finite() || v <= 0.0 {
            return Err(EikonalError::InvalidVelocity { index, value: v });
        }
        slowness.push(1.0 / v);
    }
    Ok(slowness)
}

/// Infer file format from extension.
pub fn infer_format(path: &Path) -> Result<FileFormat> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("npy") => Ok(FileFormat::Npy),
        Some("mat") => Ok(FileFormat::Mat),
        Some(ext) => Err(EikonalError::UnsupportedFileFormat(ext.to_string())),
        None => Err(EikonalError::UnsupportedFileFormat(
            "(no extension)".to_string(),
        )),
    }
}

/// Supported file formats for grid I/O.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FileFormat {
    /// NumPy .npy format.
    Npy,
    /// MATLAB .mat format (Level 5).
    Mat,
}

/// Save the solver's travel time to a file, inferring format from extension.
pub fn save_grid<const N: usize>(grid: &CartesianGrid<N>, path: &Path) -> Result<()> {
    match infer_format(path)? {
        FileFormat::Npy => save_npy(grid, path),
        FileFormat::Mat => save_mat(grid, path, "traveltime"),
    }
}

/// Load a slowness field from a file, inferring format from extension.
pub fn load_slowness(path: &Path, expected_shape: &[usize]) -> Result<Vec<f64>> {
    match infer_format(path)? {
        FileFormat::Npy => load_npy_slowness(path, expected_shape),
        FileFormat::Mat => load_mat_field(path, "slowness", expected_shape),
    }
}

/// Load a velocity field from a file and convert to slowness.
pub fn load_velocity_as_slowness(path: &Path, expected_shape: &[usize]) -> Result<Vec<f64>> {
    let velocity = match infer_format(path)? {
        FileFormat::Npy => load_npy_slowness(path, expected_shape)?,
        FileFormat::Mat => load_mat_field(path, "velocity", expected_shape)?,
    };
    velocity_to_slowness(&velocity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::CartesianGrid;

    fn make_test_grid_2d() -> CartesianGrid<2> {
        let n = 4;
        let slowness = vec![1.0; n * n];
        let grid = CartesianGrid::<2>::new([n, n], 1.0, slowness).unwrap();
        for i in 0..n {
            for j in 0..n {
                let val = (i * n + j) as f64;
                grid.set_u_init([i, j], val);
            }
        }
        grid
    }

    #[test]
    fn npy_roundtrip() {
        let grid = make_test_grid_2d();
        let tmp = std::env::temp_dir().join("eikonal_test_roundtrip.npy");
        save_npy(&grid, &tmp).unwrap();

        let loaded = load_npy_slowness(&tmp, &[4, 4]).unwrap();
        for i in 0..16 {
            let expected = i as f64;
            assert!(
                (loaded[i] - expected).abs() < 1e-10,
                "mismatch at {}: {} vs {}",
                i,
                loaded[i],
                expected
            );
        }
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn npy_shape_mismatch() {
        let grid = make_test_grid_2d();
        let tmp = std::env::temp_dir().join("eikonal_test_shape_mismatch.npy");
        save_npy(&grid, &tmp).unwrap();

        let result = load_npy_slowness(&tmp, &[3, 3]);
        assert!(matches!(result, Err(EikonalError::ShapeMismatch { .. })));
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn mat_roundtrip() {
        let grid = make_test_grid_2d();
        let tmp = std::env::temp_dir().join("eikonal_test_roundtrip.mat");
        save_mat(&grid, &tmp, "traveltime").unwrap();

        // Read back with matfile crate
        let file = std::fs::File::open(&tmp).unwrap();
        let mut reader = std::io::BufReader::new(file);
        let mat = matfile::MatFile::parse(&mut reader).unwrap();
        let arr = mat.find_by_name("traveltime").unwrap();

        match arr.data() {
            matfile::NumericData::Double { real, imag: _ } => {
                assert_eq!(real.len(), 16);
            }
            _ => panic!("Expected double data"),
        }

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn mat_write_read_values() {
        let grid = make_test_grid_2d();
        let tmp = std::env::temp_dir().join("eikonal_test_mat_values.mat");
        save_mat(&grid, &tmp, "traveltime").unwrap();

        // Read back and verify values match (accounting for col/row major)
        let loaded = load_mat_field(&tmp, "traveltime", &[4, 4]).unwrap();
        for i in 0..16 {
            let expected = i as f64;
            assert!(
                (loaded[i] - expected).abs() < 1e-10,
                "mat roundtrip mismatch at {}: {} vs {}",
                i,
                loaded[i],
                expected
            );
        }
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn velocity_to_slowness_valid() {
        let vel = vec![1.0, 2.0, 4.0, 0.5];
        let slow = velocity_to_slowness(&vel).unwrap();
        assert!((slow[0] - 1.0).abs() < 1e-10);
        assert!((slow[1] - 0.5).abs() < 1e-10);
        assert!((slow[2] - 0.25).abs() < 1e-10);
        assert!((slow[3] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn velocity_to_slowness_invalid() {
        let vel = vec![1.0, 0.0, 2.0];
        let result = velocity_to_slowness(&vel);
        assert!(matches!(
            result,
            Err(EikonalError::InvalidVelocity { index: 1, .. })
        ));

        let vel2 = vec![1.0, -1.0, 2.0];
        let result2 = velocity_to_slowness(&vel2);
        assert!(matches!(
            result2,
            Err(EikonalError::InvalidVelocity { index: 1, .. })
        ));
    }

    #[test]
    fn unsupported_format() {
        let path = Path::new("test.xyz");
        let result = infer_format(path);
        assert!(matches!(
            result,
            Err(EikonalError::UnsupportedFileFormat(_))
        ));
    }
}
