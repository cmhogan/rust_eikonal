// Copyright (c) 2026, Chad Hogan
// All rights reserved.
//
// This source code is licensed under the BSD-3-Clause license found in the
// LICENSE file in the root directory of this source tree.

use crate::core::GridData;

/// Solve the 2D eikonal update equation for a single node.
///
/// Given upwind neighbors a, b along each axis, effective slowness fa, fb
/// along the corresponding edges, and spacing h, solves the Godunov upwind
/// discretization: ((u-a)/h)^2 + ((u-b)/h)^2 = f_eff^2.
///
/// Falls back to 1D update if the 2D discriminant is negative or result is invalid.
pub fn solve_2d(a: f64, b: f64, fa: f64, fb: f64, h: f64) -> f64 {
    // If both are infinite, no update is possible
    if a.is_infinite() && b.is_infinite() {
        return f64::INFINITY;
    }
    // If one is infinite, 1D update from the finite one using that axis's slowness
    if a.is_infinite() {
        return b + fb * h;
    }
    if b.is_infinite() {
        return a + fa * h;
    }

    // For the 2D update, use the RMS of the two edge-averaged slowness values.
    // This is exact when fa == fb (reduces to f*h) and provides a consistent
    // isotropic effective slowness when they differ.
    let fh_sq = 0.5 * (fa * fa + fb * fb) * h * h;
    let diff = a - b;
    let disc = 2.0 * fh_sq - diff * diff;

    if disc >= 0.0 {
        let u = (a + b + disc.sqrt()) / 2.0;
        if u > a.max(b) {
            return u;
        }
    }

    // 1D fallback: use the axis with the smaller neighbor
    if a <= b {
        a + fa * h
    } else {
        b + fb * h
    }
}

/// Solve the 3D eikonal update equation for a single node.
///
/// Given upwind neighbors a, b, c along each axis, effective slowness
/// fa, fb, fc along the corresponding edges, and spacing h, attempts a 3D update.
/// Falls back to 2D or 1D update if the full 3D equation does not produce a valid result.
pub fn solve_3d(a: f64, b: f64, c: f64, fa: f64, fb: f64, fc: f64, h: f64) -> f64 {
    // Sort by neighbor value, carrying the per-axis slowness along
    let mut pairs = [(a, fa), (b, fb), (c, fc)];
    pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    let [(a_s, fa_s), (b_s, fb_s), (c_s, fc_s)] = pairs;

    // Count finite neighbors
    let finite_count = [a_s, b_s, c_s].iter().filter(|v| v.is_finite()).count();

    if finite_count == 0 {
        return f64::INFINITY;
    }

    if finite_count >= 3 {
        // Try 3D update: 3u^2 - 2(a+b+c)u + (a^2+b^2+c^2 - f_eff^2*h^2) = 0
        // where f_eff^2 = (fa^2 + fb^2 + fc^2) / 3  (RMS of edge slowness)
        let fh_sq = (fa_s * fa_s + fb_s * fb_s + fc_s * fc_s) / 3.0 * h * h;
        let sum = a_s + b_s + c_s;
        let sum_sq = a_s * a_s + b_s * b_s + c_s * c_s;
        let disc = sum * sum - 3.0 * (sum_sq - fh_sq);
        if disc >= 0.0 {
            let u = (sum + disc.sqrt()) / 3.0;
            if u > c_s {
                return u;
            }
        }
    }

    if finite_count >= 2 {
        // Try 2D update with the two smallest
        let u = solve_2d(a_s, b_s, fa_s, fb_s, h);
        if u.is_finite() && u > b_s {
            return u;
        }
    }

    // 1D fallback
    a_s + fa_s * h
}

/// Average slowness along an edge between two nodes.
#[inline]
fn edge_slowness(f_node: f64, f_neighbor: f64) -> f64 {
    0.5 * (f_node + f_neighbor)
}

/// Compute the updated travel time for a single node in a 2D grid.
///
/// Reads the 4 neighbors, picks the upwind direction per axis, computes
/// edge-averaged slowness, and calls `solve_2d`. Returns the new candidate value.
pub fn update_node_2d<G: GridData<2>>(grid: &G, idx: [usize; 2]) -> f64 {
    let shape = grid.shape();
    let h = grid.grid_spacing();
    let f = grid.get_f(idx);
    let [i, j] = idx;

    // Axis 0 (i): neighbors i-1 and i+1, pick the one with smaller traveltime
    let (a_min, fa) = {
        let (u_lo, f_lo) = if i > 0 {
            (grid.get_u([i - 1, j]), grid.get_f([i - 1, j]))
        } else {
            (f64::INFINITY, f)
        };
        let (u_hi, f_hi) = if i + 1 < shape[0] {
            (grid.get_u([i + 1, j]), grid.get_f([i + 1, j]))
        } else {
            (f64::INFINITY, f)
        };
        if u_lo <= u_hi {
            (u_lo, edge_slowness(f, f_lo))
        } else {
            (u_hi, edge_slowness(f, f_hi))
        }
    };

    // Axis 1 (j): neighbors j-1 and j+1
    let (b_min, fb) = {
        let (u_lo, f_lo) = if j > 0 {
            (grid.get_u([i, j - 1]), grid.get_f([i, j - 1]))
        } else {
            (f64::INFINITY, f)
        };
        let (u_hi, f_hi) = if j + 1 < shape[1] {
            (grid.get_u([i, j + 1]), grid.get_f([i, j + 1]))
        } else {
            (f64::INFINITY, f)
        };
        if u_lo <= u_hi {
            (u_lo, edge_slowness(f, f_lo))
        } else {
            (u_hi, edge_slowness(f, f_hi))
        }
    };

    solve_2d(a_min, b_min, fa, fb, h)
}

/// Compute the updated travel time for a single node in a 3D grid.
///
/// Reads the 6 neighbors, picks the upwind direction per axis, computes
/// edge-averaged slowness, and calls `solve_3d`. Returns the new candidate value.
pub fn update_node_3d<G: GridData<3>>(grid: &G, idx: [usize; 3]) -> f64 {
    let shape = grid.shape();
    let h = grid.grid_spacing();
    let f = grid.get_f(idx);
    let [i, j, k] = idx;

    // Axis 0 (i)
    let (a_min, fa) = {
        let (u_lo, f_lo) = if i > 0 {
            (grid.get_u([i - 1, j, k]), grid.get_f([i - 1, j, k]))
        } else {
            (f64::INFINITY, f)
        };
        let (u_hi, f_hi) = if i + 1 < shape[0] {
            (grid.get_u([i + 1, j, k]), grid.get_f([i + 1, j, k]))
        } else {
            (f64::INFINITY, f)
        };
        if u_lo <= u_hi {
            (u_lo, edge_slowness(f, f_lo))
        } else {
            (u_hi, edge_slowness(f, f_hi))
        }
    };

    // Axis 1 (j)
    let (b_min, fb) = {
        let (u_lo, f_lo) = if j > 0 {
            (grid.get_u([i, j - 1, k]), grid.get_f([i, j - 1, k]))
        } else {
            (f64::INFINITY, f)
        };
        let (u_hi, f_hi) = if j + 1 < shape[1] {
            (grid.get_u([i, j + 1, k]), grid.get_f([i, j + 1, k]))
        } else {
            (f64::INFINITY, f)
        };
        if u_lo <= u_hi {
            (u_lo, edge_slowness(f, f_lo))
        } else {
            (u_hi, edge_slowness(f, f_hi))
        }
    };

    // Axis 2 (k)
    let (c_min, fc) = {
        let (u_lo, f_lo) = if k > 0 {
            (grid.get_u([i, j, k - 1]), grid.get_f([i, j, k - 1]))
        } else {
            (f64::INFINITY, f)
        };
        let (u_hi, f_hi) = if k + 1 < shape[2] {
            (grid.get_u([i, j, k + 1]), grid.get_f([i, j, k + 1]))
        } else {
            (f64::INFINITY, f)
        };
        if u_lo <= u_hi {
            (u_lo, edge_slowness(f, f_lo))
        } else {
            (u_hi, edge_slowness(f, f_hi))
        }
    };

    solve_3d(a_min, b_min, c_min, fa, fb, fc, h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::CartesianGrid;

    #[test]
    fn solve_2d_known_case() {
        // Both neighbors at 0, f=1 on both axes, h=1
        // u = (0+0+sqrt(2))/2 = sqrt(2)/2 ≈ 0.707
        let u = solve_2d(0.0, 0.0, 1.0, 1.0, 1.0);
        assert!((u - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn solve_2d_1d_fallback_negative_discriminant() {
        // a=0, b=100, f=1, h=1: disc = 2 - 10000 < 0, falls back to 1D: 0+1 = 1
        let u = solve_2d(0.0, 100.0, 1.0, 1.0, 1.0);
        assert!((u - 1.0).abs() < 1e-10);
    }

    #[test]
    fn solve_2d_both_infinite() {
        let u = solve_2d(f64::INFINITY, f64::INFINITY, 1.0, 1.0, 1.0);
        assert!(u.is_infinite());
    }

    #[test]
    fn solve_2d_one_infinite() {
        // a=inf, b=5, fb=1, h=1 → 5 + 1 = 6
        let u = solve_2d(f64::INFINITY, 5.0, 1.0, 1.0, 1.0);
        assert!((u - 6.0).abs() < 1e-10);
    }

    #[test]
    fn solve_3d_known_case() {
        // All three neighbors at 0, f=1 on all axes, h=1
        // 3u^2 = 1 → u = 1/sqrt(3)
        let u = solve_3d(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0);
        let expected = 1.0 / 3.0_f64.sqrt();
        assert!((u - expected).abs() < 1e-10);
    }

    #[test]
    fn solve_3d_2d_fallback() {
        // a=0, b=0, c=100: 3D disc should fail since u won't be > 100
        // Falls to 2D with a=0, b=0 → sqrt(2)/2
        let u = solve_3d(0.0, 0.0, 100.0, 1.0, 1.0, 1.0, 1.0);
        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!((u - expected).abs() < 1e-10);
    }

    #[test]
    fn solve_3d_1d_fallback() {
        // a=0, b=100, c=200, f=1, h=1
        // 3D fails, 2D with a=0,b=100: disc=2-10000<0, 1D: 0+1=1
        let u = solve_3d(0.0, 100.0, 200.0, 1.0, 1.0, 1.0, 1.0);
        assert!((u - 1.0).abs() < 1e-10);
    }

    #[test]
    fn solve_3d_all_infinite() {
        let u = solve_3d(
            f64::INFINITY,
            f64::INFINITY,
            f64::INFINITY,
            1.0,
            1.0,
            1.0,
            1.0,
        );
        assert!(u.is_infinite());
    }

    #[test]
    fn update_node_2d_point_source() {
        // 5x5 grid, source at center [2,2] with u=0
        let slowness = vec![1.0; 25];
        let grid = CartesianGrid::<2>::new([5, 5], 1.0, slowness).unwrap();
        grid.set_u_init([2, 2], 0.0);

        // Node [2,1]: has neighbor at [2,2]=0, all others at inf
        // a_min = min(u[1,1], u[3,1]) = inf, b_min = min(u[2,0], u[2,2]) = 0
        // solve_2d(inf, 0, 1, 1) = 0 + 1 = 1
        let u = update_node_2d(&grid, [2, 1]);
        assert!((u - 1.0).abs() < 1e-10);

        // Diagonal node [1,1]: neighbors are all inf except none adjacent to source
        // a_min = min(u[0,1], u[2,1]) = inf, b_min = min(u[1,0], u[1,2]) = inf
        let u = update_node_2d(&grid, [1, 1]);
        assert!(u.is_infinite());
    }

    #[test]
    fn update_node_3d_basic() {
        let slowness = vec![1.0; 5 * 5 * 5];
        let grid = CartesianGrid::<3>::new([5, 5, 5], 1.0, slowness).unwrap();
        grid.set_u_init([2, 2, 2], 0.0);

        // Node [2, 2, 1]: neighbor along axis 2 at [2,2,2]=0
        let u = update_node_3d(&grid, [2, 2, 1]);
        assert!((u - 1.0).abs() < 1e-10);
    }

    #[test]
    fn no_nan_produced() {
        // Test various edge cases to ensure no NaN
        let cases = [
            (0.0, 0.0, 1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0, 1.0, 1.0),
            (f64::INFINITY, 0.0, 1.0, 1.0, 1.0),
            (0.0, f64::INFINITY, 1.0, 1.0, 1.0),
            (f64::INFINITY, f64::INFINITY, 1.0, 1.0, 1.0),
            (0.0, 0.0, 0.001, 0.001, 0.001),
            (0.0, 0.0, 1000.0, 1000.0, 1.0),
        ];
        for (a, b, fa, fb, h) in cases {
            let u = solve_2d(a, b, fa, fb, h);
            assert!(
                !u.is_nan(),
                "NaN for solve_2d({}, {}, {}, {}, {})",
                a,
                b,
                fa,
                fb,
                h
            );
        }

        let cases_3d = [
            (0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
            (f64::INFINITY, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0),
            (f64::INFINITY, f64::INFINITY, 0.0, 1.0, 1.0, 1.0, 1.0),
            (
                f64::INFINITY,
                f64::INFINITY,
                f64::INFINITY,
                1.0,
                1.0,
                1.0,
                1.0,
            ),
        ];
        for (a, b, c, fa, fb, fc, h) in cases_3d {
            let u = solve_3d(a, b, c, fa, fb, fc, h);
            assert!(
                !u.is_nan(),
                "NaN for solve_3d({}, {}, {}, {}, {}, {}, {})",
                a,
                b,
                c,
                fa,
                fb,
                fc,
                h
            );
        }
    }
}
