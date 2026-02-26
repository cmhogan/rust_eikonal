// Copyright (c) 2026, Chad Hogan
// All rights reserved.
//
// This source code is licensed under the BSD-3-Clause license found in the
// LICENSE file in the root directory of this source tree.

use eikonal_fim::core::{CartesianGrid, GridData};
use eikonal_fim::scheduler::FimSolver;

/// Test 1: Point Source (Homogeneous) - 2D
/// Uniform f=1.0, source at center. Analytical solution: u(x) = |x - xs|.
/// Check O(h) convergence by comparing L∞ error at two resolutions.
/// The physical domain is fixed; h decreases with refinement.
#[test]
fn point_source_homogeneous_2d_convergence() {
    let domain_size = 128.0; // fixed physical domain

    let run = |n: usize| -> (f64, f64) {
        let h = domain_size / (n - 1) as f64;
        let slowness = vec![1.0; n * n];
        let grid = CartesianGrid::<2>::new([n, n], h, slowness).unwrap();
        let center = domain_size / 2.0;
        let mut solver = FimSolver::new(grid, 1e-10 * h)
            .unwrap()
            .with_tile_size([8, 8])
            .unwrap()
            .with_threads(4);
        solver.add_source([center, center]).unwrap();
        solver.solve(None).unwrap();

        let mut max_err = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let u = solver.grid().get_u([i, j]);
                let xi = i as f64 * h;
                let yj = j as f64 * h;
                let dist = ((xi - center).powi(2) + (yj - center).powi(2)).sqrt();
                // Skip near-source nodes (within 3h)
                if dist > 3.0 * h {
                    let err = (u - dist).abs();
                    if err > max_err {
                        max_err = err;
                    }
                }
            }
        }
        (max_err, h)
    };

    let (err_128, _h1) = run(129);
    let (err_256, _h2) = run(257);

    // For O(h) convergence, error ratio should be ~2.0 (±30%)
    let ratio = err_128 / err_256;
    assert!(
        ratio > 1.4 && ratio < 2.6,
        "convergence ratio = {} (expected ~2.0, errors: 129={}, 257={})",
        ratio,
        err_128,
        err_256
    );
}

/// Test 2: Linear Velocity Gradient (2D)
/// v(y) = v0 + g*y, source at grid center.
/// Analytical traveltime via arccosh formula.
#[test]
fn linear_velocity_gradient_2d() {
    let n = 256;
    let h = 1.0;
    let v0 = 1.0;
    let g = 0.5;

    let mut slowness = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let y = j as f64 * h;
            let v = v0 + g * y;
            slowness[i * n + j] = 1.0 / v;
        }
    }

    let grid = CartesianGrid::<2>::new([n, n], h, slowness).unwrap();
    let center_x = (n / 2) as f64;
    let center_y = (n / 2) as f64;
    let mut solver = FimSolver::new(grid, 1e-10)
        .unwrap()
        .with_tile_size([8, 8])
        .unwrap()
        .with_threads(4);
    solver.add_source([center_x, center_y]).unwrap();
    solver.solve(None).unwrap();

    // Analytical travel time via arccosh formula:
    // T = (1/g) * arccosh(1 + g^2 * ((xr-xs)^2 + (yr-ys)^2) / (2 * v(ys) * v(yr)))
    let vs = v0 + g * center_y;

    let mut max_err = 0.0_f64;
    let mut test_points = 0;

    // Check at several receiver positions >= 10h from source
    for i in (0..n).step_by(16) {
        for j in (0..n).step_by(16) {
            let xr = i as f64 * h;
            let yr = j as f64 * h;
            let dx = xr - center_x;
            let dy = yr - center_y;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < 10.0 * h {
                continue;
            }

            let vr = v0 + g * yr;
            let arg = 1.0 + g * g * (dx * dx + dy * dy) / (2.0 * vs * vr);
            let analytical = arg.acosh() / g;

            let u = solver.grid().get_u([i, j]);
            let err = (u - analytical).abs();
            if err > max_err {
                max_err = err;
            }
            test_points += 1;
        }
    }

    assert!(test_points > 10, "not enough test points: {}", test_points);
    // O(h) error should be bounded
    assert!(
        max_err < 5.0 * h,
        "linear gradient max error = {} (expected O(h)=O({}))",
        max_err,
        h
    );
}

/// Test 3: Checkerboard Slowness (2D)
/// Alternating f=1.0 and f=2.0 blocks.
/// Verify solver terminates and all values are non-negative and finite.
#[test]
fn checkerboard_slowness_2d() {
    let n = 128;
    let h = 1.0;
    let block_size = 8;

    let mut slowness = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let bi = i / block_size;
            let bj = j / block_size;
            slowness[i * n + j] = if (bi + bj) % 2 == 0 { 1.0 } else { 2.0 };
        }
    }

    let grid = CartesianGrid::<2>::new([n, n], h, slowness).unwrap();
    let mut solver = FimSolver::new(grid, 1e-6)
        .unwrap()
        .with_tile_size([8, 8])
        .unwrap()
        .with_threads(4);
    solver.add_source([64.0, 64.0]).unwrap();
    solver.solve(None).unwrap();

    // Verify all values are non-negative and finite
    for i in 0..n {
        for j in 0..n {
            let u = solver.grid().get_u([i, j]);
            assert!(u >= 0.0, "negative travel time at [{}, {}]: {}", i, j, u);
            assert!(
                u.is_finite(),
                "non-finite travel time at [{}, {}]: {}",
                i,
                j,
                u
            );
        }
    }
}

/// Test 4: Multi-Source (2D)
/// Two sources in homogeneous medium.
/// u(x) ≈ min(|x-s1|, |x-s2|) with O(h) error.
#[test]
fn multi_source_2d() {
    let n = 128;
    let h = 1.0;
    let slowness = vec![1.0; n * n];

    let grid = CartesianGrid::<2>::new([n, n], h, slowness).unwrap();
    let s1 = [32.0, 64.0];
    let s2 = [96.0, 64.0];
    let mut solver = FimSolver::new(grid, 1e-8)
        .unwrap()
        .with_tile_size([8, 8])
        .unwrap()
        .with_threads(4);
    solver.add_source(s1).unwrap();
    solver.add_source(s2).unwrap();
    solver.solve(None).unwrap();

    let mut max_err = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let d1 = ((i as f64 - s1[0]).powi(2) + (j as f64 - s1[1]).powi(2)).sqrt();
            let d2 = ((i as f64 - s2[0]).powi(2) + (j as f64 - s2[1]).powi(2)).sqrt();
            let analytical = d1.min(d2);

            let u = solver.grid().get_u([i, j]);

            // Skip near-source nodes
            if d1 < 2.0 * h || d2 < 2.0 * h {
                continue;
            }

            let err = (u - analytical).abs();
            if err > max_err {
                max_err = err;
            }
        }
    }

    // O(h) error
    assert!(
        max_err < 3.0 * h,
        "multi-source max error = {} (expected O(h)=O({}))",
        max_err,
        h
    );
}

/// Test 1 3D variant: Point Source (Homogeneous) - 3D
/// Fixed physical domain, smaller grids: 33^3 and 65^3.
#[test]
fn point_source_homogeneous_3d_convergence() {
    let domain_size = 32.0;

    let run = |n: usize| -> (f64, f64) {
        let h = domain_size / (n - 1) as f64;
        let slowness = vec![1.0; n * n * n];
        let grid = CartesianGrid::<3>::new([n, n, n], h, slowness).unwrap();
        let center = domain_size / 2.0;
        let mut solver = FimSolver::new(grid, 1e-10 * h)
            .unwrap()
            .with_tile_size([4, 4, 4])
            .unwrap()
            .with_threads(4);
        solver.add_source([center, center, center]).unwrap();
        solver.solve(None).unwrap();

        let mut max_err = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let u = solver.grid().get_u([i, j, k]);
                    let xi = i as f64 * h;
                    let yj = j as f64 * h;
                    let zk = k as f64 * h;
                    let dist =
                        ((xi - center).powi(2) + (yj - center).powi(2) + (zk - center).powi(2))
                            .sqrt();
                    if dist > 3.0 * h {
                        let err = (u - dist).abs();
                        if err > max_err {
                            max_err = err;
                        }
                    }
                }
            }
        }
        (max_err, h)
    };

    let (err_33, _h1) = run(33);
    let (err_65, _h2) = run(65);

    let ratio = err_33 / err_65;
    assert!(
        ratio > 1.2 && ratio < 3.0,
        "3D convergence ratio = {} (expected ~2.0, errors: 33^3={}, 65^3={})",
        ratio,
        err_33,
        err_65
    );
}

/// Test 3 3D variant: Checkerboard Slowness - 3D
#[test]
fn checkerboard_slowness_3d() {
    let n = 32;
    let h = 1.0;
    let block_size = 4;

    let mut slowness = vec![0.0; n * n * n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let bi = i / block_size;
                let bj = j / block_size;
                let bk = k / block_size;
                slowness[i * n * n + j * n + k] = if (bi + bj + bk) % 2 == 0 { 1.0 } else { 2.0 };
            }
        }
    }

    let grid = CartesianGrid::<3>::new([n, n, n], h, slowness).unwrap();
    let mut solver = FimSolver::new(grid, 1e-6)
        .unwrap()
        .with_tile_size([4, 4, 4])
        .unwrap()
        .with_threads(4);
    solver.add_source([16.0, 16.0, 16.0]).unwrap();
    solver.solve(None).unwrap();

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let u = solver.grid().get_u([i, j, k]);
                assert!(
                    u >= 0.0,
                    "negative travel time at [{}, {}, {}]: {}",
                    i,
                    j,
                    k,
                    u
                );
                assert!(
                    u.is_finite(),
                    "non-finite travel time at [{}, {}, {}]: {}",
                    i,
                    j,
                    k,
                    u
                );
            }
        }
    }
}
