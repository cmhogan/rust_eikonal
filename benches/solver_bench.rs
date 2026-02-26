// Copyright (c) 2026, Chad Hogan
// All rights reserved.
//
// This source code is licensed under the BSD-3-Clause license found in the
// LICENSE file in the root directory of this source tree.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use eikonal_fim::core::CartesianGrid;
use eikonal_fim::scheduler::FimSolver;

fn make_solver_2d(n: usize, threads: usize) -> FimSolver<2> {
    let h = 1.0;
    let slowness = vec![1.0; n * n];
    let grid = CartesianGrid::<2>::new([n, n], h, slowness).unwrap();
    let center = (n / 2) as f64;
    let mut solver = FimSolver::new(grid, 1e-6)
        .unwrap()
        .with_tile_size([8, 8])
        .unwrap()
        .with_threads(threads);
    solver.add_source([center, center]).unwrap();
    solver
}

fn make_solver_3d(n: usize, threads: usize) -> FimSolver<3> {
    let h = 1.0;
    let slowness = vec![1.0; n * n * n];
    let grid = CartesianGrid::<3>::new([n, n, n], h, slowness).unwrap();
    let center = (n / 2) as f64;
    let mut solver = FimSolver::new(grid, 1e-6)
        .unwrap()
        .with_tile_size([4, 4, 4])
        .unwrap()
        .with_threads(threads);
    solver.add_source([center, center, center]).unwrap();
    solver
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Single-thread baseline: 512^2 homogeneous, 1 thread.
fn bench_single_thread_2d(c: &mut Criterion) {
    c.bench_function("2d_512x512_1thread", |b| {
        b.iter_with_setup(
            || make_solver_2d(512, 1),
            |solver| {
                solver.solve(None).unwrap();
                black_box(solver)
            },
        );
    });
}

/// Thread scaling: 1024^2 homogeneous with varying thread counts.
fn bench_thread_scaling_2d(c: &mut Criterion) {
    let cpus = num_cpus();
    let mut group = c.benchmark_group("thread_scaling_1024x1024");
    for &threads in &[1, 2, 4, 8] {
        if threads <= cpus {
            group.bench_function(format!("{}threads", threads), |b| {
                b.iter_with_setup(
                    || make_solver_2d(1024, threads),
                    |solver| {
                        solver.solve(None).unwrap();
                        black_box(solver)
                    },
                );
            });
        }
    }
    group.bench_function(format!("{}threads_all", cpus), |b| {
        b.iter_with_setup(
            || make_solver_2d(1024, cpus),
            |solver| {
                solver.solve(None).unwrap();
                black_box(solver)
            },
        );
    });
    group.finish();
}

/// 3D scaling: 128^3 homogeneous, 1 thread and all-cores.
fn bench_3d_scaling(c: &mut Criterion) {
    let cpus = num_cpus();
    let mut group = c.benchmark_group("3d_128x128x128");
    group.bench_function("1thread", |b| {
        b.iter_with_setup(
            || make_solver_3d(128, 1),
            |solver| {
                solver.solve(None).unwrap();
                black_box(solver)
            },
        );
    });
    group.bench_function(format!("{}threads_all", cpus), |b| {
        b.iter_with_setup(
            || make_solver_3d(128, cpus),
            |solver| {
                solver.solve(None).unwrap();
                black_box(solver)
            },
        );
    });
    group.finish();
}

/// Grid size scaling: varying 2D grids at all-cores.
fn bench_grid_size_scaling(c: &mut Criterion) {
    let cpus = num_cpus();
    let mut group = c.benchmark_group("grid_size_scaling");
    for &n in &[128, 256, 512, 1024] {
        group.bench_function(format!("{}x{}", n, n), |b| {
            b.iter_with_setup(
                || make_solver_2d(n, cpus),
                |solver| {
                    solver.solve(None).unwrap();
                    black_box(solver)
                },
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_single_thread_2d,
    bench_thread_scaling_2d,
    bench_3d_scaling,
    bench_grid_size_scaling,
);
criterion_main!(benches);
