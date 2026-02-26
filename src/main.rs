// Copyright (c) 2026, Chad Hogan
// All rights reserved.
//
// This source code is licensed under the BSD-3-Clause license found in the
// LICENSE file in the root directory of this source tree.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Parser;

use eikonal_fim::core::CartesianGrid;
use eikonal_fim::io;
use eikonal_fim::scheduler::{FimSolver, ProgressInfo};

#[derive(Parser)]
#[command(name = "eikonal-fim", about = "Fast Iterative Method eikonal solver")]
struct Cli {
    /// Dimensionality (2 or 3)
    #[arg(short = 'd', long)]
    dim: usize,

    /// Grid size, comma-separated (e.g., 256,256 or 128,128,128)
    #[arg(short = 's', long)]
    size: String,

    /// Source coordinates, comma-separated (repeatable for multiple sources)
    #[arg(long, num_args = 1)]
    source: Vec<String>,

    /// Grid spacing
    #[arg(long, default_value = "1.0")]
    spacing: f64,

    /// Slowness field: "uniform:<val>", "gradient:<v0>,<g>",
    /// "slowness-file:<path>", or "velocity-file:<path>"
    #[arg(long, default_value = "uniform:1.0")]
    slowness: String,

    /// Convergence tolerance
    #[arg(short = 't', long, default_value = "1e-6")]
    tolerance: f64,

    /// Tile edge length
    #[arg(long)]
    tile_size: Option<usize>,

    /// Max Gauss-Seidel iterations per tile activation
    #[arg(long, default_value = "4")]
    max_local_iters: usize,

    /// Output file path (.npy or .mat)
    #[arg(short = 'o', long, default_value = "output.npy")]
    output: PathBuf,

    /// Number of Rayon worker threads
    #[arg(long)]
    threads: Option<usize>,

    /// Safety limit on total tile pops before aborting
    #[arg(long)]
    max_tile_pops: Option<u64>,

    /// Print convergence progress to stderr (see --progress-interval)
    #[arg(long)]
    progress: bool,

    /// Progress reporting interval in milliseconds (used with --progress)
    #[arg(long, default_value = "500")]
    progress_interval: u64,
}

fn parse_size(s: &str, dim: usize) -> Result<Vec<usize>> {
    let parts: Vec<usize> = s
        .split(',')
        .map(|p| p.trim().parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("invalid --size: expected comma-separated integers")?;
    if parts.len() != dim {
        bail!("--size has {} components but --dim is {}", parts.len(), dim);
    }
    Ok(parts)
}

fn parse_source(s: &str, dim: usize) -> Result<Vec<f64>> {
    let parts: Vec<f64> = s
        .split(',')
        .map(|p| p.trim().parse::<f64>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("invalid --source: expected comma-separated floats")?;
    if parts.len() != dim {
        bail!(
            "--source has {} components but --dim is {}",
            parts.len(),
            dim
        );
    }
    Ok(parts)
}

fn build_slowness_field(mode: &str, shape: &[usize], h: f64, dim: usize) -> Result<Vec<f64>> {
    if let Some(val_str) = mode.strip_prefix("uniform:") {
        let val: f64 = val_str.parse().context("invalid uniform slowness value")?;
        if !val.is_finite() || val <= 0.0 {
            bail!("uniform slowness must be positive and finite, got {}", val);
        }
        let num: usize = shape.iter().product();
        return Ok(vec![val; num]);
    }

    if let Some(params) = mode.strip_prefix("gradient:") {
        let parts: Vec<&str> = params.split(',').collect();
        if parts.len() != 2 {
            bail!("gradient mode expects 'gradient:<v0>,<g>', got '{}'", mode);
        }
        let v0: f64 = parts[0].parse().context("invalid v0 in gradient")?;
        let g: f64 = parts[1].parse().context("invalid g in gradient")?;

        // depth axis: y (axis 1) in 2D, z (axis 2) in 3D
        let depth_axis = dim - 1;
        let y_max = (shape[depth_axis] - 1) as f64 * h;
        let v_min = v0;
        let v_max = v0 + g * y_max;

        if !v0.is_finite() || v_min <= 0.0 {
            bail!("gradient: v0={} must be positive and finite", v0);
        }
        if !v_max.is_finite() || v_max <= 0.0 {
            bail!(
                "gradient: v(y_max) = {} must be positive and finite (v0={}, g={}, y_max={})",
                v_max,
                v0,
                g,
                y_max
            );
        }

        let num: usize = shape.iter().product();
        let mut slowness = vec![0.0; num];

        if dim == 2 {
            let ny = shape[1];
            let nx = shape[0];
            for i in 0..nx {
                for j in 0..ny {
                    let y = j as f64 * h;
                    let v = v0 + g * y;
                    slowness[i * ny + j] = 1.0 / v;
                }
            }
        } else {
            let ny = shape[1];
            let nz = shape[2];
            let nx = shape[0];
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let z = k as f64 * h;
                        let v = v0 + g * z;
                        slowness[i * ny * nz + j * nz + k] = 1.0 / v;
                    }
                }
            }
        }

        return Ok(slowness);
    }

    if let Some(path_str) = mode.strip_prefix("slowness-file:") {
        let path = Path::new(path_str);
        return io::load_slowness(path, shape).map_err(|e| anyhow::anyhow!("{}", e));
    }

    if let Some(path_str) = mode.strip_prefix("velocity-file:") {
        let path = Path::new(path_str);
        return io::load_velocity_as_slowness(path, shape).map_err(|e| anyhow::anyhow!("{}", e));
    }

    bail!(
        "unknown --slowness mode: '{}'. Expected 'uniform:<val>', 'gradient:<v0>,<g>', \
         'slowness-file:<path>', or 'velocity-file:<path>'",
        mode
    );
}

fn run_2d(cli: &Cli, shape: [usize; 2], slowness: Vec<f64>) -> Result<()> {
    let grid = CartesianGrid::<2>::new(shape, cli.spacing, slowness)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let tile_size = cli.tile_size.unwrap_or(8);
    let mut solver = FimSolver::new(grid, cli.tolerance)
        .map_err(|e| anyhow::anyhow!("{}", e))?
        .with_tile_size([tile_size, tile_size])
        .map_err(|e| anyhow::anyhow!("{}", e))?
        .with_max_local_iters(cli.max_local_iters);

    if let Some(threads) = cli.threads {
        solver = solver.with_threads(threads);
    }
    if let Some(max_pops) = cli.max_tile_pops {
        solver = solver.with_max_tile_pops(max_pops);
    }

    for src_str in &cli.source {
        let coords = parse_source(src_str, 2)?;
        solver
            .add_source([coords[0], coords[1]])
            .map_err(|e| anyhow::anyhow!("{}", e))?;
    }

    let progress_cb: Option<Box<dyn Fn(ProgressInfo) + Sync>> = if cli.progress {
        let interval_ms = cli.progress_interval;
        let last_print = std::sync::atomic::AtomicU64::new(0);
        let start = std::time::Instant::now();
        Some(Box::new(move |info: ProgressInfo| {
            let now_ms = start.elapsed().as_millis() as u64;
            let prev = last_print.load(std::sync::atomic::Ordering::Relaxed);
            if now_ms >= prev + interval_ms {
                last_print.store(now_ms, std::sync::atomic::Ordering::Relaxed);
                eprintln!(
                    "[{:.1}s] tiles_processed={} active={} in_flight={}",
                    info.elapsed.as_secs_f64(),
                    info.tiles_processed,
                    info.active_list_size,
                    info.in_flight,
                );
            }
        }))
    } else {
        None
    };

    solver
        .solve(progress_cb.as_deref())
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    io::save_grid(solver.grid(), &cli.output).map_err(|e| anyhow::anyhow!("{}", e))?;

    Ok(())
}

fn run_3d(cli: &Cli, shape: [usize; 3], slowness: Vec<f64>) -> Result<()> {
    let grid = CartesianGrid::<3>::new(shape, cli.spacing, slowness)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let tile_size = cli.tile_size.unwrap_or(4);
    let mut solver = FimSolver::new(grid, cli.tolerance)
        .map_err(|e| anyhow::anyhow!("{}", e))?
        .with_tile_size([tile_size, tile_size, tile_size])
        .map_err(|e| anyhow::anyhow!("{}", e))?
        .with_max_local_iters(cli.max_local_iters);

    if let Some(threads) = cli.threads {
        solver = solver.with_threads(threads);
    }
    if let Some(max_pops) = cli.max_tile_pops {
        solver = solver.with_max_tile_pops(max_pops);
    }

    for src_str in &cli.source {
        let coords = parse_source(src_str, 3)?;
        solver
            .add_source([coords[0], coords[1], coords[2]])
            .map_err(|e| anyhow::anyhow!("{}", e))?;
    }

    let progress_cb: Option<Box<dyn Fn(ProgressInfo) + Sync>> = if cli.progress {
        let interval_ms = cli.progress_interval;
        let last_print = std::sync::atomic::AtomicU64::new(0);
        let start = std::time::Instant::now();
        Some(Box::new(move |info: ProgressInfo| {
            let now_ms = start.elapsed().as_millis() as u64;
            let prev = last_print.load(std::sync::atomic::Ordering::Relaxed);
            if now_ms >= prev + interval_ms {
                last_print.store(now_ms, std::sync::atomic::Ordering::Relaxed);
                eprintln!(
                    "[{:.1}s] tiles_processed={} active={} in_flight={}",
                    info.elapsed.as_secs_f64(),
                    info.tiles_processed,
                    info.active_list_size,
                    info.in_flight,
                );
            }
        }))
    } else {
        None
    };

    solver
        .solve(progress_cb.as_deref())
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    io::save_grid(solver.grid(), &cli.output).map_err(|e| anyhow::anyhow!("{}", e))?;

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.dim != 2 && cli.dim != 3 {
        bail!("--dim must be 2 or 3, got {}", cli.dim);
    }

    if cli.source.is_empty() {
        bail!("at least one --source must be specified");
    }

    let size = parse_size(&cli.size, cli.dim)?;
    let slowness = build_slowness_field(&cli.slowness, &size, cli.spacing, cli.dim)?;

    match cli.dim {
        2 => {
            let shape = [size[0], size[1]];
            run_2d(&cli, shape, slowness)?;
        }
        3 => {
            let shape = [size[0], size[1], size[2]];
            run_3d(&cli, shape, slowness)?;
        }
        _ => unreachable!(),
    }

    Ok(())
}
