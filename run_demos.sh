#!/usr/bin/env bash
#
# Demonstration script for the eikonal-fim solver.
# Runs four examples relevant to exploration geophysics.
#
set -euo pipefail

BINARY="./target/release/eikonal-fim"
OUTDIR="demo_output"
mkdir -p "$OUTDIR"

# Build release binary
echo "=== Building release binary ==="
cargo build --release 2>&1

NCPUS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# --------------------------------------------------------------------------
# Demo 1: Marmousi first-arrival traveltimes
# --------------------------------------------------------------------------
# The Marmousi model (marmousi_vel.npy) has shape (601, 1881), dz=dx=5 m.
# Axis 0 = depth (0..3000 m), Axis 1 = offset (0..9400 m).
# Source at the surface (z=0 m) near the center of the line (offset=4700 m).
# --------------------------------------------------------------------------
echo ""
echo "=== Demo 1: Marmousi first-arrival traveltimes ==="

if [ ! -f marmousi_vel.npy ]; then
    echo "  ERROR: marmousi_vel.npy not found. Skipping Demo 1."
    echo "  (Place the Marmousi velocity model as marmousi_vel.npy in the project root.)"
else

echo "  Running solver (${NCPUS} threads) ..."
time "$BINARY" \
    --dim 2 \
    --size 601,1881 \
    --spacing 5.0 \
    --slowness "velocity-file:marmousi_vel.npy" \
    --source "0.0,4700.0" \
    --tile-size 16 \
    --tolerance 1e-6 \
    --threads "$NCPUS" \
    --progress \
    -o "${OUTDIR}/marmousi_traveltime.npy"

echo "  Generating plot ..."
python3 - "$OUTDIR" <<'PYEOF'
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

outdir = sys.argv[1]
tt = np.load(f"{outdir}/marmousi_traveltime.npy")          # (601, 1881)
vel = np.load("marmousi_vel.npy")

dz = dx = 5.0  # metres
nz, nx = tt.shape
z = np.arange(nz) * dz
x = np.arange(nx) * dx

fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True, sharey=True)

im0 = axes[0].imshow(vel, extent=[x[0], x[-1], z[-1], z[0]],
                      aspect="auto", cmap="jet")
axes[0].set_title("Marmousi Velocity (m/s)")
axes[0].set_xlabel("Offset (m)")
axes[0].set_ylabel("Depth (m)")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

# Overlay traveltime contours on the velocity
im1 = axes[1].imshow(vel, extent=[x[0], x[-1], z[-1], z[0]],
                      aspect="auto", cmap="jet", alpha=0.4)
cs = axes[1].contour(x, z, tt, levels=30, colors="k", linewidths=0.6)
axes[1].clabel(cs, fontsize=6, fmt="%.2f")
axes[1].set_title("First-Arrival Traveltime Contours (s)")
axes[1].set_xlabel("Offset (m)")
axes[1].set_ylabel("Depth (m)")

plt.tight_layout()
plt.savefig(f"{outdir}/demo1_marmousi.png", dpi=150)
print(f"  Saved {outdir}/demo1_marmousi.png")
PYEOF

fi  # end marmousi_vel.npy check

# --------------------------------------------------------------------------
# Demo 3: Linear v(z) gradient — analytic comparison
# --------------------------------------------------------------------------
# v(z) = v0 + g*z with v0=1500 m/s, g=0.5 /s on a 512x512 grid, h=10 m.
# The CLI gradient mode varies velocity along axis 1 (depth).
# Axis 0 = offset (horizontal), Axis 1 = depth.
# Source at surface (depth=0) at mid-offset: (2550, 0).
# Analytic traveltime for v(z) = v0 + g*z, source at depth zs:
#   t = (1/g) * acosh(1 + g^2*((x-xs)^2 + (z-zs)^2) / (2*v(zs)*v(z)))
# --------------------------------------------------------------------------
echo ""
echo "=== Demo 3: Linear v(z) gradient — analytic comparison ==="
echo "  Running solver ..."
time "$BINARY" \
    --dim 2 \
    --size 512,512 \
    --spacing 10.0 \
    --slowness "gradient:1500.0,0.5" \
    --source "2550.0,0.0" \
    --tile-size 16 \
    --tolerance 1e-6 \
    --threads "$NCPUS" \
    -o "${OUTDIR}/gradient_traveltime.npy"

echo "  Computing analytic solution and plotting ..."
python3 - "$OUTDIR" <<'PYEOF'
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

outdir = sys.argv[1]
tt = np.load(f"{outdir}/gradient_traveltime.npy")  # (512, 512)

h = 10.0
n0, n1 = tt.shape   # axis 0 = offset, axis 1 = depth
v0, g = 1500.0, 0.5

# Source is at axis 0 = 255 (offset 2550 m), axis 1 = 0 (depth 0 m)
src_offset = 255.0 * h  # 2550 m
src_depth  = 0.0

offset = np.arange(n0) * h   # axis 0
depth  = np.arange(n1) * h   # axis 1

# Build 2D coordinate grids: OFFSET[i,j], DEPTH[i,j]
OFFSET, DEPTH = np.meshgrid(offset, depth, indexing="ij")
doff2 = (OFFSET - src_offset)**2
ddep2 = (DEPTH - src_depth)**2

# Analytic: t = acosh(1 + g^2*(doff^2 + ddep^2) / (2 * v(zs) * v(z))) / g
v_src = v0 + g * src_depth   # = v0 since source is at surface
v_z   = v0 + g * DEPTH
arg = 1.0 + g**2 * (doff2 + ddep2) / (2.0 * v_src * v_z)
tt_analytic = np.arccosh(arg) / g

# Mask the source neighbourhood to avoid singularity
r = np.sqrt(doff2 + ddep2)
mask = r > 3 * h

error = np.abs(tt - tt_analytic)
max_err = np.max(error[mask])
mean_err = np.mean(error[mask])
print(f"  Max absolute error  = {max_err:.6e} s")
print(f"  Mean absolute error = {mean_err:.6e} s")

# For display: imshow shows rows as vertical axis, columns as horizontal.
# axis 0 = rows = offset (vertical in imshow), axis 1 = columns = depth (horiz).
# Transpose so that offset runs horizontal and depth runs vertical (downward).
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ext = [offset[0], offset[-1], depth[-1], depth[0]]

im0 = axes[0].imshow(tt.T, extent=ext, aspect="auto", cmap="inferno")
axes[0].set_title("FIM Traveltime (s)")
axes[0].set_xlabel("Offset (m)")
axes[0].set_ylabel("Depth (m)")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(tt_analytic.T, extent=ext, aspect="auto", cmap="inferno")
axes[1].set_title("Analytic Traveltime (s)")
axes[1].set_xlabel("Offset (m)")
plt.colorbar(im1, ax=axes[1], shrink=0.8)

err_plot = np.where(mask, error, np.nan)
im2 = axes[2].imshow(err_plot.T, extent=ext, aspect="auto", cmap="hot")
axes[2].set_title(f"Absolute Error (max={max_err:.2e} s)")
axes[2].set_xlabel("Offset (m)")
plt.colorbar(im2, ax=axes[2], shrink=0.8)

plt.tight_layout()
plt.savefig(f"{outdir}/demo3_gradient.png", dpi=150)
print(f"  Saved {outdir}/demo3_gradient.png")
PYEOF

# --------------------------------------------------------------------------
# Demo 4: Salt body (high-contrast velocity inclusion)
# --------------------------------------------------------------------------
# Background velocity 2500 m/s with an elliptical "salt body" at 4500 m/s.
# Grid: 512x1024, h=10 m. Source at surface centre.
# Demonstrates wavefront healing and shadow zones.
# --------------------------------------------------------------------------
echo ""
echo "=== Demo 4: Salt body — high-contrast inclusion ==="

echo "  Building synthetic velocity model ..."
python3 - "$OUTDIR" <<'PYEOF'
import sys, numpy as np

outdir = sys.argv[1]
nz, nx = 512, 1024
h = 10.0

# Background velocity
vel = np.full((nz, nx), 2500.0, dtype=np.float64)

# Elliptical salt body centred at (z=2500, x=5120) in metres
cz, cx = 250, 512   # grid indices
rz, rx = 100, 200   # semi-axes in grid cells
Z, X = np.mgrid[0:nz, 0:nx]
ellipse = ((Z - cz) / rz)**2 + ((X - cx) / rx)**2
vel[ellipse <= 1.0] = 4500.0

# Save as slowness (1/v) for the solver
slowness = 1.0 / vel
np.save(f"{outdir}/salt_slowness.npy", slowness)
np.save(f"{outdir}/salt_velocity.npy", vel)
print(f"  Salt model: {nz}x{nx}, h={h} m")
print(f"  Background: 2500 m/s, Salt: 4500 m/s")
PYEOF

echo "  Running solver ..."
time "$BINARY" \
    --dim 2 \
    --size 512,1024 \
    --spacing 10.0 \
    --slowness "slowness-file:${OUTDIR}/salt_slowness.npy" \
    --source "0.0,5120.0" \
    --tile-size 16 \
    --tolerance 1e-6 \
    --threads "$NCPUS" \
    -o "${OUTDIR}/salt_traveltime.npy"

echo "  Generating plot ..."
python3 - "$OUTDIR" <<'PYEOF'
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

outdir = sys.argv[1]
tt = np.load(f"{outdir}/salt_traveltime.npy")
vel = np.load(f"{outdir}/salt_velocity.npy")

h = 10.0
nz, nx = tt.shape
z = np.arange(nz) * h
x = np.arange(nx) * h

fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True, sharey=True)

im0 = axes[0].imshow(vel, extent=[x[0], x[-1], z[-1], z[0]],
                      aspect="auto", cmap="seismic",
                      vmin=2000, vmax=5000)
axes[0].set_title("Velocity Model (m/s)")
axes[0].set_xlabel("Offset (m)")
axes[0].set_ylabel("Depth (m)")
plt.colorbar(im0, ax=axes[0], shrink=0.8)

# Traveltime with velocity overlay
im1 = axes[1].imshow(vel, extent=[x[0], x[-1], z[-1], z[0]],
                      aspect="auto", cmap="gray", alpha=0.3)
cs = axes[1].contour(x, z, tt, levels=40, cmap="viridis", linewidths=0.7)
axes[1].clabel(cs, fontsize=6, fmt="%.2f")
axes[1].set_title("Traveltime Contours over Salt Body")
axes[1].set_xlabel("Offset (m)")
axes[1].set_ylabel("Depth (m)")

plt.tight_layout()
plt.savefig(f"{outdir}/demo4_salt.png", dpi=150)
print(f"  Saved {outdir}/demo4_salt.png")
PYEOF

# --------------------------------------------------------------------------
# Demo 5: Performance — FIM vs scikit-fmm
# --------------------------------------------------------------------------
# Head-to-head benchmark on 2D and 3D problems of increasing size.
# scikit-fmm (C++ FMM, single-threaded) vs this Rust FIM (1 thread and all
# cores).  Produces a summary table and a bar chart.
# --------------------------------------------------------------------------
echo ""
echo "=== Demo 5: Performance — eikonal-fim vs scikit-fmm ==="

python3 - "$BINARY" "$NCPUS" "$OUTDIR" <<'PYEOF'
import subprocess, sys, time, json, os
import numpy as np

binary  = sys.argv[1]
ncpus   = int(sys.argv[2])
outdir  = sys.argv[3]

# ---- helpers ----------------------------------------------------------------

def time_skfmm(phi, speed, dx, order=1, repeats=1):
    """Return best wall-clock time (seconds) over `repeats` runs."""
    import skfmm
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        skfmm.travel_time(phi, speed, dx=dx, order=order)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best

def time_fim(dim, size_str, spacing, slowness_arg, source, threads,
             tile_size, outfile, repeats=1):
    """Return best wall-clock time (seconds) over `repeats` runs of the CLI."""
    best = float("inf")
    cmd = [
        binary,
        "--dim", str(dim),
        "--size", size_str,
        "--spacing", str(spacing),
        "--slowness", slowness_arg,
        "--source", source,
        "--tile-size", str(tile_size),
        "--tolerance", "1e-6",
        "--threads", str(threads),
        "-o", outfile,
    ]
    for _ in range(repeats):
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True, capture_output=True)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best

# ---- benchmark suite --------------------------------------------------------

results = []   # list of dicts

# --- 2D homogeneous problems -------------------------------------------------

for n in [512, 1024, 2048, 4096]:
    label = f"2D {n}x{n}"
    nodes = n * n
    c = n // 2
    print(f"  Benchmarking {label} ({nodes:,} nodes) ...")

    # skfmm
    phi = np.ones((n, n))
    phi[c, c] = -1.0
    speed = np.ones((n, n))
    t_skfmm = time_skfmm(phi, speed, dx=1.0, repeats=2)

    # FIM — 1 thread
    t_fim1 = time_fim(2, f"{n},{n}", 1.0, "uniform:1.0",
                       f"{c}.0,{c}.0", 1, 16,
                       os.path.join(outdir, "bench_tmp.npy"), repeats=2)

    # FIM — all threads
    t_fimN = time_fim(2, f"{n},{n}", 1.0, "uniform:1.0",
                       f"{c}.0,{c}.0", ncpus, 16,
                       os.path.join(outdir, "bench_tmp.npy"), repeats=2)

    results.append(dict(
        label=label, nodes=nodes,
        skfmm=t_skfmm, fim1=t_fim1, fimN=t_fimN, threads=ncpus,
    ))

# --- 2D Marmousi (heterogeneous) ---------------------------------------------

vel_path = "marmousi_vel.npy"
if os.path.exists(vel_path):
    label = "Marmousi"
    nodes = 601 * 1881
    print(f"  Benchmarking {label} ({nodes:,} nodes) ...")

    vel = np.load(vel_path)
    phi = np.ones_like(vel)
    phi[0, 940] = -5.0

    t_skfmm = time_skfmm(phi, vel, dx=5.0, repeats=2)
    t_fim1 = time_fim(2, "601,1881", 5.0,
                       f"velocity-file:{vel_path}",
                       "0.0,4700.0", 1, 16,
                       os.path.join(outdir, "bench_tmp.npy"), repeats=2)
    t_fimN = time_fim(2, "601,1881", 5.0,
                       f"velocity-file:{vel_path}",
                       "0.0,4700.0", ncpus, 16,
                       os.path.join(outdir, "bench_tmp.npy"), repeats=2)

    results.append(dict(
        label=label, nodes=nodes,
        skfmm=t_skfmm, fim1=t_fim1, fimN=t_fimN, threads=ncpus,
    ))

# --- 3D homogeneous problems -------------------------------------------------

for n in [64, 128, 256]:
    label = f"3D {n}^3"
    nodes = n ** 3
    c = n // 2
    print(f"  Benchmarking {label} ({nodes:,} nodes) ...")

    phi = np.ones((n, n, n))
    phi[c, c, c] = -1.0
    speed = np.ones((n, n, n))
    t_skfmm = time_skfmm(phi, speed, dx=1.0, repeats=1)

    t_fim1 = time_fim(3, f"{n},{n},{n}", 1.0, "uniform:1.0",
                       f"{c}.0,{c}.0,{c}.0", 1, 4,
                       os.path.join(outdir, "bench_tmp.npy"), repeats=1)

    t_fimN = time_fim(3, f"{n},{n},{n}", 1.0, "uniform:1.0",
                       f"{c}.0,{c}.0,{c}.0", ncpus, 4,
                       os.path.join(outdir, "bench_tmp.npy"), repeats=1)

    results.append(dict(
        label=label, nodes=nodes,
        skfmm=t_skfmm, fim1=t_fim1, fimN=t_fimN, threads=ncpus,
    ))

# ---- summary table ----------------------------------------------------------

print()
print(f"  Performance comparison (skfmm order=1 vs eikonal-fim, {ncpus} cores)")
print(f"  {'Problem':<25s}  {'Nodes':>12s}  {'skfmm':>8s}  {'FIM x1':>8s}  {'FIM xN':>8s}  {'Speedup':>8s}")
print(f"  {'-'*25}  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
for r in results:
    speedup = r["skfmm"] / r["fimN"]
    print(f"  {r['label']:<25s}  {r['nodes']:>12,}  {r['skfmm']:>7.3f}s  {r['fim1']:>7.3f}s  {r['fimN']:>7.3f}s  {speedup:>7.1f}x")

# ---- bar chart --------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    colors = {"skfmm": "#4e79a7", "fim1": "#f28e2b", "fimN": "#e15759"}
    bar_labels = ["skfmm", f"FIM x1", f"FIM x{ncpus}"]

    for i, (ax, r) in enumerate(zip(axes, results)):
        vals = [r["skfmm"], r["fim1"], r["fimN"]]
        bars = ax.bar(range(3), vals, color=[colors["skfmm"], colors["fim1"], colors["fimN"]])

        # Annotate speedup on the FIM-N bar
        speedup = r["skfmm"] / r["fimN"]
        ax.text(2, r["fimN"], f"{speedup:.0f}x",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

        # Annotate times on each bar
        for bar, v in zip(bars, vals):
            if v >= 1.0:
                label = f"{v:.1f}s"
            else:
                label = f"{v*1000:.0f}ms"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 0.5, label,
                    ha="center", va="center", fontsize=7, color="white",
                    fontweight="bold")

        ax.set_title(r["label"], fontsize=10)
        ax.set_xticks(range(3))
        ax.set_xticklabels(bar_labels, fontsize=7, rotation=30, ha="right")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)
        if i == 0:
            ax.set_ylabel("Wall-clock time (s)")

    fig.suptitle(f"scikit-fmm vs eikonal-fim ({ncpus}-core machine)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    outpath = os.path.join(outdir, "demo5_performance.png")
    plt.savefig(outpath, dpi=150)
    print(f"\n  Saved {outpath}")
except Exception as e:
    print(f"\n  (plot skipped: {e})")

# clean up temp file
tmp = os.path.join(outdir, "bench_tmp.npy")
if os.path.exists(tmp):
    os.remove(tmp)
PYEOF

echo ""
echo "=== All demos complete. Output in ${OUTDIR}/ ==="
ls -lh "${OUTDIR}/"*.png
