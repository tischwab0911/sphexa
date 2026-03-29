#!/usr/bin/env python3
"""Plot distributed FMM vs Barnes-Hut scaling from benchmark output.

Usage:
    python scripts/plot_distributed_scaling.py <output-file>
    # produces a PDF with the same basename next to the input file
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_output(path):
    """Parse structured benchmark output into a list of dicts."""
    text = Path(path).read_text()
    blocks = re.split(r"===\s*Distributed FMM vs BH Scaling\s*===", text)
    records = []
    for block in blocks:
        if not block.strip():
            continue
        def grab(pattern, s=block):
            m = re.search(pattern, s)
            return m if m is None else m
        # header
        m = re.search(r"N=(\d+),\s*numRanks=(\d+),\s*particlesPerRank=(\d+),\s*theta=([\d.]+)", block)
        if m is None:
            continue
        rec = {
            "N": int(m.group(1)),
            "numRanks": int(m.group(2)),
            "ppr": int(m.group(3)),
            "theta": float(m.group(4)),
        }
        # timing lines: "BH  p50 [ms]           : min=7.487    max=9.290    mean=8.308"
        for key in ["BH  p50", "BH  p90", "FMM p50", "FMM p90"]:
            m = re.search(
                re.escape(key) + r"\s*\[ms\]\s*:\s*min=([\d.]+)\s+max=([\d.]+)\s+mean=([\d.]+)",
                block,
            )
            if m:
                tag = key.replace("  ", "_").replace(" ", "_")
                rec[f"{tag}_min"] = float(m.group(1))
                rec[f"{tag}_max"] = float(m.group(2))
                rec[f"{tag}_mean"] = float(m.group(3))
        # speedup
        m = re.search(r"Speedup.*?:\s*min=([\d.]+)x\s+max=([\d.]+)x\s+mean=([\d.]+)x", block)
        if m:
            rec["speedup_min"] = float(m.group(1))
            rec["speedup_max"] = float(m.group(2))
            rec["speedup_mean"] = float(m.group(3))
        # error
        m = re.search(r"Err p99.*?:\s*max=([\d.eE+-]+)", block)
        if m:
            rec["err_p99_max"] = float(m.group(1))
        records.append(rec)
    records.sort(key=lambda r: r["ppr"])
    return records


def human_count(n):
    """Format particle count as human-readable string."""
    if n >= 1e9:
        return f"{n / 1e9:.0f}B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}K"
    return str(n)


def fit_models(x, y):
    """Fit O(N), O(N·log N), and power-law models.

    Returns (c_lin, r2_lin, c_nlogn, r2_nlogn, c_pow, alpha, r2_pow).
    """
    log_x = np.log(x)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    # t = c * N (linear in c)
    c_lin = np.dot(x, y) / np.dot(x, x)
    r2_lin = 1 - np.sum((y - c_lin * x) ** 2) / ss_tot

    # t = c * N * log(N) (linear in c)
    xlogx = x * log_x
    c_nlogn = np.dot(xlogx, y) / np.dot(xlogx, xlogx)
    r2_nlogn = 1 - np.sum((y - c_nlogn * xlogx) ** 2) / ss_tot

    # t = c * N^alpha (power-law, linear regression in log-log space)
    log_y = np.log(y)
    alpha, log_c = np.polyfit(log_x, log_y, 1)
    c_pow = np.exp(log_c)
    y_pred = c_pow * x ** alpha
    r2_pow = 1 - np.sum((y - y_pred) ** 2) / ss_tot

    return c_lin, r2_lin, c_nlogn, r2_nlogn, c_pow, alpha, r2_pow


def data_crossover(ppr, speedup_mean):
    """Find crossover N by log-interpolating where speedup crosses 1.0×."""
    for i in range(len(speedup_mean) - 1):
        s0, s1 = speedup_mean[i], speedup_mean[i + 1]
        if (s0 - 1.0) * (s1 - 1.0) <= 0:
            # log-linear interpolation between ppr[i] and ppr[i+1]
            log_n0, log_n1 = np.log(ppr[i]), np.log(ppr[i + 1])
            t = (1.0 - s0) / (s1 - s0)
            return np.exp(log_n0 + t * (log_n1 - log_n0))
    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)

    inpath = Path(sys.argv[1])
    records = parse_output(inpath)
    if not records:
        print(f"ERROR: no data blocks found in {inpath}", file=sys.stderr)
        sys.exit(1)

    num_ranks = records[0]["numRanks"]
    theta = records[0]["theta"]

    ppr = np.array([r["ppr"] for r in records], dtype=np.float64)
    bh_mean = np.array([r["BH_p50_mean"] for r in records])
    bh_min = np.array([r["BH_p50_min"] for r in records])
    bh_max = np.array([r["BH_p50_max"] for r in records])
    fmm_mean = np.array([r["FMM_p50_mean"] for r in records])
    fmm_min = np.array([r["FMM_p50_min"] for r in records])
    fmm_max = np.array([r["FMM_p50_max"] for r in records])
    speedup_mean = np.array([r["speedup_mean"] for r in records])
    speedup_min = np.array([r.get("speedup_min", np.nan) for r in records])
    speedup_max = np.array([r.get("speedup_max", np.nan) for r in records])

    # --- Fits ---
    bh_c_lin, bh_r2_lin, bh_c_nlogn, bh_r2_nlogn, bh_c_pow, bh_alpha, bh_r2_pow = fit_models(ppr, bh_mean)
    fmm_c_lin, fmm_r2_lin, fmm_c_nlogn, fmm_r2_nlogn, fmm_c_pow, fmm_alpha, fmm_r2_pow = fit_models(ppr, fmm_mean)

    # Extended x range for fitted curves (0.5x min to 2x max)
    x_ext = np.geomspace(ppr[0] * 0.5, ppr[-1] * 2, 200)

    # Crossover: log-interpolate actual speedup data where it crosses 1.0×
    crossover_n = data_crossover(ppr, speedup_mean)
    crossover_y = None
    if crossover_n is not None:
        crossover_y = fmm_c_pow * crossover_n ** fmm_alpha

    # --- Print summary ---
    print(f"numRanks={num_ranks}, theta={theta}")
    print(f"BH  power-law fit:  α = {bh_alpha:.3f},  c = {bh_c_pow:.4e}  (R² = {bh_r2_pow:.6f})")
    print(f"FMM power-law fit:  α = {fmm_alpha:.3f},  c = {fmm_c_pow:.4e}  (R² = {fmm_r2_pow:.6f})")
    if crossover_n is not None:
        print(f"Crossover (data):   ~{human_count(crossover_n)} particles/rank")
    print(f"  Reference R² — BH:  O(N) {bh_r2_lin:.6f}, O(N log N) {bh_r2_nlogn:.6f}")
    print(f"  Reference R² — FMM: O(N) {fmm_r2_lin:.6f}, O(N log N) {fmm_r2_nlogn:.6f}")

    # --- Plot ---
    fig, (ax1, ax_mid, ax2) = plt.subplots(
        3, 1, figsize=(8, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 1.2], "hspace": 0.08},
    )

    # Top panel: wall time (log-log) — plot in seconds
    ms2s = 1e-3
    ax1.fill_between(ppr, bh_min * ms2s, bh_max * ms2s, alpha=0.15, color="C0")
    ax1.fill_between(ppr, fmm_min * ms2s, fmm_max * ms2s, alpha=0.15, color="C1")
    ax1.plot(ppr, bh_mean * ms2s, "o-", color="C0", label="Barnes-Hut (mean p50)", zorder=5)
    ax1.plot(ppr, fmm_mean * ms2s, "s-", color="C1", label="FMM (mean p50)", zorder=5)
    # Power-law fitted curves
    ax1.plot(x_ext, bh_c_pow * x_ext ** bh_alpha * ms2s, "--", color="C0", alpha=0.6,
             label=rf"BH fit: $N^{{{bh_alpha:.2f}}}$")
    ax1.plot(x_ext, fmm_c_pow * x_ext ** fmm_alpha * ms2s, "--", color="C1", alpha=0.6,
             label=rf"FMM fit: $N^{{{fmm_alpha:.2f}}}$")
    if crossover_n is not None:
        crossover_y_s = crossover_y * ms2s
        ax1.axvline(crossover_n, color="gray", ls=":", alpha=0.5)
        ax1.annotate(f"crossover\n~{human_count(crossover_n)}/rank",
                     xy=(crossover_n, crossover_y_s), fontsize=8, color="gray",
                     ha="right", va="bottom", xytext=(-8, 4), textcoords="offset points")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_ylabel("Wall time [s]")
    ax1.set_title(f"Distributed FMM vs Barnes-Hut — {num_ranks} GH200 GPUs, θ={theta}")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, which="both", ls=":", alpha=0.3)

    # Middle panel: normalized time t/N vs N (log-x, linear-y) — plot in nanoseconds
    ms2ns = 1e6
    ax_mid.plot(ppr, bh_mean / ppr * ms2ns, "o-", color="C0", label="BH  t/N", zorder=5)
    ax_mid.plot(ppr, fmm_mean / ppr * ms2ns, "s-", color="C1", label="FMM t/N", zorder=5)
    ax_mid.fill_between(ppr, bh_min / ppr * ms2ns, bh_max / ppr * ms2ns, alpha=0.15, color="C0")
    ax_mid.fill_between(ppr, fmm_min / ppr * ms2ns, fmm_max / ppr * ms2ns, alpha=0.15, color="C1")
    if crossover_n is not None:
        ax_mid.axvline(crossover_n, color="gray", ls=":", alpha=0.5)
    ax_mid.set_xscale("log")
    ax_mid.set_ylabel("t / N  [ns/particle]")
    ax_mid.legend(fontsize=8, loc="upper left")
    ax_mid.grid(True, which="both", ls=":", alpha=0.3)

    # Bottom panel: speedup
    ax2.plot(ppr, speedup_mean, "D-", color="C2", zorder=5, label="Speedup (BH/FMM)")
    ax2.fill_between(ppr, speedup_min, speedup_max, alpha=0.15, color="C2")
    ax2.axhline(1.0, color="gray", ls="--", alpha=0.6)
    if crossover_n is not None:
        ax2.axvline(crossover_n, color="gray", ls=":", alpha=0.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("Particles per rank")
    ax2.set_ylabel("Speedup (BH / FMM)")
    ax2.grid(True, which="both", ls=":", alpha=0.3)
    ax2.legend(fontsize=8, loc="upper left")

    # Human-readable x ticks (disable x-axis minor ticks only, keep y-axis ones)
    for ax in (ax1, ax_mid, ax2):
        ax.set_xticks(ppr)
        ax.xaxis.set_minor_locator(plt.NullLocator())
    ax2.set_xticklabels([human_count(int(n)) for n in ppr], fontsize=8)

    outpath = inpath.with_suffix(".pdf")
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Saved: {outpath}")
    plt.close(fig)


if __name__ == "__main__":
    main()
