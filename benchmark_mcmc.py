"""
MAc:
python benchmark_mcmc.py \
  --csv ./data/BindSchalder_Macayeal_IceStreams.csv \
  --sgs_bed ./sgs_beds/sgs_0_bindshadler_macayeal.txt \
  --data_weight ./data/data_weight.txt \
  --n_iter 50000 \
  --skip_cpu \
W:
python benchmark_mcmc.py `
   --csv ./data/BindSchalder_Macayeal_IceStreams.csv `
   --sgs_bed ./sgs_beds/sgs_0_bindshadler_macayeal.txt `
   --data_weight ./data/data_weight.txt `
   --n_iter 50000 --skip_cpu
"""

import argparse
import json
import time
import sys
import os

import numpy as np
import pandas as pd
sys.path.append('..')
#from gstatsMCMC.gpu import MCMC_gpu, Topography_gpu
from gstatsMCMC import MCMC_test


from gstatsMCMC import MCMC, Topography

import torch

def build_parser():
    p = argparse.ArgumentParser(
        description="Benchmark MCMC (NumPy) vs MCMC_gpu (PyTorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required paths ────────────────────────────────────────────────────────
    p.add_argument("--csv",         required=True,  help="Path to glacier CSV")
    p.add_argument("--sgs_bed",     required=True,  help="Path to initial SGS bed .txt")
    p.add_argument("--data_weight", required=True,  help="Path to CRF data-weight .txt")

    # ── Chain settings ────────────────────────────────────────────────────────
    p.add_argument("--n_iter",      type=int,   default=10000)
    p.add_argument("--rng_seed",    type=int,   default=23198104)
    p.add_argument("--sigma_mc",    type=float, default=DEFAULT_SIGMA)
    p.add_argument("--resolution",  type=float, default=500.0)

    # ── RandField ─────────────────────────────────────────────────────────────
    p.add_argument("--range_min_x", type=float, default=10e3)
    p.add_argument("--range_max_x", type=float, default=50e3)
    p.add_argument("--range_min_y", type=float, default=10e3)
    p.add_argument("--range_max_y", type=float, default=50e3)
    p.add_argument("--scale_min",   type=float, default=50.0)
    p.add_argument("--scale_max",   type=float, default=150.0)
    p.add_argument("--nugget_max",  type=float, default=0.0)
    p.add_argument("--rf_model",    type=str,   default="Matern",
                   choices=["Gaussian", "Exponential", "Matern"])
    p.add_argument("--smoothness",  type=float, default=1.0,
                   help="Matern smoothness (nu). Required when --rf_model=Matern")

    # ── Block sizes ───────────────────────────────────────────────────────────
    p.add_argument("--min_block_x", type=int, default=50)
    p.add_argument("--max_block_x", type=int, default=80)
    p.add_argument("--min_block_y", type=int, default=50)
    p.add_argument("--max_block_y", type=int, default=80)

    # ── Logistic weight params ────────────────────────────────────────────────
    p.add_argument("--logis_L",      type=float, default=2.0)
    p.add_argument("--logis_x0",     type=float, default=0.0)
    p.add_argument("--logis_k",      type=float, default=6.0)
    p.add_argument("--logis_offset", type=float, default=1.0)
    p.add_argument("--max_dist",     type=float, default=50e3)

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--output",    type=str, default="benchmark_results.json")
    p.add_argument("--skip_cpu",  action="store_true", help="Skip the NumPy/CPU run")
    p.add_argument("--skip_gpu",  action="store_true", help="Skip the PyTorch/GPU run")

    return p


try:
    import config
    DEFAULT_SIGMA = config.sigma3
except ImportError:
    DEFAULT_SIGMA = 5.0


def load_data(csv_path: str, resolution: float):
    """Load glacier CSV and return all required numpy arrays."""
    df = pd.read_csv(csv_path)

    x_uniq = np.unique(df.x)
    y_uniq = np.unique(df.y)
    xx, yy = np.meshgrid(x_uniq, y_uniq)

    def reshape(col):
        return df[col].values.reshape(xx.shape)

    dhdt              = reshape("dhdt")
    smb               = reshape("smb")
    velx              = reshape("velx")
    vely              = reshape("vely")
    bedmap_mask       = reshape("bedmap_mask")
    bedmachine_thickness = reshape("bedmachine_thickness")
    bedmap_surf       = reshape("bedmap_surf")
    highvel_mask      = reshape("highvel_mask")
    bedmap_bed        = reshape("bedmap_bed")

    bedmachine_bed = bedmap_surf - bedmachine_thickness

    cond_bed = np.where(
        bedmap_mask == 1,
        df["bed"].values.reshape(xx.shape),
        bedmap_bed,
    )
    data_mask         = ~np.isnan(cond_bed)
    grounded_ice_mask = bedmap_mask == 1

    return dict(
        xx=xx, yy=yy,
        dhdt=dhdt, smb=smb, velx=velx, vely=vely,
        bedmap_surf=bedmap_surf, bedmap_bed=bedmap_bed,
        bedmachine_bed=bedmachine_bed,
        bedmap_mask=bedmap_mask, highvel_mask=highvel_mask,
        cond_bed=cond_bed, data_mask=data_mask,
        grounded_ice_mask=grounded_ice_mask,
    )

def mc_summary(loss_mc_array: np.ndarray):
    """Return a small dict summarising the mass-conservation loss series"""
    a = np.asarray(loss_mc_array, dtype=float)
    return {
        "final":  float(a[-1]),
        "min":    float(np.nanmin(a)),
        "max":    float(np.nanmax(a)),
        "mean":   float(np.nanmean(a)),
    }


def run_cpu(args, data, initial_bed, highvel_mask):
    print("\n[CPU] Setting up chain_crf …")
    chain = MCMC.chain_crf(
        data["xx"], data["yy"],
        initial_bed,
        data["bedmap_surf"], data["velx"], data["vely"],
        data["dhdt"], data["smb"],
        data["cond_bed"], data["data_mask"],
        data["grounded_ice_mask"],
        args.resolution,
    )

    chain.set_update_region(True, highvel_mask)
    chain.set_loss_type(sigma_mc=args.sigma_mc, massConvInRegion=True)
    chain.set_update_type("CRF_weight")
    chain.set_random_generator(rng_seed=args.rng_seed)

    # RandField - Numpy
    rf = MCMC.RandField(
        args.range_min_x, args.range_max_x,
        args.range_min_y, args.range_max_y,
        args.scale_min, args.scale_max,
        args.nugget_max, args.rf_model,
        isotropic=True,
        smoothness=args.smoothness,
    )
    rf.set_block_sizes(
        args.min_block_x, args.max_block_x,
        args.min_block_y, args.max_block_y,
    )
    rf.set_weight_param(
        args.logis_L, args.logis_x0, args.logis_k, args.logis_offset,
        args.max_dist, args.resolution,
    )
    rf.set_generation_method(spectral=True)

    crf_weight = np.loadtxt(args.data_weight)
    chain.crf_data_weight = crf_weight

    print(f"[CPU] Running {args.n_iter} iterations …")
    t0 = time.perf_counter()
    result = chain.run(
        n_iter=args.n_iter, RF=rf,
        only_save_last_bed=True,
        plot=False, progress_bar=True,
    )
    wall_time = time.perf_counter() - t0

    last_bed, loss_mc, loss_data, loss_total, steps, resampled, blocks = result[:7]

    # Compute final mass-conservation residual on the returned bed
    mc_res_final = Topography.get_mass_conservation_residual(
        last_bed,
        data["bedmap_surf"], data["velx"], data["vely"],
        data["dhdt"], data["smb"], args.resolution,
    )
    mc_res_in_region = mc_res_final[highvel_mask == 1]
    mc_res_in_region = mc_res_in_region[~np.isnan(mc_res_in_region)]

    acceptance_rate = float(np.sum(steps)) / args.n_iter

    return {
        "backend":          "cpu_numpy",
        "n_iter":           args.n_iter,
        "wall_time_s":      round(wall_time, 4),
        "iter_per_sec":     round(args.n_iter / wall_time, 4),
        "acceptance_rate":  round(acceptance_rate, 6),
        # Per-iteration arrays (lists so JSON-serialisable)
        "loss_total":       loss_total.tolist(),
        "loss_mc":          loss_mc.tolist(),
        "loss_data":        loss_data.tolist(),
        "steps_accepted":   steps.astype(int).tolist(),
        # Summary stats
        "loss_summary": {
            "final": float(loss_total[-1]),
            "min":   float(np.nanmin(loss_total)),
            "max":   float(np.nanmax(loss_total)),
            "mean":  float(np.nanmean(loss_total)),
        },
        "mass_conservation_summary": mc_summary(loss_mc),
        # Final residual distribution in the high-vel region
        "final_mc_residual_in_region": {
            "mean":   float(np.nanmean(mc_res_in_region)),
            "std":    float(np.nanstd(mc_res_in_region)),
            "absmax": float(np.nanmax(np.abs(mc_res_in_region))),
        },
    }

def run_gpu(args, data, initial_bed, highvel_mask):

    if torch.cuda.is_available():
        dev_name = "cuda"
    elif torch.backends.mps.is_available():
        dev_name = "mps"
    else:
        dev_name = "cpu"
    print(f"\n[GPU] PyTorch device: {dev_name}")

    print("[GPU] Setting up chain_crf …")
    chain = MCMC_test.chain_crf_gpu(
        data["xx"], data["yy"],
        initial_bed,
        data["bedmap_surf"], data["velx"], data["vely"],
        data["dhdt"], data["smb"],
        data["cond_bed"], data["data_mask"],
        data["grounded_ice_mask"],
        args.resolution,
    )

    chain.set_update_region(True, highvel_mask)
    chain.set_loss_type(sigma_mc=args.sigma_mc, massConvInRegion=True)
    chain.set_update_type("CRF_weight")
    chain.set_random_generator(rng_seed=args.rng_seed)

    # RandField
    rf = MCMC_test.RandField(
        args.range_min_x, args.range_max_x,
        args.range_min_y, args.range_max_y,
        args.scale_min, args.scale_max,
        args.nugget_max, args.rf_model,
        isotropic=True,
        smoothness=args.smoothness,
    )
    rf.set_block_sizes(
        args.min_block_x, args.max_block_x,
        args.min_block_y, args.max_block_y,
    )
    rf.set_weight_param(
        args.logis_L, args.logis_x0, args.logis_k, args.logis_offset,
        args.max_dist, args.resolution,
    )
    rf.set_generation_method(spectral=True)

    crf_weight = np.loadtxt(args.data_weight)
    chain.crf_data_weight = crf_weight

    print(f"[GPU] Running {args.n_iter} iterations …")
    t0 = time.perf_counter()
    result = chain.run(
        n_iter=args.n_iter, RF=rf,
        only_save_last_bed=True,
        plot=False, progress_bar=True,
    )
    # Synchronise device before stopping the clock
    if dev_name == "cuda":
        torch.cuda.synchronize()
    elif dev_name == "mps":
        torch.mps.synchronize()
    wall_time = time.perf_counter() - t0

    last_bed, loss_mc, loss_data, loss_total, steps, resampled, blocks = result[:7]

    # Compute final mass-conservation residual on the returned bed
    mc_res_final = Topography.get_mass_conservation_residual(
        last_bed,
        data["bedmap_surf"], data["velx"], data["vely"],
        data["dhdt"], data["smb"], args.resolution,
    )
    mc_res_in_region = mc_res_final[highvel_mask == 1]
    mc_res_in_region = mc_res_in_region[~np.isnan(mc_res_in_region)]

    acceptance_rate = float(np.sum(steps)) / args.n_iter

    return {
        "backend":          f"gpu_pytorch_{dev_name}",
        "n_iter":           args.n_iter,
        "wall_time_s":      round(wall_time, 4),
        "iter_per_sec":     round(args.n_iter / wall_time, 4),
        "acceptance_rate":  round(acceptance_rate, 6),
        # Per-iteration arrays
        "loss_total":       loss_total.tolist(),
        "loss_mc":          loss_mc.tolist(),
        "loss_data":        loss_data.tolist(),
        "steps_accepted":   steps.astype(int).tolist(),
        # Summary stats
        "loss_summary": {
            "final": float(loss_total[-1]),
            "min":   float(np.nanmin(loss_total)),
            "max":   float(np.nanmax(loss_total)),
            "mean":  float(np.nanmean(loss_total)),
        },
        "mass_conservation_summary": mc_summary(loss_mc),
        # Final residual distribution in the high-vel region
        "final_mc_residual_in_region": {
            "mean":   float(np.nanmean(mc_res_in_region)),
            "std":    float(np.nanstd(mc_res_in_region)),
            "absmax": float(np.nanmax(np.abs(mc_res_in_region))),
        },
    }


def compare(cpu_res: dict, gpu_res: dict) -> dict:
    """Return a concise side-by-side comparison dict."""
    speedup = gpu_res["iter_per_sec"] / cpu_res["iter_per_sec"]

    cpu_loss  = np.asarray(cpu_res["loss_total"])
    gpu_loss  = np.asarray(gpu_res["loss_total"])
    loss_diff = np.abs(cpu_loss - gpu_loss)

    return {
        "speedup_gpu_over_cpu": round(speedup, 4),
        "iter_per_sec": {
            "cpu": cpu_res["iter_per_sec"],
            "gpu": gpu_res["iter_per_sec"],
        },
        "wall_time_s": {
            "cpu": cpu_res["wall_time_s"],
            "gpu": gpu_res["wall_time_s"],
        },
        "acceptance_rate": {
            "cpu": cpu_res["acceptance_rate"],
            "gpu": gpu_res["acceptance_rate"],
        },
        "final_loss": {
            "cpu": cpu_res["loss_summary"]["final"],
            "gpu": gpu_res["loss_summary"]["final"],
        },
        "final_loss_mc": {
            "cpu": cpu_res["mass_conservation_summary"]["final"],
            "gpu": gpu_res["mass_conservation_summary"]["final"],
        },
        "loss_difference_per_iter": {
            "mean":  float(np.nanmean(loss_diff)),
            "max":   float(np.nanmax(loss_diff)),
        },
        "final_mc_residual_std_in_region": {
            "cpu": cpu_res["final_mc_residual_in_region"]["std"],
            "gpu": gpu_res["final_mc_residual_in_region"]["std"],
        },
    }

def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.skip_cpu and args.skip_gpu:
        parser.error("Cannot use both --skip_cpu and --skip_gpu at the same time.")

    # ── Load shared data ───────────────────────────────────────────────────────
    print("Loading data …")
    data = load_data(args.csv, args.resolution)

    initial_bed = np.loadtxt(args.sgs_bed)
    highvel_mask = data["highvel_mask"]

    # Guard: replace negative thicknesses in initial bed
    thickness = data["bedmap_surf"] - initial_bed
    initial_bed = np.where(
        (thickness <= 0) & (data["bedmap_mask"] == 1),
        data["bedmap_surf"] - 1,
        initial_bed,
    )

    # ── Run experiments ────────────────────────────────────────────────────────
    cpu_result = None
    gpu_result = None

    if not args.skip_cpu:
        cpu_result = run_cpu(args, data, initial_bed.copy(), highvel_mask)
        print(
            f"\n[CPU] Done  —  {cpu_result['iter_per_sec']:.2f} it/s  "
            f"|  final loss: {cpu_result['loss_summary']['final']:.4e}  "
            f"|  acceptance: {cpu_result['acceptance_rate']:.4f}"
        )

    if not args.skip_gpu:
        gpu_result = run_gpu(args, data, initial_bed.copy(), highvel_mask)
        print(
            f"\n[GPU] Done  —  {gpu_result['iter_per_sec']:.2f} it/s  "
            f"|  final loss: {gpu_result['loss_summary']['final']:.4e}  "
            f"|  acceptance: {gpu_result['acceptance_rate']:.4f}"
        )

    # ── Build output ───────────────────────────────────────────────────────────
    output = {
        "config": vars(args),
        "results": {},
    }

    if cpu_result:
        output["results"]["cpu"] = cpu_result
    if gpu_result:
        output["results"]["gpu"] = gpu_result
    if cpu_result and gpu_result:
        output["comparison"] = compare(cpu_result, gpu_result)
        c = output["comparison"]
        print(
            f"\n── Comparison ──────────────────────────────────────────\n"
            f"  Speedup (GPU / CPU)  : {c['speedup_gpu_over_cpu']:.2f}×\n"
            f"  Wall time  CPU / GPU : {c['wall_time_s']['cpu']:.1f}s / {c['wall_time_s']['gpu']:.1f}s\n"
            f"  Final loss CPU / GPU : {c['final_loss']['cpu']:.4e} / {c['final_loss']['gpu']:.4e}\n"
            f"  Mean |Δloss| per iter: {c['loss_difference_per_iter']['mean']:.4e}\n"
            f"  MC residual σ  C / G : {c['final_mc_residual_std_in_region']['cpu']:.4e} / "
            f"{c['final_mc_residual_std_in_region']['gpu']:.4e}\n"
            f"────────────────────────────────────────────────────────"
        )

    # ── Write JSON ─────────────────────────────────────────────────────────────
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()



