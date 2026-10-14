import glob
import re

import h5py
import matplotlib.pyplot as plt

# GPU_SCAN_FILE_PATTERN = "benchmarks/*Ns=30000_gpus=*.hdf5"
GPU_SCAN_FILE_PATTERN = "data/RNN_gpu_scaling/*.hdf5"
GPU_SCAN_FILE_RE = re.compile(r"gpus=(\d+)")

CATEGORIES = [
    "compute objective function and gradient",
    "solve",
    "sampling",
]

# CATEGORY_TO_PLOT = CATEGORIES[2]
CATEGORY_TO_PLOT = "total"


def load_gpu_scan(category):
    files = {}
    for path in glob.glob(GPU_SCAN_FILE_PATTERN):
        print(path)
        match = GPU_SCAN_FILE_RE.search(path)
        n_gpus = int(match.group(1))
        files[n_gpus] = path

    n_gpus_list = sorted(files.keys())
    times = []
    for n_gpus in n_gpus_list:
        with h5py.File(files[n_gpus], "r") as f:
            if category == "total":
                time = sum(f["timings"][c]["last_timing"][()] for c in CATEGORIES)
            else:
                time = f["timings"][category]["last_timing"][()]
        times.append(time)
    return n_gpus_list, times


def main():
    n_gpus_list, times = load_gpu_scan(CATEGORY_TO_PLOT)

    baseline_gpus = n_gpus_list[0]
    baseline_time = times[0]
    ideal = [n_gpus / baseline_gpus for n_gpus in n_gpus_list]
    measured = [baseline_time / t for t in times]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(n_gpus_list, ideal, "k--", label="ideal")
    ax.plot(n_gpus_list, measured, marker="x", color="red", label="measured")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("No. of GPUs")
    ax.set_ylabel("Speedup")
    ax.set_title(CATEGORY_TO_PLOT.capitalize())
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = f"gpu_scan_{CATEGORY_TO_PLOT.replace(' ', '_')}.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
