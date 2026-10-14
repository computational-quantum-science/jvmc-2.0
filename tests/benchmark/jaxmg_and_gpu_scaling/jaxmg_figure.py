import glob
import math
import re

import h5py
import matplotlib.pyplot as plt

JAXMG_FILE_PATTERN = "data/jaxmg/*.hdf5"
JAXMG_FILE_RE = re.compile(r"jaxmg_(on|off)_(\d+)(?:_(\d+))?\.hdf5$")

CATEGORIES = [
    "compute objective function and gradient",
    "solve",
    "sampling",
]

TIMING_KEY = "last_timing"
# TIMING_KEY = "first_timing"

USE_RERUNS = True


def read_total_time(path):
    with h5py.File(path, "r") as f:
        if "timings" not in f:
            return math.nan
        return sum(f["timings"][c][TIMING_KEY][()] for c in CATEGORIES)


def load_jaxmg_scan():
    # runs[mode][n_params][rerun index] = path
    runs = {"on": {}, "off": {}}
    for path in glob.glob(JAXMG_FILE_PATTERN):
        match = JAXMG_FILE_RE.search(path)
        mode, n_params, rerun = match.group(1), int(match.group(2)), int(match.group(3) or 1)
        runs[mode].setdefault(n_params, {})[rerun] = path

    n_params_list = sorted({n for mode_runs in runs.values() for n in mode_runs})
    times = {}
    for mode, mode_runs in runs.items():
        times[mode] = []
        for n_params in n_params_list:
            files = mode_runs.get(n_params, {})
            path = files.get(2) if USE_RERUNS and 2 in files else files.get(1)
            time = read_total_time(path) if path else math.nan
            print(f"jaxmg {mode}, {n_params}: {time:.2f} s ({path})")
            times[mode].append(time)
    return n_params_list, times


def main():
    n_params_list, times = load_jaxmg_scan()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(n_params_list, times["on"], marker="o", color="tab:blue", label="jaxmg on")
    ax.plot(n_params_list, times["off"], marker="x", color="tab:red", label="jaxmg off")
    ax.set_xlabel("No. of parameters")
    ax.set_ylabel(f"Total execution time, {TIMING_KEY.replace('_', ' ')} (s)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    rerun_tag = "_reruns" if USE_RERUNS else ""
    output_path = f"jaxmg_total_time_{TIMING_KEY}{rerun_tag}.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
