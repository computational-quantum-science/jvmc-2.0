import math

import h5py
import matplotlib.pyplot as plt

DIR = "data/jaxmg2"

CURVE1 = ["net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_jaxmg_off_hs=14.hdf5",
          "net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_jaxmg_off_hs=20.hdf5",
          "net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_jaxmg_off_hs=28.hdf5",
          "net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_jaxmg_off_hs=40.hdf5",
          "net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_jaxmg_off_hs=56.hdf5",
          "net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_jaxmg_off_hs=80.hdf5",
          "net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_jaxmg_off_hs=90.hdf5"]

CURVE2 = ["net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_TA=256_hs=20.hdf5",
          "net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_TA=256_hs=28.hdf5",
          "net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_TA=256_hs=40.hdf5",
          "net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_TA=256_hs=56.hdf5",
          "net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_TA=256_hs=80.hdf5",
          "net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_TA=256_hs=90.hdf5",
          "net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_TA=256_hs=100.hdf5",
          "net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_TA=256_hs=110.hdf5",
          "net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_TA=256_hs=126.hdf5"]

CURVE3 = ["net=RNN_bs=1024_chains=200_Ns=4096_gpus=32_TA=256_hs=134.hdf5",
          "net=RNN_bs=1024_chains=200_Ns=4096_gpus=32_TA=256_hs=142.hdf5"]

CURVE4 = ["net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_hs=20.hdf5",
          "net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_hs=28.hdf5",
          "net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_hs=40.hdf5",
          "net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_hs=56.hdf5",
          "net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_hs=80.hdf5",
          "net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_hs=90.hdf5",
          "net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_hs=110.hdf5",
          "net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_hs=115.hdf5",
          "net=RNN_bs=1024_chains=200_Ns=4096_gpus=16_hs=120.hdf5"]



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


def load_jaxmg_scan(files):
    # runs[mode][n_params][rerun index] = path

    n_params_list = []
    times = []
    for path in files:
        path = f"{DIR}/{path}"
        times.append(read_total_time(path))
        with h5py.File(path, "r") as f:
            n_params_list.append(f["metadata"]["num_params"][()])
    return n_params_list, times


def main():
    n_params_list1, times1 = load_jaxmg_scan(CURVE1)
    n_params_list2, times2 = load_jaxmg_scan(CURVE2)
    n_params_list3, times3 = load_jaxmg_scan(CURVE3)
    n_params_list4, times4 = load_jaxmg_scan(CURVE4)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(n_params_list1, times1, marker="o", color="tab:blue", label="jaxmg off")
    ax.plot(n_params_list2, times2, marker="s", color="tab:red", label="jaxmg on T_A=256, 16 GPUs")
    ax.plot(n_params_list3, times3, marker="^", color="tab:green", label="jaxmg on T_A=256, 32 GPUs")
    # ax.plot(n_params_list4, times4, marker="x", color="tab:purple", label="jaxmg on T_A=max, 16 GPUs")
    ax.set_xlabel("No. of parameters")
    ax.set_ylabel(f"Total execution time (s)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = f"jaxmg_total_time.png"
    fig.savefig(output_path, dpi=200)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
