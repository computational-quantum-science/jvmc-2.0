import h5py
import matplotlib.pyplot as plt
import numpy as np

MODE = "batch_size"  # "batch_size" or "mc_chains"

BATCH_SIZE_SCAN_FILE = "data/batch_size_scan_RWKV_Ns=32768_mc_chains=100_10097+memory.hdf5"
MC_CHAIN_SCAN_FILE = "data/mc_chain_scan_batch_size=1000.hdf5" # doesn't have memory information

CATEGORIES = [
    ("peak_bytes_in_use", "Peak Memory Usage"),
]

def load_scan(filename):
    with h5py.File(filename, "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("=")[1]))
        x = [int(k.split("=")[1]) for k in keys]
        data = {category: [] for category, _ in CATEGORIES}
        for key in keys:
            for category, _ in CATEGORIES: # /bc=10000/memory/device_0/peak_bytes_in_use
                memory = f[key]["memory"]["device_0"][category][()]
                data[category].append(memory)
    return x, data


def plot_memory(ax, x, data, categories, xlabel, cutoff=None):
    for category, label in categories:
        memory = np.array(data[category]) / 1e9  # Convert bytes to gigabytes
        # baseline = memory[0]
        # speedup = [baseline / t for t in memory]
        ax.plot(x, memory, marker="x", label=label) if cutoff is None else ax.plot(x[:cutoff], memory[:cutoff], marker="x", label="peak memory usage")
    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Memory Usage (Gb)")
    ax.grid(True, which="both", alpha=0.3)


def main():
    if MODE == "batch_size":
        x_bc, data_bc = load_scan(BATCH_SIZE_SCAN_FILE)
        fig_bc, ax_bc = plt.subplots(figsize=(5, 4))
        plot_memory(ax_bc, x_bc, data_bc, [CATEGORIES[0]], "Batch Size", cutoff=None)
        ax_bc.legend(loc="upper left")
        fig_bc.tight_layout()
        bc_output_path = "batch_size_scan_memory.png"
        fig_bc.savefig(bc_output_path, dpi=200)
        print(f"Saved figure to {bc_output_path}")

    if MODE == "mc_chains":
        x_mc, data_mc = load_scan(MC_CHAIN_SCAN_FILE)
        
        fig_mc, ax_mc = plt.subplots(figsize=(5, 4))
        plot_memory(ax_mc, x_mc, data_mc, [CATEGORIES[2]], "# MC-chains")
        ax_mc.legend()
        fig_mc.tight_layout()
        mc_output_path = "benchmarks/mc_chains.png"
        fig_mc.savefig(mc_output_path, dpi=200)
        print(f"Saved figure to {mc_output_path}")


if __name__ == "__main__":
    main()
