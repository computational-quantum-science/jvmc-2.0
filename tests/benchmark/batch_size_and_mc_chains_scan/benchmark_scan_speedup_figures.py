import h5py
import matplotlib.pyplot as plt

MODE = "mc_chains"  # "batch_size" or "mc_chains"

BATCH_SIZE_SCAN_FILE = "data/batch_size_scan_RWKV_Ns=32768_mc_chains=100_10097+memory.hdf5"
MC_CHAIN_SCAN_FILE = "data/mc_chain_scan_batch_size=1000.hdf5"

CATEGORIES = [
    ("compute objective function and gradient", "Compute Objective Function and Gradient"),
    ("solve", "Solve"),
    ("sampling", "Sample Generation"),
]


def load_scan(filename):
    with h5py.File(filename, "r") as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split("=")[1]))
        x = [int(k.split("=")[1]) for k in keys]
        data = {category: [] for category, _ in CATEGORIES}
        for key in keys:
            for category, _ in CATEGORIES:
                last_timing = f[key]["timings"][category]["last_timing"][()]
                data[category].append(last_timing)
    return x, data


def plot_speedup(ax, x, data, categories, xlabel, cutoff=None):
    for category, label in categories:
        if label == "Compute Objective Function and Gradient":
            label = "Objective Function and Gradient"
        timings = data[category]
        baseline = timings[0]
        speedup = [baseline / t for t in timings]
        ax.plot(x, speedup, marker="x", label=label) if cutoff is None else ax.plot(x[:cutoff], speedup[:cutoff], marker="x", label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Speedup")
    ax.grid(True, which="both", alpha=0.3)


def main():
    if MODE == "batch_size":
        x_bc, data_bc = load_scan(BATCH_SIZE_SCAN_FILE)

        fig_bc, ax_bc = plt.subplots(figsize=(5, 4))
        plot_speedup(ax_bc, x_bc, data_bc, [CATEGORIES[0]], "Batch Size", cutoff=None)
        plot_speedup(ax_bc, x_bc, data_bc, [CATEGORIES[1]], "Batch Size", cutoff=None)
        plot_speedup(ax_bc, x_bc, data_bc, [CATEGORIES[2]], "Batch Size", cutoff=None)
        ax_bc.legend(loc="upper left")
        fig_bc.tight_layout()
        bc_output_path = "batch_size_scan.png"
        fig_bc.savefig(bc_output_path, dpi=200)
        print(f"Saved figure to {bc_output_path}")

    if MODE == "mc_chains":
        x_mc, data_mc = load_scan(MC_CHAIN_SCAN_FILE)

        fig_mc, ax_mc = plt.subplots(figsize=(5, 4))
        plot_speedup(ax_mc, x_mc, data_mc, [CATEGORIES[2]], "# MC-chains")
        ax_mc.legend()
        fig_mc.tight_layout()
        mc_output_path = "mc_chains.png"
        fig_mc.savefig(mc_output_path, dpi=200)
        print(f"Saved figure to {mc_output_path}")


if __name__ == "__main__":
    main()
