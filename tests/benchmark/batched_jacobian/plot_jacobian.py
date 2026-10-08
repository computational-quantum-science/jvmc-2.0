import glob
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Folder written by jacobian_sweep.sh, the latest one by default
out_dir = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("results_*"))[-1]

fig, (ax_mem, ax_time) = plt.subplots(1, 2, figsize=(14, 5.5))

for i, quantity in enumerate(["force", "qgt"]):
    csv_name = os.path.join(out_dir, f"jacobian_{quantity}.csv")
    if not os.path.exists(csv_name):
        continue
    df = pd.read_csv(csv_name).drop_duplicates(subset=["jacobian", "n_batches"], keep="last")
    df_dense = df[df["jacobian"] == "dense"]
    df_batched = df[df["jacobian"] == "batched"].sort_values("n_batches")
    n = df_batched["n_batches"]
    color = f"C{i}"

    memory = df_batched["phase_peak_bytes"] / 1e9
    ax_mem.plot(n, memory, "o-", color=color, label=f"{quantity}, batched")
    ax_mem.plot(n, memory.iloc[0] / n, ":", color=color, label=f"{quantity}, ideal 1/n")
    if len(df_dense):
        ax_mem.axhline(df_dense["phase_peak_bytes"].iloc[0] / 1e9, color=color, ls="--", lw=1,
                       label=f"{quantity}, dense")
    if quantity == "qgt":
        # Size of the output itself: the memory can't go below this
        ax_mem.axhline(df_batched["out_bytes_per_device"].iloc[0] / 1e9, color=color, ls="-.", lw=1,
                       label="S shard per device")

    ax_time.plot(n, df_batched["time"], "o-", color=color, label=f"{quantity}, batched")
    if len(df_dense):
        ax_time.axhline(df_dense["time"].iloc[0], color=color, ls="--", lw=1, label=f"{quantity}, dense")

info = df.iloc[0]
fig.suptitle(
    f"{info['n_devices']} devices, {info['n_samples']} samples, {info['n_params']} parameters"
    f"{' (holomorphic)' if info['holomorphic'] else ''}"
)
for ax in (ax_mem, ax_time):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("n_batches")
    ax.legend(fontsize=8)
ax_mem.set_ylabel("peak memory of the computation per device [GB]")
ax_time.set_ylabel("time [s]")

fig.tight_layout()
fig.savefig(os.path.join(out_dir, "jacobian_memory_sweep.png"))
