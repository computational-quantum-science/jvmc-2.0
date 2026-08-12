import pandas as pd
import matplotlib.pyplot as plt


name = ["force", "qgt"]

plt.figure(figsize=(15, 6))

for n in name:
    df = pd.read_csv(f"jacobian_test_{n}_1gpu.csv")

    df_dense = df[df["jacobian"] == "dense"].sort_values("n_batches")
    df_batched = df[df["jacobian"] == "batched"].sort_values("n_batches")

    dense_time = df_dense["time"].iloc[0]
    dense_peak_gb = df_dense["peak_bytes_in_use"].iloc[0] / 1e9

    time_ratio = df_batched["time"] / dense_time
    memory_ratio = dense_peak_gb / (df_batched["peak_bytes_in_use"] / 1e9)

    plt.subplot(1, 2, 1)
    plt.plot(df_batched["n_batches"], time_ratio, "o-", label=n)
    plt.axhline(1, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("n_batches")
    plt.ylabel("time_batched / time_dense")
    # plt.yscale("log")

    plt.subplot(1, 2, 2)
    plt.plot(df_batched["n_batches"], memory_ratio, "o-", label=n)
    plt.axhline(1, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("n_batches")
    plt.ylabel("memory_dense / memory_batched")
    # plt.yscale("log")

plt.subplot(1, 2, 1)
plt.legend()
plt.subplot(1, 2, 2)
plt.legend()

plt.savefig("jacobian_memory_sweep.png")
plt.show()
