import numpy as np
import time

class OutputManager:
    """
    Builds an in-memory nested dictionary of timeseries data and timings.

    Use write_observables(), write_metadata(), write_network_checkpoint() to
    accumulate data, then call save_to_h5() whenever you want to persist it.

    Examples
    --------
        outp = OutputManager()
        outp.write_observables(t, energy=1.0, momentum=0.5)
        outp.save_to_h5("results.h5")
        outp.save_to_h5("results.h5", append=True)  # append to existing file
    """

    def __init__(self):
        self.data = {}
        self._timings = {}

    def write_observables(self, time: float, **kwargs):
        self._write("observables", time, **kwargs)

    def write_metadata(self, time: float, **kwargs):
        self._write("metadata", time, **kwargs)

    def write_network_checkpoint(self, time: float, weights):
        self._write("network_checkpoints", time, checkpoints=weights)

    def get_network_checkpoint(self, time: float = None, idx: int = None):
        if time is not None and idx is not None:
            raise ValueError("Specify either 'time' or 'idx', not both.")

        times = self.data["network_checkpoints"]["times"]
        checkpoints = self.data["network_checkpoints"]["checkpoints"]

        if not times:
            raise RuntimeError("No checkpoints have been recorded yet.")

        if time is not None:
            idx = int(np.argmin(np.abs(np.array(times) - time)))
        elif idx is None:
            idx = -1

        return times[idx], checkpoints[idx]
    
    def start_timing(self, name: str):
        if name not in self._timings:
            self._timings[name] = {"total": 0.0, "last_total": 0.0, "count": 0, "init": 0.0}
        self._timings[name]["init"] = time.perf_counter()

    def stop_timing(self, name: str):
        elapsed = time.perf_counter() - self._timings[name]["init"]
        self._timings[name]["total"] += elapsed
        self._timings[name]["count"] += 1

        return elapsed

    def add_timing(self, name: str, elapsed: float):
        if name not in self._timings:
            self._timings[name] = {"total": 0.0, "last_total": 0.0, "count": 0, "init": 0.0}
        self._timings[name]["total"] += elapsed
        self._timings[name]["count"] += 1

    def print_timings(self, indent: str = ""):
        print(f"{indent}Recorded timings:", flush=True)
        for key, item in self._timings.items():
            delta = item["total"] - item["last_total"]
            print(f"{indent}    • {key}: {delta:.6f}s", flush=True)
            item["last_total"] = item["total"]

    def save_to_h5(self, filename: str, append: bool = False):
        """
        Write the in-memory data dictionary to an HDF5 file.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file.
        append : bool
            If True, append to an existing file. If False (default), overwrite.
        """
        import h5py

        mode = "a" if append else "w"
        with h5py.File(filename, mode) as f:
            self._write_dict_to_h5(f, "/", self.data)

    @staticmethod
    def load_from_h5(filename: str) -> dict:
        import h5py

        def read_recursive(group):
            out = {}
            for key, item in group.items():
                if isinstance(item, h5py.Dataset):
                    out[key] = item[()]
                elif isinstance(item, h5py.Group):
                    out[key] = read_recursive(item)
            return out

        with h5py.File(filename, "r") as f:
            return read_recursive(f)

    def _write(self, subgroup: str, time_val: float, **kwargs):
        if subgroup not in self.data:
            self.data[subgroup] = {}
        if "times" not in self.data[subgroup]:
            self.data[subgroup]["times"] = []
        self.data[subgroup]["times"].append(time_val)
        self._store_recursive(self.data[subgroup], kwargs)

    def _store_recursive(self, store: dict, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                if key not in store:
                    store[key] = {}
                self._store_recursive(store[key], value)
            else:
                if key not in store:
                    store[key] = []
                store[key].append(value)

    def _write_dict_to_h5(self, f, path: str, d: dict):
        for key, value in d.items():
            full_path = f"{path}/{key}".replace("//", "/")
            if isinstance(value, dict):
                if full_path not in f:
                    f.create_group(full_path)
                self._write_dict_to_h5(f, full_path, value)
            else:
                arr = np.asarray(value, dtype="f8")
                if full_path in f:
                    del f[full_path]
                f.create_dataset(full_path, data=arr)