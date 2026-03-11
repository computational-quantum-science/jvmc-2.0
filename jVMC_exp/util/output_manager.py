import h5py
import numpy as np
import jax.numpy as jnp
import time

class OutputManager:
    '''
    This class provides functionality for I/O and timing.

    Upon initialization with
    
    :code:`outp = OutputManager("filename.h5")`
    
    an HDF5 file :code:`filename.h5` is created. If the file exists, `append=True` will prevent overwriting it.
    This HDF5 file is intended for numerical output and output can be written to it using the :code:`write_observables()`,
    :code:`write_metadata()`, and :code:`write_network_checkpoint()` member functions.

    Furthermore, timings can be recorded using the :code:`start_timing()` and :code:`stop_timing()` member functions. The recorded
    timings can be printed to screen using the :code:`print_timings()` function.
    '''
    def __init__(self, dataFileName, group="/", append=False):
        self._file_name = dataFileName
        self._current_group = "/"
        self._append = 'w' if append else 'a'

        self.set_group(group)
        self._timings = {}

    def set_group(self, group):
        if group != "/":
            self._current_group = "/" + group

        with h5py.File(self._file_name, self._append) as f:
            if not self._current_group in f:
                f.create_group(self._current_group)
        self._append = 'a'

    def _write_timeseries(self, group_name, time, **kwargs):
        full_group_name = self._current_group + "/" + group_name
        time_path = full_group_name + "/times" 

        with h5py.File(self._file_name, "a") as f:
            if group_name not in f[self._current_group]:
                f.create_group(full_group_name)

            if "times" not in f[full_group_name]:
                f.create_dataset(time_path, (0,), maxshape=(None,), dtype='f8', chunks=True)

            new_len = len(f[time_path]) + 1
            f[time_path].resize((new_len,))
            f[time_path][-1] = time
            self._write_timeseries_recursive(f, full_group_name, kwargs)

    def _write_timeseries_recursive(self, f, full_group_name, data_dict):
        for key, value in data_dict.items():
            new_full_group_name = full_group_name + "/" + key

            if isinstance(value, dict):
                if key not in f[full_group_name]:
                    f.create_group(new_full_group_name)
                self._write_timeseries_recursive(f, new_full_group_name, value)

            else:
                value = self.to_array(value)
                if key not in f[full_group_name]:
                    f.create_dataset(new_full_group_name, (0,) + value.shape, maxshape=(None,) + value.shape, dtype='f8', chunks=True)
                
                new_len = len(f[new_full_group_name]) + 1
                f[new_full_group_name].resize((new_len,) + value.shape)
                f[new_full_group_name][-1] = value

    def write_observables(self, time, **kwargs):
        self._write_timeseries("observables", time, **kwargs)

    def write_metadata(self, time, **kwargs):
        self._write_timeseries("metadata", time, **kwargs)
    
    def write_network_checkpoint(self, time, weights):
        self._write_timeseries("network_checkpoints", time, checkpoints=weights)

    def get_network_checkpoint(self, time=None, idx=None):
        if time is not None and idx is not None:
            raise ValueError("Cannot specify both 'time' and 'idx'. Choose one.")
        if time is None and idx is None:
            idx = -1

        full_group_name = self._current_group + "/" + "network_checkpoints"

        with h5py.File(self._file_name, "r") as f:
            times = np.array(f[full_group_name + "/times"])
            if time is not None:
                idx = np.argmin(np.abs(times - time))
            weights = f[full_group_name + "/" + "checkpoints"][idx]
        time = times[idx]

        return time, weights

    def write_error_data(self, name, data):
        group_name = "error_data"

        with h5py.File(self._file_name, "a") as f:
            if group_name not in f["/"]:
                f.create_group("/" + group_name)

            f.create_dataset("/" + group_name + "/" + name, data=np.array(data))

    def write_dataset(self, name, data, group_name="/"):
        with h5py.File(self._file_name, "a") as f:
            if group_name != "/":
                if group_name not in f["/"]:
                    f.create_group("/" + group_name)

            f.create_dataset("/" + group_name + "/" + name, data=np.array(data))

    def start_timing(self, name):
        if name not in self._timings:
            self._timings[name] = {"total": 0.0, "last_total": 0.0, "newest": 0.0, "count": 0, "init": 0.0}

        self._timings[name]["init"] = time.perf_counter()

    def stop_timing(self, name):
        toc = time.perf_counter()

        if name not in self._timings:
            self._timings[name] = {"total": 0.0, "last_total": 0.0, "newest": 0.0, "count": 0, "init": 0.0}

        elapsed = toc - self._timings[name]["init"]

        self._timings[name]["total"] += elapsed
        self._timings[name]["newest"] = elapsed
        self._timings[name]["count"] += 1

    def add_timing(self, name, elapsed):
        if name not in self._timings:
            self._timings[name] = {"total": 0.0, "last_total": 0.0, "newest": 0.0, "count": 0, "init": 0.0}

        self._timings[name]["total"] += elapsed
        self._timings[name]["newest"] = elapsed
        self._timings[name]["count"] += 1

    def print_timings(self, indent=""):
        print(f"{indent}Recorded timings:", flush=True)
        
        for key, item in self._timings.items():
            print(f"{indent}    • {key}: {item['total'] - item['last_total']}s", flush=True)

        for key in self._timings:
            self._timings[key]["last_total"] = self._timings[key]["total"]

    def to_array(self, x):
        if not isinstance(x, (np.ndarray, jnp.ndarray)):
            x = np.array([x])

        return x
    
    def _recursive_read(self, f):
        out = {}
        for key, item in f.items():
            if isinstance(item, h5py.Dataset):
                out[key] = item[()]   # read dataset
            elif isinstance(item, h5py.Group):
                out[key] = self._recursive_read(item)
        return out
    
    def to_dict(self):
        with h5py.File(self._file_name, "r") as f:
            return self._recursive_read(f)
