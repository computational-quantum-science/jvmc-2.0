from __future__ import annotations
import time
from pathlib import Path
from typing import Any

import flax.serialization
import h5py
import jax
import jax.numpy as jnp
import numpy as np

def _as_h5_array(value: Any):
    try:
        arr = jax.device_get(jnp.asarray(value))
    except (TypeError, ValueError):
        arr = jax.device_get(value)
    dtype = getattr(arr, "dtype", None)
    if isinstance(arr, (str, bytes)) or dtype is not None and (dtype == object or dtype.kind in {"U", "S"}):
        if hasattr(arr, "shape") and arr.shape == ():
            text = arr.item()
        else:
            text = arr
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        return str(text)
    return arr


def check_network_parameter_shape(weights: Any) -> bool:
    try:
        _as_h5_array(weights)
    except (TypeError, ValueError):
        return False
    return True


class OutputManager:
    """
    Collect observables, metadata, timings, and optional HDF5 output.

    The manager can be used without a path for in-memory output. If a path is
    passed to the constructor, writes are mirrored to HDF5 immediately. If no
    path is passed at initialization, data stays in memory until save_to_h5() is
    called. Passing a path to save_to_h5() stores that path on the instance and
    makes it reusable by later save_to_h5() calls.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        HDF5 file to create or update.
    group : str
        Root group inside the HDF5 file. This is useful for storing multiple
        runs in the same file.
    append : bool
        If True, open an existing file for updates. If False, overwrite it.

    Common methods
    --------------
    write_observables(step, **observables)
        Append scalar, array, or nested-dictionary observables in memory and, if
        a path is bound, under ``observables`` in HDF5.
    write_metadata(step=None, **metadata)
        Write run metadata. With ``step`` it is appended as a timeseries. Without
        ``step`` each metadata entry is stored once and replaced in HDF5 if a
        path is bound.
    save_to_h5(path=None, append=False)
        Persist the in-memory data dictionary to HDF5. Passing ``path`` binds it
        for later calls.
    write_dataset(name, data, group="/")
        Add or replace an arbitrary dataset under ``group``.
    write_parameters(step, params, attrs=None)
        Save a Flax/JAX parameter tree under ``parameters/{step:08d}`` and mark
        it as the latest checkpoint.
    start_timing(name), stop_timing(name), add_timing(name, elapsed),
    flush_timings()
        Accumulate timing totals and counts.

    Examples
    --------
        outp = OutputManager()
        outp.write_observables(0, energy=-1.0)
        outp.save_to_h5("results.h5")
        outp.write_observables(1, energy=-1.1)
        outp.save_to_h5()  # reuses results.h5

        outp = OutputManager("results.h5", group="run_000", append=False)
        outp.write_metadata(system_size=64, model="RBM")
        outp.write_observables(0, energy=-1.0, magnetization={"mean": 0.2})
        outp.write_dataset("initial_samples", samples, group="diagnostics")

        outp.start_timing("optimization_step")
        params = update_network(params)
        outp.stop_timing("optimization_step")
        outp.flush_timings()

        outp.write_parameters(0, params, attrs={"optimizer": "SR"})
        outp.save_to_h5()
    """

    def __init__(self, path: str | Path | None = None, group: str = "/", append: bool = True):
        self.path: Path | None = None
        self.group = self._normalize_group(group)
        self.mode = "a"
        self.data: dict[str, Any] = {}
        self._timings: dict[str, dict[str, float | int]] = {}
        self.timings = self._timings
        self._parameter_attrs: dict[str, dict[str, Any]] = {}
        self._parameter_latest: str | None = None
        if path is not None:
            self._set_path(path, append=append)

    @staticmethod
    def _normalize_group(group: str) -> str:
        group = str(group or "/").strip()
        if group == "/":
            return "/"
        return "/" + group.strip("/")

    def _set_path(self, path: str | Path, append: bool = True) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with h5py.File(self.path, mode) as handle:
            handle.require_group(self.group)
        self.mode = "a"
        
    def _root(self, handle):
        return handle[self.group]

    def _append_h5(self, parent, name: str, value: Any) -> None:
        arr = _as_h5_array(value)
        if isinstance(arr, str):
            arr = np.asarray(arr, dtype=h5py.string_dtype(encoding="utf-8"))
        arr = np.asarray(arr)
        item_shape = arr.shape
        if name not in parent:
            parent.create_dataset(
                name,
                shape=(0,) + item_shape,
                maxshape=(None,) + item_shape,
                dtype=arr.dtype,
                chunks=True,
            )
        dataset = parent[name]
        if dataset.chunks is None:
            old_data = dataset[()]
            del parent[name]
            dataset = parent.create_dataset(
                name,
                shape=old_data.shape,
                maxshape=(None,) + old_data.shape[1:],
                dtype=old_data.dtype,
                chunks=True,
            )
            dataset[...] = old_data
        new_len = dataset.shape[0] + 1
        dataset.resize((new_len,) + dataset.shape[1:])
        dataset[-1] = arr

    def _append_nested_h5(self, parent, data: dict[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                self._append_nested_h5(parent.require_group(str(key)), value)
            else:
                self._append_h5(parent, str(key), value)

    def _add(self, subgroup: str, time_val: float | int, **kwargs) -> None:
        if subgroup not in self.data:
            self.data[subgroup] = {}
        if "times" not in self.data[subgroup]:
            self.data[subgroup]["times"] = []
        self.data[subgroup]["times"].append(time_val)
        self._store_recursive(self.data[subgroup], kwargs)

    def _store_recursive(self, store: dict, data: dict) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                if key not in store:
                    store[key] = {}
                self._store_recursive(store[key], value)
            else:
                if key not in store:
                    store[key] = []
                store[key].append(value)

    def write_observables(self, step: int | float, **observables) -> None:
        self._add("observables", step, **observables)
        if self.path is not None:
            with h5py.File(self.path, "a") as handle:
                group = self._root(handle).require_group("observables")
                self._append_h5(group, "times", step)
                self._append_nested_h5(group, observables)

    def write_metadata(self, step: int | float | None = None, **metadata) -> None:
        if step is not None:
            self._add("metadata", step, **metadata)
        else:
            self.data.setdefault("metadata", {}).update(metadata)

        if self.path is not None:
            with h5py.File(self.path, "a") as handle:
                group = self._root(handle).require_group("metadata")
                if step is not None:
                    self._append_h5(group, "times", step)
                    self._append_nested_h5(group, metadata)
                else:
                    for key, value in metadata.items():
                        arr = _as_h5_array(value)
                        if key in group:
                            del group[key]
                        group.create_dataset(str(key), data=arr)

    def write_network_checkpoint(self, time: float, weights) -> None:
        if check_network_parameter_shape(weights):
            self._add("network_checkpoints", time, checkpoints=weights)
        else:
            raise ValueError("Network weights must be serialized for checkpointing.")
        if self.path is not None:
            with h5py.File(self.path, "a") as handle:
                group = self._root(handle).require_group("network_checkpoints")
                self._append_h5(group, "times", time)
                self._append_h5(group, "checkpoints", weights)

    def write_dataset(self, name: str, data: Any, group: str = "/", path: str | Path | None = None) -> None:
        if path is not None:
            raise ValueError("write_dataset() does not bind paths. Use save_to_h5(path=...) to choose an output file.")
        self._store_dataset(name, data, group=group)
        if self.path is not None:
            with h5py.File(self.path, "a") as handle:
                self._write_dataset_to_h5(handle, name, data, group=group)

    def _store_dataset(self, name: str, data: Any, group: str = "/") -> None:
        store = self.data
        group = self._normalize_group(group)
        if group != "/":
            for part in group.strip("/").split("/"):
                store = store.setdefault(part, {})
        store[str(name)] = data

    def _write_dataset_to_h5(self, handle, name: str, data: Any, group: str = "/") -> None:
        root = self._root(handle)
        parent = root if group == "/" else root.require_group(group.strip("/"))
        if name in parent:
            del parent[name]
        parent.create_dataset(str(name), data=_as_h5_array(data))

    def _write_param_tree(self, group, tree: dict[str, Any]) -> None:
        for key, value in tree.items():
            if isinstance(value, dict):
                self._write_param_tree(group.require_group(str(key)), value)
            else:
                if key in group:
                    del group[key]
                group.create_dataset(str(key), data=_as_h5_array(value))

    def _read_param_tree(self, group) -> dict[str, Any]:
        out = {}
        for key, value in group.items():
            if isinstance(value, h5py.Group):
                out[key] = self._read_param_tree(value)
            else:
                out[key] = value[()]
        return out

    def write_parameters(
        self,
        step: int,
        params,
        attrs: dict[str, Any] | None = None,
        path: str | Path | None = None,
    ) -> str:
        if path is not None:
            raise ValueError("write_parameters() does not bind paths. Use save_to_h5(path=...) to choose an output file.")
        state = flax.serialization.to_state_dict(params)
        group_name = f"{int(step):08d}"
        self.data.setdefault("parameters", {})[group_name] = state
        checkpoint_attrs = {"step": int(step)}
        checkpoint_attrs.update(attrs or {})
        self._parameter_attrs[group_name] = checkpoint_attrs
        self._parameter_latest = group_name
        if self.path is not None:
            with h5py.File(self.path, "a") as handle:
                self._write_parameters_to_h5(handle, group_name, state, checkpoint_attrs)
        return group_name

    def save_to_h5(self, path: str | Path | None = None, append: bool = False) -> None:
        if path is not None:
            self._set_path(path, append=append)
        if self.path is None:
            raise ValueError("A path is required for this operation. Pass a path to save_to_h5().")
    
        mode = "a" if append else "w"
        with h5py.File(self.path, mode) as handle:
            handle.require_group(self.group)
            self._write_dict_to_h5(self._root(handle), self.data)
            self._write_parameter_attrs_to_h5(handle)

    def _write_parameters_to_h5(self, handle, group_name: str, state: dict[str, Any], attrs: dict[str, Any]) -> None:
        params_root = self._root(handle).require_group("parameters")
        if group_name in params_root:
            del params_root[group_name]
        group = params_root.create_group(group_name)
        for key, value in attrs.items():
            group.attrs[str(key)] = value
        self._write_param_tree(group, state)
        params_root.attrs["latest"] = group_name

    def _write_parameter_attrs_to_h5(self, handle) -> None:
        if "parameters" not in self.data:
            return
        params_root = self._root(handle).require_group("parameters")
        if self._parameter_latest is not None:
            params_root.attrs["latest"] = self._parameter_latest
        for group_name, attrs in self._parameter_attrs.items():
            if group_name not in params_root:
                continue
            for key, value in attrs.items():
                params_root[group_name].attrs[str(key)] = value

    def _write_dict_to_h5(self, parent, data: dict) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                group = parent.require_group(str(key))
                self._write_dict_to_h5(group, value)
            else:
                if key in parent:
                    del parent[key]
                parent.create_dataset(str(key), data=_as_h5_array(value))

    def start_timing(self, name: str) -> None:
        entry = self._timings.setdefault(name, {"total": 0.0, "last_total": 0.0, "count": 0, "start": 0.0})
        entry["start"] = time.perf_counter()

    def stop_timing(self, name: str) -> float:
        elapsed = time.perf_counter() - float(self._timings[name]["start"])
        self.add_timing(name, elapsed)
        return elapsed

    def add_timing(self, name: str, elapsed: float) -> None:
        entry = self._timings.setdefault(name, {"total": 0.0, "last_total": 0.0, "count": 0, "start": 0.0})
        entry["total"] = float(entry["total"]) + float(elapsed)
        entry["count"] = int(entry["count"]) + 1

    def print_timings(self, indent: str = "") -> None:
        print(f"{indent}Recorded timings:", flush=True)
        for key, item in self._timings.items():
            delta = item["total"] - item["last_total"]
            print(f"{indent}    - {key}: {delta:.6f}s", flush=True)
            item["last_total"] = item["total"]

    def flush_timings(self, path: str | Path | None = None) -> None:
        if path is not None:
            raise ValueError("flush_timings() does not bind paths. Use save_to_h5(path=...) to choose an output file.")
        timings = self.data.setdefault("timings", {})
        for key, value in self._timings.items():
            timings[str(key)] = {
                "total": value["total"],
                "count": value["count"],
            }
        if self.path is not None:
            with h5py.File(self.path, "a") as handle:
                self._write_dict_to_h5(self._root(handle).require_group("timings"), timings)

    def get_network_checkpoint(self, time: float | None = None, idx: int | None = None):
        if time is not None and idx is not None:
            raise ValueError("Specify either 'time' or 'idx', not both.")
        checkpoints = self.data.get("network_checkpoints", {})
        times = checkpoints.get("times", [])
        weights = checkpoints.get("checkpoints", [])
        if not times:
            raise RuntimeError("No checkpoints have been recorded yet.")
        if time is not None:
            idx = int(np.argmin(np.abs(np.asarray(times) - time)))
        elif idx is None:
            idx = -1
        return times[idx], weights[idx]