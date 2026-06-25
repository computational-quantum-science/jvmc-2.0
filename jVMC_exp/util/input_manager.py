from __future__ import annotations

from pathlib import Path
from typing import Any

import flax.serialization
import h5py
import numpy as np


class InputManager:
    """Load data and parameter checkpoints written by OutputManager.

    Examples
    --------
        inp = InputManager("results.h5")
        latest_params = inp.load_parameters(template_params, step="latest")
        step_0_params = inp.load_parameters(template_params, step=0)
        data = inp.load_from_h5()
    """

    def __init__(self, path: str | Path, group: str = "/"):
        self.path: Path | None = None
        self.group = self._normalize_group(group)
        self.data: dict[str, Any] = {}
        self.set_path(path)

    @property
    def get_data(self):
        return self.data

    @staticmethod
    def _normalize_group(group: str) -> str:
        group = str(group or "/").strip()
        if group == "/":
            return "/"
        return "/" + group.strip("/")

    def _root(self, handle):
        return handle[self.group]

    def set_path(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.data = self.load_from_h5(self.path, group=self.group)


    def get_network_checkpoint(self, time: float = None, idx: int = None):
        if time is not None and idx is not None:
            raise ValueError("Specify either 'time' or 'idx', not both.")

        times = self.data["network_checkpoints"]["times"]
        checkpoints = self.data["network_checkpoints"]["checkpoints"]

        if len(times) == 0:
            raise RuntimeError("No checkpoints have been recorded yet.")

        if time is not None:
            idx = int(np.argmin(np.abs(np.array(times) - time)))
        elif idx is None:
            idx = -1

        return times[idx], checkpoints[idx]
    
    @staticmethod
    def load_from_h5(path: str | Path, group: str = "/") -> dict:
        def read_recursive(h5_group):
            out = {}
            for key, item in h5_group.items():
                if isinstance(item, h5py.Dataset):
                    out[key] = item[()]
                elif isinstance(item, h5py.Group):
                    out[key] = read_recursive(item)
            return out

        group = InputManager._normalize_group(group)
        with h5py.File(Path(path), "r") as handle:
            return read_recursive(handle[group])

    def _read_param_tree(self, group) -> dict[str, Any]:
        out = {}
        for key, value in group.items():
            if isinstance(value, h5py.Group):
                out[key] = self._read_param_tree(value)
            else:
                out[key] = value[()]
        return out

    def load_parameters(self, template_params, step: int | str = "latest"):
        with h5py.File(self.path, "r") as handle:
            params_root = self._root(handle)["parameters"]
            group_name = params_root.attrs["latest"] if step == "latest" else f"{int(step):08d}"
            if isinstance(group_name, bytes):
                group_name = group_name.decode("utf-8")
            state = self._read_param_tree(params_root[group_name])
        return flax.serialization.from_state_dict(template_params, state)