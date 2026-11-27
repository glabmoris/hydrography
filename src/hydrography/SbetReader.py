"""
SbetReader.py - Reader for SBET (Smoothed Best Estimate of Trajectory) files.

An SBET file is a headerless binary file made of fixed-size records. Each
record is 17 little-endian float64 values (136 bytes):

    idx  field        units
    ---  -----------  ------------------------------------
     0   time         s (GPS seconds of week)
     1   lat          rad
     2   lon          rad
     3   alt          m (ellipsoidal height)
     4   x_vel        m/s (wander frame)
     5   y_vel        m/s
     6   z_vel        m/s
     7   roll         rad
     8   pitch        rad
     9   heading      rad (platform heading)
    10   wander       rad (wander angle)
    11   x_acc        m/s^2 (body frame)
    12   y_acc        m/s^2
    13   z_acc        m/s^2
    14   x_ang_rate   rad/s (body frame)
    15   y_ang_rate   rad/s
    16   z_ang_rate   rad/s

Example
-------
>>> sbet = SbetReader("trajectory.out")
>>> arr = sbet.to_array()                 # (N, 17) float64 array
>>> rec = sbet.read(degrees=True)         # structured array, angles in degrees
>>> rec["lat"], rec["lon"], rec["alt"]
>>> sbet["time"]                          # single column by name
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np
import numpy.lib.recfunctions as rfn

PathLike = Union[str, os.PathLike]


class SbetReader:
    """Read a SBET file into NumPy arrays.

    Parameters
    ----------
    path : str or PathLike
        Path to the SBET file (often ``sbet_*.out``).
    memmap : bool, default False
        If True, memory-map the file instead of loading it into RAM. Useful
        for very large trajectories; data is read lazily from disk.
    strict : bool, default True
        If True, raise when the file size is not a multiple of the record
        size. If False, warn and ignore the trailing partial record.
    """

    FIELDS = (
        "time", "lat", "lon", "alt",
        "x_vel", "y_vel", "z_vel",
        "roll", "pitch", "heading", "wander",
        "x_acc", "y_acc", "z_acc",
        "x_ang_rate", "y_ang_rate", "z_ang_rate",
    )
    ANGLE_FIELDS = ("lat", "lon", "roll", "pitch", "heading", "wander")
    ANGULAR_RATE_FIELDS = ("x_ang_rate", "y_ang_rate", "z_ang_rate")

    DTYPE = np.dtype([(name, "<f8") for name in FIELDS])
    RECORD_SIZE = DTYPE.itemsize  # 136 bytes

    def __init__(self, path: PathLike, *, memmap: bool = False, strict: bool = True):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"SBET file not found: {self.path}")

        size = self.path.stat().st_size
        self.n_records, remainder = divmod(size, self.RECORD_SIZE)
        if remainder:
            msg = (
                f"{self.path.name}: size {size} bytes is not a multiple of "
                f"{self.RECORD_SIZE}; {remainder} trailing bytes will be ignored."
            )
            if strict:
                raise ValueError(msg)
            warnings.warn(msg)

        self._memmap = memmap
        self._data: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    @property
    def data(self) -> np.ndarray:
        """Raw structured array (radians), loaded lazily on first access."""
        if self._data is None:
            self._data = self._load()
        return self._data

    def _load(self) -> np.ndarray:
        if self.n_records == 0:
            return np.empty(0, dtype=self.DTYPE)
        if self._memmap:
            return np.memmap(self.path, dtype=self.DTYPE, mode="r",
                             shape=(self.n_records,))
        return np.fromfile(self.path, dtype=self.DTYPE, count=self.n_records)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def read(
        self,
        *,
        degrees: bool = False,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> np.ndarray:
        """Return the records as a structured array with named fields.

        Parameters
        ----------
        degrees : bool, default False
            Convert lat, lon, roll, pitch, heading, wander (and angular
            rates) from radians to degrees.
        start_time, end_time : float, optional
            Keep only records with ``start_time <= time <= end_time``.
        """
        arr = self.data

        if start_time is not None or end_time is not None:
            t = arr["time"]
            mask = np.ones(t.shape, dtype=bool)
            if start_time is not None:
                mask &= t >= start_time
            if end_time is not None:
                mask &= t <= end_time
            arr = arr[mask]  # boolean indexing always copies

        if degrees:
            arr = np.array(arr, copy=True)  # never modify the cached/memmapped data
            for name in self.ANGLE_FIELDS + self.ANGULAR_RATE_FIELDS:
                arr[name] = np.degrees(arr[name])

        return arr

    def to_array(self, **kwargs) -> np.ndarray:
        """Return the records as a plain ``(N, 17)`` float64 array.

        Columns follow the order in :attr:`FIELDS`. Accepts the same keyword
        arguments as :meth:`read`.
        """
        return rfn.structured_to_unstructured(self.read(**kwargs))

    def iter_chunks(self, chunk_size: int = 100_000) -> Iterator[np.ndarray]:
        """Yield structured-array chunks without loading the whole file."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        with open(self.path, "rb") as f:
            remaining = self.n_records
            while remaining > 0:
                count = min(chunk_size, remaining)
                yield np.fromfile(f, dtype=self.DTYPE, count=count)
                remaining -= count

    def to_dataframe(self, **kwargs):
        """Return the records as a pandas DataFrame (requires pandas)."""
        import pandas as pd
        return pd.DataFrame(self.read(**kwargs))

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    @property
    def time_span(self) -> tuple[float, float]:
        """(first_time, last_time) in seconds; (nan, nan) if empty."""
        if self.n_records == 0:
            return (float("nan"), float("nan"))
        t = self.data["time"]
        return float(t[0]), float(t[-1])

    def __len__(self) -> int:
        return self.n_records

    def __getitem__(self, key):
        """``sbet["lat"]`` returns a column; ints/slices return records."""
        if isinstance(key, str) and key not in self.FIELDS:
            raise KeyError(f"Unknown field {key!r}. Valid fields: {self.FIELDS}")
        return self.data[key]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(path={str(self.path)!r}, n_records={self.n_records})"
