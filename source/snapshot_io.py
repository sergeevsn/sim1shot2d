# -*- coding: utf-8 -*-
"""
Snapshot storage in HDF5: write frames during simulation and read them lazily
for visualization and RTM.
"""
import numpy as np
import h5py


# Dataset names in HDF5 (without fwd/bwd suffix)
COMPONENT_DSETS = ("P", "Vz", "Vx")

# Attributes stored in the HDF5 file root
ATTR_N_SAVE = "n_save"
ATTR_NX = "nx"
ATTR_NZ = "nz"
ATTR_RUN_TYPE = "run_type"
ATTR_SNAPSHOT_DT_MS = "snapshot_dt_ms"
ATTR_TMAX_MS = "tmax_ms"
ATTR_SEISMOGRAM_SOURCE = "seismogram_source"


def create_snapshots_h5_writer(
    path,
    n_save,
    nx,
    nz,
    run_type="forward",
    snapshot_dt_ms=None,
    tmax_ms=None,
    seismogram_source=None,
    **extra_attrs
):
    """
    Create an HDF5 file for snapshot storage and return a lightweight writer
    object for frame‑by‑frame writing.

    Parameters
    ----------
    path : str
        Path to the .h5 file.
    n_save, nx, nz : int
        Dimensions: number of saved frames and grid size (nx, nz) per frame.
    run_type : {"forward", "backward"}
        Run type.
    snapshot_dt_ms, tmax_ms : float, optional
        Metadata to reconstruct time axis.
    seismogram_source : str, optional
        Name of the seismogram (for backward runs).
    **extra_attrs
        Extra attributes to store in the file root.

    Returns
    -------
    writer : object
        Has methods write_frame(save_idx, p_slice, vx_slice, vz_slice) and close().
    """
    f = h5py.File(path, "w")
    for name in COMPONENT_DSETS:
        f.create_dataset(name, shape=(n_save, nx, nz), dtype=np.float32, chunks=(1, nx, nz))
    f.attrs[ATTR_N_SAVE] = n_save
    f.attrs[ATTR_NX] = nx
    f.attrs[ATTR_NZ] = nz
    f.attrs[ATTR_RUN_TYPE] = run_type
    if snapshot_dt_ms is not None:
        f.attrs[ATTR_SNAPSHOT_DT_MS] = float(snapshot_dt_ms)
    if tmax_ms is not None:
        f.attrs[ATTR_TMAX_MS] = float(tmax_ms)
    if seismogram_source is not None:
        f.attrs[ATTR_SEISMOGRAM_SOURCE] = str(seismogram_source)
    for k, v in extra_attrs.items():
        try:
            f.attrs[k] = v
        except (TypeError, ValueError):
            pass

    class _Writer:
        def __init__(self, h5file):
            self._file = h5file

        def write_frame(self, save_idx, p_slice, vx_slice, vz_slice):
            """Write one frame. p_slice, vx_slice, vz_slice are (nx, nz) float32 arrays."""
            self._file["P"][save_idx] = np.asarray(p_slice, dtype=np.float32)
            if vx_slice is not None:
                self._file["Vx"][save_idx] = np.asarray(vx_slice, dtype=np.float32)
            if vz_slice is not None:
                self._file["Vz"][save_idx] = np.asarray(vz_slice, dtype=np.float32)

        def close(self):
            self._file.close()
            self._file = None

    return _Writer(f)


class H5SnapshotComponent:
    """
    Proxy for one snapshot component stored in HDF5: array‑like with .shape and [idx].
    Supports len() and time indexing, convenient for GUI display.
    """

    def __init__(self, path, dset_name):
        self._path = path
        self._dset_name = dset_name
        self._shape = None

    def _open_read(self):
        return h5py.File(self._path, "r")

    @property
    def shape(self):
        if self._shape is None:
            with self._open_read() as f:
                self._shape = f[self._dset_name].shape
        return self._shape

    @property
    def ndim(self):
        return 3

    @property
    def size(self):
        return int(np.prod(self.shape))

    def read_full(self):
        """Load the full array (n_save, nx, nz) into memory (for percentile/RTM)."""
        return read_component_full(self._path, self._dset_name)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        with self._open_read() as f:
            dset = f[self._dset_name]
            out = dset[idx]
            if not isinstance(out, np.ndarray):
                out = np.asarray(out)
            return np.asarray(out, dtype=np.float64)


def snapshots_dict_from_h5(path, run_type="forward"):
    """
    Build a dict of components for the GUI: keys are "P fwd", "Vz fwd", "Vx fwd"
    (or bwd), values are H5SnapshotComponent objects with .shape and [idx].

    Parameters
    ----------
    path : str
        Path to the .h5 file.
    run_type : {"forward", "backward"}
    """
    suffix = " fwd" if run_type == "forward" else " bwd"
    return {
        name + suffix: H5SnapshotComponent(path, name)
        for name in COMPONENT_DSETS
    }


def read_component_full(path, dset_name):
    """
    Load full (n_save, nx, nz) array for a single component from HDF5.
    Used for building seismograms from P and for RTM image construction.
    """
    with h5py.File(path, "r") as f:
        return np.asarray(f[dset_name][...], dtype=np.float64)


def compute_seismogram_from_h5(path, receivers, dx, dz, snapshot_dt_ms=None, progress_callback=None, component="P"):
    """
    Build a seismogram for a given component stored in HDF5 by reading one
    frame at a time without loading the whole 3D array into memory.

    Parameters
    ----------
    path : str
        Path to the forward snapshot .h5 file.
    receivers : list of (x, z)
        Receiver coordinates in meters.
    dx, dz : float
        Grid spacing.
    snapshot_dt_ms : float, optional
        Time sampling in ms; if None, taken from file attributes.
    progress_callback : callable, optional
        Called as progress_callback(current, total): first with (0, n_save) and
        later with (t+1, n_save) inside the loop.

    Returns
    -------
    data : (n_save, n_receivers) float64
    t_ms : (n_save,) float64
    """
    if not receivers:
        return None, None
    with h5py.File(path, "r") as f:
        comp = str(component or "P")
        if comp not in f:
            comp = "P"
        dset = f[comp]
        n_save, nx, nz = dset.shape
        if progress_callback is not None:
            progress_callback(0, n_save)
        dt_ms = float(snapshot_dt_ms) if snapshot_dt_ms is not None else float(f.attrs.get(ATTR_SNAPSHOT_DT_MS, 0))
        if dt_ms <= 0:
            dt_ms = 2.0
        # Grid node indices for each receiver
        rec_ij = []
        for x, z in receivers:
            ix = int(round(x / dx))
            jz = int(round(z / dz))
            ix = max(0, min(nx - 1, ix))
            jz = max(0, min(nz - 1, jz))
            rec_ij.append((ix, jz))
        data = np.zeros((n_save, len(receivers)), dtype=np.float64)
        for t in range(n_save):
            frame = dset[t]  # a single frame (nx, nz) in memory
            for rec_idx, (ix, jz) in enumerate(rec_ij):
                data[t, rec_idx] = float(frame[ix, jz])
            if progress_callback is not None:
                progress_callback(t + 1, n_save)
        t_ms = np.arange(n_save, dtype=np.float64) * dt_ms
    return data, t_ms


def build_rtm_image(fwd_source, bwd_source, progress_callback=None):
    """
    Build RTM image by correlating forward and backward fields in time:
    image += fwd[t] * bwd_reversed[t], with optional progress reporting.

    Parameters
    ----------
    fwd_source, bwd_source : np.ndarray (n_save, nx, nz) or (path, dset_name) tuples for HDF5.
    progress_callback : callable(current, total), optional

    Returns
    -------
    image : (nx, nz) float64  (for display you usually use image.T → (nz, nx)).
    """
    if isinstance(fwd_source, np.ndarray):
        n_save, nx, nz = fwd_source.shape
    else:
        with h5py.File(fwd_source[0], "r") as f:
            n_save, nx, nz = f[fwd_source[1]].shape

    image = np.zeros((nx, nz), dtype=np.float64)

    if isinstance(fwd_source, np.ndarray) and isinstance(bwd_source, np.ndarray):
        bwd_rev = bwd_source[::-1]
        for t in range(n_save):
            image += fwd_source[t] * bwd_rev[t]
            if progress_callback is not None:
                progress_callback(t + 1, n_save)
        return image

    fwd_file = h5py.File(fwd_source[0], "r") if isinstance(fwd_source, tuple) else None
    bwd_file = h5py.File(bwd_source[0], "r") if isinstance(bwd_source, tuple) else None
    fwd_dset = fwd_file[fwd_source[1]] if fwd_file else None
    bwd_dset = bwd_file[bwd_source[1]] if bwd_file else None
    try:
        for t in range(n_save):
            ft = np.asarray(fwd_dset[t], dtype=np.float64) if fwd_dset is not None else fwd_source[t]
            bt = np.asarray(bwd_dset[n_save - 1 - t], dtype=np.float64) if bwd_dset is not None else bwd_source[n_save - 1 - t]
            image += ft * bt
            if progress_callback is not None:
                progress_callback(t + 1, n_save)
    finally:
        if fwd_file:
            fwd_file.close()
        if bwd_file:
            bwd_file.close()
    return image


def sample_frames_for_percentile(path, dset_name, max_frames=50):
    """
    Read up to `max_frames` frames from HDF5 (evenly spaced in time) to estimate
    vmin/vmax without loading the full array — avoids OOM for huge snapshot sets.
    """
    with h5py.File(path, "r") as f:
        dset = f[dset_name]
        n_save, nx, nz = dset.shape
        if n_save <= max_frames:
            indices = list(range(n_save))
        else:
            indices = np.linspace(0, n_save - 1, max_frames, dtype=np.intp)
        sample = np.array([dset[i] for i in indices], dtype=np.float64)
    return sample


def read_snapshots_metadata(path):
    """Read only metadata from the HDF5 root (n_save, nx, nz, run_type, ...)."""
    with h5py.File(path, "r") as f:
        a = f.attrs
        return {
            "n_save": int(a.get(ATTR_N_SAVE, 0)),
            "nx": int(a.get(ATTR_NX, 0)),
            "nz": int(a.get(ATTR_NZ, 0)),
            "run_type": a.get(ATTR_RUN_TYPE, "forward"),
            "snapshot_dt_ms": float(a.get(ATTR_SNAPSHOT_DT_MS, 0)),
            "tmax_ms": float(a.get(ATTR_TMAX_MS, 0)),
            "seismogram_source": a.get(ATTR_SEISMOGRAM_SOURCE, ""),
        }
