"""
Load a velocity model from SEG-Y for the GUI.

dz = f.bin[segyio.BinField.Interval] / 1e6
dx = X‑step from CDP_X attributes (median of differences between unique X values).
"""
import numpy as np
import segyio


def load_velocity_from_segy(path):
    """
    Load a velocity model from a SEG-Y file.

    Parameters
    ----------
    path : str
        Path to the .sgy / .segy file.

    Notes
    -----
    dz = f.bin[segyio.BinField.Interval] / 1e6
    dx = median difference between neighbouring unique CDP_X values (grid step along X).

    Returns
    -------
    vp, dx, dz
        vp : (nz, nx) float64; axis 0 — depth Z, axis 1 — X.
        On error raises RuntimeError.
    """
    try:
        with segyio.open(path, "r", ignore_geometry=True) as f:
            # dz from binary header (sample interval), in meters after division by 1e3
            interval = f.bin[segyio.BinField.Interval]
            dz = float(interval) / 1_000.0

            # dx — X step: median of differences between unique CDP_X values
            # f.attributes returns a generator — convert to a list
            cdpx = np.array(list(f.attributes(segyio.TraceField.CDP_X)), dtype=np.float64)
            ux = np.unique(np.sort(cdpx))
            if len(ux) > 1:
                diffs = np.diff(ux)
                dx = float(np.median(diffs))
            else:
                dx = float(np.median(cdpx)) if len(cdpx) > 0 else 1.0

            # Data: traces — columns along X, samples — along depth Z
            # f.trace[:] in segyio is a generator — convert to a list
            raw = f.trace.raw[:]
            vp = raw.T  # (n_samples, n_traces) = (nz, nx)
        return vp, dx, dz
    except Exception as e:
        raise RuntimeError("SEG-Y load error: {}: {}".format(type(e).__name__, e)) from e
