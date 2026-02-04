"""
Acoustic 1st‑order simulation (velocity–pressure system).

Equations:
  ∂vx/∂t = - (1/ρ) ∂P/∂x
  ∂vz/∂t = - (1/ρ) ∂P/∂z
  ∂P/∂t  = - ρ c² (∂vx/∂x + ∂vz/∂z)
"""

from numba import jit
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


def ricker(f_hz, nt, dt, t0=None):
    """Ricker wavelet; f_hz — dominant frequency (Hz), nt — number of samples, dt — time step (s)."""
    t = np.arange(nt, dtype=np.float32) * dt
    if t0 is None:
        t0 = 1.0 / f_hz
    s = (1 - 2 * (np.pi * f_hz * (t - t0)) ** 2) * np.exp(-(np.pi * f_hz * (t - t0)) ** 2)
    return s.astype(np.float32)


def prepare_migration_velocity(vp_true, sigma_meters, dx, dz):
    """
    Smooth 2D velocity model with a Gaussian filter.

    vp_true : (nx, nz)
        True P‑wave velocity model.
    sigma_meters : float
        Smoothing radius in meters.

    Returns
    -------
    vp_mig : (nx, nz)
        Smoothed velocity model of the same shape as vp_true.
    """
    sigma_x = sigma_meters / dx
    sigma_z = sigma_meters / dz
    vp_mig = gaussian_filter(vp_true, sigma=[sigma_x, sigma_z], mode="nearest")
    return vp_mig


def _inject_record_interp(p_it, record_vals, xrec, zrec, dx, dz, n_absorb, scale):
    """
    Add recorded pressure samples into the pressure field `p_it` at receiver positions (xrec, zrec).

    For collinear receivers injection is done into the nearest grid nodes; otherwise
    interpolation is performed with `griddata(..., method=\"linear\")`.

    Parameters
    ----------
    scale : float
        Scaling factor (for the 1st‑order scheme we usually pass dt here).
    """
    nx, nz = p_it.shape
    xrec = np.asarray(xrec, dtype=np.float64).ravel()
    zrec = np.asarray(zrec, dtype=np.float64).ravel()
    vals = np.asarray(record_vals, dtype=np.float64).ravel()
    x_phys = (np.arange(nx) - n_absorb) * dx
    z_phys = (np.arange(nz) - n_absorb) * dz

    if np.unique(xrec).size == 1:
        ix = int(round((xrec[0] + n_absorb * dx) / dx))
        ix = max(0, min(nx - 1, ix))
        jrec = np.floor(zrec / dz).astype(int) + int(n_absorb)
        jrec = np.clip(jrec, 0, nz - 1)
        vals_f = np.asarray(vals, dtype=np.float32) * scale
        np.add.at(p_it, (np.full_like(jrec, ix), jrec), vals_f)
        return

    if np.unique(zrec).size == 1:
        jz = int(round((zrec[0] + n_absorb * dz) / dz))
        jz = max(0, min(nz - 1, jz))
        irec = np.floor(xrec / dx).astype(int) + int(n_absorb)
        irec = np.clip(irec, 0, nx - 1)
        vals_f = np.asarray(vals, dtype=np.float32) * scale
        np.add.at(p_it, (irec, np.full_like(irec, jz)), vals_f)
        return

    X_grid, Z_grid = np.meshgrid(x_phys, z_phys, indexing="ij")
    points_rec = np.column_stack([xrec, zrec])
    interp = griddata(points_rec, vals, (X_grid, Z_grid), method="linear", fill_value=0.0)
    p_it += np.asarray(interp, dtype=np.float32) * scale


@jit(nopython=True)
def update_velocity_stress_staggered(p, vx, vz, vp, rho, dt, dx, dz, nx, nz):
    """
    Update fields on a staggered grid, 2nd‑order in space.

    P  is stored at integer nodes [i, j],
    Vx at [i+0.5, j] (staggered in X), updated for i = 0..nx-2,
    Vz at [i, j+0.5] (staggered in Z), updated for j = 0..nz-2.

    Arrays p, vx, vz all have shape (nx, nz); values outside the update stencil are left unchanged.
    """
    # 1. Update velocities (Vx, Vz) from pressure P
    # Vx: dP/dx = (P[i+1] - P[i]) / dx  (2nd order)
    for i in range(nx - 1):
        for j in range(nz):
            rho_x = 0.5 * (rho[i + 1, j] + rho[i, j])
            vx[i, j] = vx[i, j] - (dt / rho_x) * (p[i + 1, j] - p[i, j]) / dx

    # Vz: dP/dz = (P[i,j+1] - P[i,j]) / dz
    for i in range(nx):
        for j in range(nz - 1):
            rho_z = 0.5 * (rho[i, j + 1] + rho[i, j])
            vz[i, j] = vz[i, j] - (dt / rho_z) * (p[i, j + 1] - p[i, j]) / dz

    # 2. Update pressure P from velocity divergence
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            k_modulus = rho[i, j] * vp[i, j] ** 2
            div_v = (vx[i, j] - vx[i - 1, j]) / dx + (vz[i, j] - vz[i, j - 1]) / dz
            p[i, j] = p[i, j] - dt * k_modulus * div_v

    return p, vx, vz


@jit(nopython=True)
def update_velocity_stress_staggered_4th(p, vx, vz, vp, rho, dt, dx, dz, nx, nz):
    """
    Same scheme, but 4th‑order in space in the interior — less numerical dispersion.

    First‑derivative stencil on half‑shifted nodes:
    f'_{i+1/2} = (9(f_{i+1}-f_i) - (f_{i+2}-f_{i-1})) / (8*dx).

    Near the boundary (one cell from each side) we fall back to 2nd order so that
    the whole grid is updated.
    """
    idx = 1.0 / dx
    idz = 1.0 / dz
    c_x  =  9.0 /  8.0 * idx      # same as before — correct
    c_x2 = -1.0 / 24.0 * idx      # main change compared to 2nd order: 8 → 24
    c_z  =  9.0 /  8.0 * idz      # same
    c_z2 = -1.0 / 24.0 * idz

    # 1. Vx: interior i=1..nx-3 → 4th order; i=0 and i=nx-2 → 2nd order
    for i in range(nx - 1):
        for j in range(nz):
            rho_x = 0.5 * (rho[i + 1, j] + rho[i, j])
            if 1 <= i <= nx - 3:
                dp_dx = c_x * (p[i + 1, j] - p[i, j]) + c_x2 * (p[i + 2, j] - p[i - 1, j])
            else:
                dp_dx = (p[i + 1, j] - p[i, j]) * idx
            vx[i, j] = vx[i, j] - (dt / rho_x) * dp_dx

    # Vz: interior j=1..nz-3 → 4th order; boundaries → 2nd order
    for i in range(nx):
        for j in range(nz - 1):
            rho_z = 0.5 * (rho[i, j + 1] + rho[i, j])
            if 1 <= j <= nz - 3:
                dp_dz = c_z * (p[i, j + 1] - p[i, j]) + c_z2 * (p[i, j + 2] - p[i, j - 1])
            else:
                dp_dz = (p[i, j + 1] - p[i, j]) * idz
            vz[i, j] = vz[i, j] - (dt / rho_z) * dp_dz

    # 2. P: interior (2..nx-3, 2..nz-3) → 4th order; elsewhere → 2nd order
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            k_modulus = rho[i, j] * vp[i, j] ** 2
            if 2 <= i <= nx - 3 and 2 <= j <= nz - 3:
                dvx_dx = c_x * (vx[i, j] - vx[i - 1, j]) + c_x2 * (vx[i + 1, j] - vx[i - 2, j])
                dvz_dz = c_z * (vz[i, j] - vz[i, j - 1]) + c_z2 * (vz[i, j + 1] - vz[i, j - 2])
            else:
                dvx_dx = (vx[i, j] - vx[i - 1, j]) * idx
                dvz_dz = (vz[i, j] - vz[i, j - 1]) * idz
            div_v = dvx_dx + dvz_dz
            p[i, j] = p[i, j] - dt * k_modulus * div_v

    return p, vx, vz


def absorb(nx, nz, thickness):
    """Absorbing boundary coefficients (same shape as in classic simlib)."""
    FW = thickness
    a = 0.0053
    coeff = np.exp(-(a ** 2) * np.arange(FW, dtype=np.float32) ** 2)
    absorb_coeff = np.ones((nx, nz), dtype=np.float32)
    for i in range(FW):
        absorb_coeff[i, :] *= coeff[FW - 1 - i]
        absorb_coeff[nx - 1 - i, :] *= coeff[FW - 1 - i]
    for j in range(FW):
        absorb_coeff[:, j] *= coeff[FW - 1 - j]
        absorb_coeff[:, nz - 1 - j] *= coeff[FW - 1 - j]
    return absorb_coeff


def fd2d_forward_first_order(
    src,
    vp,
    nt,
    dt,
    dx,
    dz,
    xsrc,
    zsrc,
    rho=None,
    n_absorb=50,
    save_every=1,
    return_vz=True,
    return_vx=False,
    order=4,
    progress_callback=None,
    snapshots_h5_path=None,
    snapshot_dt_ms=None,
    tmax_ms=None,
):
    """
    Forward 2D acoustic modelling in 1st‑order velocity–pressure form,
    Virieux‑style staggered‑grid scheme.

    Parameters
    ----------
    src : (nt,) array
        Source time function.
    vp : (nx, nz) array
        P‑wave velocity [m/s].
    nt, dt, dx, dz : int, float
        Number of time steps and sampling intervals in time and space.
    xsrc, zsrc : float
        Source coordinates in meters.
    rho : (nx, nz) array or float, optional
        Density. If None, 1.0 is used.
    n_absorb : int
        Absorbing boundary thickness in grid nodes.
    save_every : int
        Save field snapshots every `save_every` steps (1 = every step).
    return_vz : bool
        Whether to store vz history (needed for P‑up/P‑down separation).
    return_vx : bool
        Whether to store vx history (needed for p_diff_from_poynting). If True returns (p, vx, vz).
    order : int, 2 or 4
        Spatial approximation order. Default 4 (less numerical dispersion).
    progress_callback : callable, optional
        If provided, called as progress_callback(current, total) on each time step
        (e.g. to drive a GUI progress bar).

    Returns
    -------
    If return_vx is True: (p_cropped, vx_cropped, vz_cropped).
    Else if return_vz is True: (p_cropped, vz_cropped). Otherwise: (p_cropped, None).
    """
    vp_padded = np.pad(vp, n_absorb, mode="edge")
    nx, nz = vp_padded.shape

    if rho is None:
        rho = np.ones_like(vp_padded, dtype=np.float32)
    else:
        rho = np.asarray(rho, dtype=np.float32)
        if rho.ndim == 0:
            rho = np.full_like(vp_padded, float(rho))
        else:
            rho = np.pad(rho, n_absorb, mode="edge")

    isrc = int(round(xsrc / dx)) + n_absorb
    jsrc = int(round(zsrc / dz)) + n_absorb
    if order == 4:
        isrc = max(2, min(nx - 3, isrc))
        jsrc = max(2, min(nz - 3, jsrc))
    else:
        isrc = max(1, min(nx - 2, isrc))
        jsrc = max(1, min(nz - 2, jsrc))

    p = np.zeros((nx, nz), dtype=np.float32)
    vx = np.zeros((nx, nz), dtype=np.float32)
    vz = np.zeros((nx, nz), dtype=np.float32)

    nx_crop = vp.shape[0]
    nz_crop = vp.shape[1]
    # n_save: 0, save_every, 2*save_every, ... и последний шаг (nt-1), чтобы Tmax был включён
    n_save = (nt - 1) // save_every + 1
    if (nt - 1) % save_every != 0:
        n_save += 1
    write_to_h5 = snapshots_h5_path is not None
    if write_to_h5:
        from . import snapshot_io
        _tmax_ms = float(tmax_ms) if tmax_ms is not None else nt * dt * 1000.0
        _snapshot_dt_ms = float(snapshot_dt_ms) if snapshot_dt_ms is not None else save_every * dt * 1000.0
        writer = snapshot_io.create_snapshots_h5_writer(
            snapshots_h5_path, n_save, nx_crop, nz_crop,
            run_type="forward", snapshot_dt_ms=_snapshot_dt_ms, tmax_ms=_tmax_ms,
        )
        p_history = vx_history = vz_history = None
    else:
        writer = None
        p_history = np.zeros((n_save, nx_crop, nz_crop), dtype=np.float32)
        vx_history = np.zeros((n_save, nx_crop, nz_crop), dtype=np.float32) if return_vx else None
        vz_history = np.zeros((n_save, nx_crop, nz_crop), dtype=np.float32) if return_vz else None

    absorb_coeff = absorb(nx, nz, n_absorb)

    i0, i1 = n_absorb, n_absorb + nx_crop
    j0, j1 = n_absorb, n_absorb + nz_crop

    update_fn = update_velocity_stress_staggered_4th if order == 4 else update_velocity_stress_staggered
    save_idx = 0
    for it in range(nt):
        update_fn(p, vx, vz, vp_padded, rho, dt, dx, dz, nx, nz)

        # Inject source into pressure (explosive source)
        p[isrc, jsrc] += src[it]

        p *= absorb_coeff
        vx *= absorb_coeff
        vz *= absorb_coeff

        if it % save_every == 0 and save_idx < n_save:
            if write_to_h5:
                writer.write_frame(
                    save_idx,
                    p[i0:i1, j0:j1],
                    vx[i0:i1, j0:j1] if return_vx else None,
                    vz[i0:i1, j0:j1] if return_vz else None,
                )
            else:
                p_history[save_idx] = p[i0:i1, j0:j1]
                if return_vx:
                    vx_history[save_idx] = vx[i0:i1, j0:j1]
                if return_vz:
                    vz_history[save_idx] = vz[i0:i1, j0:j1]
            save_idx += 1

        if progress_callback is not None:
            progress_callback(it + 1, nt)

    # Сохранить последний шаг (nt-1), если он ещё не сохранён — чтобы Tmax был в снапшотах
    if (nt - 1) % save_every != 0 and save_idx < n_save:
        if write_to_h5:
            writer.write_frame(
                save_idx,
                p[i0:i1, j0:j1],
                vx[i0:i1, j0:j1] if return_vx else None,
                vz[i0:i1, j0:j1] if return_vz else None,
            )
        else:
            p_history[save_idx] = p[i0:i1, j0:j1]
            if return_vx:
                vx_history[save_idx] = vx[i0:i1, j0:j1]
            if return_vz:
                vz_history[save_idx] = vz[i0:i1, j0:j1]
        save_idx += 1

    if write_to_h5:
        writer.close()
        return (snapshots_h5_path,)
    if save_idx < n_save:
        p_history = p_history[:save_idx]
        if return_vx:
            vx_history = vx_history[:save_idx]
        if return_vz:
            vz_history = vz_history[:save_idx]

    if return_vx:
        return (p_history, vx_history, vz_history)
    return (p_history, vz_history) if return_vz else (p_history, None)


def fd2d_backward_first_order(
    record,
    vp,
    nt,
    dt,
    dx,
    dz,
    xrec,
    zrec,
    rho=None,
    n_absorb=50,
    save_every=1,
    return_vz=True,
    return_vx=False,
    order=4,
    progress_callback=None,
    snapshots_h5_path=None,
    snapshot_dt_ms=None,
    tmax_ms=None,
    seismogram_source=None,
):
    """
    Reverse‑time propagation for RTM: inject recorded traces backwards in time,
    using the same 1st‑order (velocity–pressure) scheme as fd2d_forward_first_order.

    Parameters
    ----------
    record : (nt, nrec) array
        Recorded pressure at receivers (forward time). nrec is number of receivers.
    vp : (nx, nz) array
        P‑wave velocity [m/s].
    nt, dt, dx, dz : int, float
        Number of time steps and grid spacings.
    xrec, zrec : (nrec,) array
        Receiver coordinates in meters.
    rho : (nx, nz) array or float, optional
        Density. If None, 1.0 is used.
    n_absorb : int
        Absorbing boundary thickness.
    save_every : int
        Save snapshots every `save_every` steps.
    return_vz : bool
        Whether to store vz history.
    return_vx : bool
        Whether to store vx history (for p_diff_from_poynting). If True returns (p, vx, vz).
    order : int, 2 or 4
        Spatial order. Default 4 (should match the forward run for RTM).
    progress_callback : callable, optional
        If provided, called as progress_callback(current, total) at each time step.

    Returns
    -------
    If return_vx is True: (p_cropped, vx_cropped, vz_cropped).
    Else if return_vz is True: (p_cropped, vz_cropped). Otherwise: (p_cropped, None).
    """
    vp_padded = np.pad(vp, n_absorb, mode="edge")
    nx, nz = vp_padded.shape

    if rho is None:
        rho = np.ones_like(vp_padded, dtype=np.float32)
    else:
        rho = np.asarray(rho, dtype=np.float32)
        if rho.ndim == 0:
            rho = np.full_like(vp_padded, float(rho))
        else:
            rho = np.pad(rho, n_absorb, mode="edge")

    record = np.asarray(record, dtype=np.float32)
    if record.ndim == 1:
        record = record[:, np.newaxis]
    record_rev = record[::-1]

    p = np.zeros((nx, nz), dtype=np.float32)
    vx = np.zeros((nx, nz), dtype=np.float32)
    vz = np.zeros((nx, nz), dtype=np.float32)

    nx_crop = vp.shape[0]
    nz_crop = vp.shape[1]
    # n_save: 0, save_every, ... и последний шаг (nt-1), чтобы Tmax был включён
    n_save = (nt - 1) // save_every + 1
    if (nt - 1) % save_every != 0:
        n_save += 1
    write_to_h5 = snapshots_h5_path is not None
    if write_to_h5:
        from . import snapshot_io
        _tmax_ms = float(tmax_ms) if tmax_ms is not None else nt * dt * 1000.0
        _snapshot_dt_ms = float(snapshot_dt_ms) if snapshot_dt_ms is not None else save_every * dt * 1000.0
        writer = snapshot_io.create_snapshots_h5_writer(
            snapshots_h5_path, n_save, nx_crop, nz_crop,
            run_type="backward", snapshot_dt_ms=_snapshot_dt_ms, tmax_ms=_tmax_ms,
            seismogram_source=seismogram_source,
        )
        p_history = vx_history = vz_history = None
    else:
        writer = None
        p_history = np.zeros((n_save, nx_crop, nz_crop), dtype=np.float32)
        vx_history = np.zeros((n_save, nx_crop, nz_crop), dtype=np.float32) if return_vx else None
        vz_history = (
            np.zeros((n_save, nx_crop, nz_crop), dtype=np.float32) if return_vz else None
        )

    absorb_coeff = absorb(nx, nz, n_absorb)
    inject_scale = 1.0

    i0, i1 = n_absorb, n_absorb + nx_crop
    j0, j1 = n_absorb, n_absorb + nz_crop

    update_fn = update_velocity_stress_staggered_4th if order == 4 else update_velocity_stress_staggered
    save_idx = 0
    for it in range(nt):
        update_fn(p, vx, vz, vp_padded, rho, dt, dx, dz, nx, nz)

        _inject_record_interp(
            p, record_rev[it], xrec, zrec, dx, dz, n_absorb, inject_scale
        )

        p *= absorb_coeff
        vx *= absorb_coeff
        vz *= absorb_coeff

        if it % save_every == 0 and save_idx < n_save:
            if write_to_h5:
                writer.write_frame(
                    save_idx,
                    p[i0:i1, j0:j1],
                    vx[i0:i1, j0:j1] if return_vx else None,
                    vz[i0:i1, j0:j1] if return_vz else None,
                )
            else:
                p_history[save_idx] = p[i0:i1, j0:j1]
                if return_vx:
                    vx_history[save_idx] = vx[i0:i1, j0:j1]
                if return_vz:
                    vz_history[save_idx] = vz[i0:i1, j0:j1]
            save_idx += 1

        if progress_callback is not None:
            progress_callback(it + 1, nt)

    # Сохранить последний шаг (nt-1), если он ещё не сохранён
    if (nt - 1) % save_every != 0 and save_idx < n_save:
        if write_to_h5:
            writer.write_frame(
                save_idx,
                p[i0:i1, j0:j1],
                vx[i0:i1, j0:j1] if return_vx else None,
                vz[i0:i1, j0:j1] if return_vz else None,
            )
        else:
            p_history[save_idx] = p[i0:i1, j0:j1]
            if return_vx:
                vx_history[save_idx] = vx[i0:i1, j0:j1]
            if return_vz:
                vz_history[save_idx] = vz[i0:i1, j0:j1]
        save_idx += 1

    if write_to_h5:
        writer.close()
        return (snapshots_h5_path,)
    if save_idx < n_save:
        p_history = p_history[:save_idx]
        if return_vx:
            vx_history = vx_history[:save_idx]
        if return_vz:
            vz_history = vz_history[:save_idx]

    if return_vx:
        return (p_history, vx_history, vz_history)
    return (p_history, vz_history) if return_vz else (p_history, None)


def p_down_up_from_p_vz(p, vz, vp, rho=1.0):
    """
    Split total pressure into downgoing and upgoing waves using P and Vz (impedance method):
    p_down = 0.5 * (p + Z * vz),  p_up = 0.5 * (p - Z * vz),  Z = ρ * vp.
    """
    Z = np.asarray(rho, dtype=np.float32) * np.asarray(vp, dtype=np.float32)
    if np.isscalar(Z) or Z.ndim == 0:
        Z = np.full_like(p, float(Z))
    p_down = 0.5 * (p + Z * vz)
    p_up = 0.5 * (p - Z * vz)
    return p_down, p_up


def p_down_up_from_poynting(
    p, vx, vz, axis="z", eps=None, sigma_smooth=2.0, transition_scale=0.1
):
    """
    Split total pressure into downgoing and upgoing waves using the Poynting vector S = P·v.

    The energy‑flow direction is given by the sign of the vertical component S_z = P·vz:
    S_z > 0 → energy downward (incident), S_z < 0 → upward (reflected).
    Weights w_down and w_up are built from a smoothed sign(S), then
    p_down = P·w_down, p_up = P·w_up, so that p_down + p_up = P.

    To reduce high‑frequency artefacts we smooth S in space (sigma_smooth) and use
    a soft transition tanh(S/scale) instead of a hard sign (transition_scale).

    Parameters
    ----------
    p : (..., nx, nz) array
        Pressure.
    vx : array of same shape as p or None
        Horizontal particle velocity. For axis='z' this can be None (not used).
    vz : array of same shape as p
        Vertical particle velocity.
    axis : {'z', 'x'}
        Split along 'z' (up/down, S_z = P*vz) or along 'x' (S_x = P*vx).
    eps : float, optional
        Regularization used only when transition_scale=0. If None, chosen automatically.
    sigma_smooth : float or tuple, optional
        Spatial smoothing of S before taking the sign (in grid cells).
        For 2D: (sigma_x, sigma_z); for 3D: (0, sigma_x, sigma_z) — no smoothing in time.
        Single float → same smoothing in x and z (for 3D: (0, sigma, sigma)). Default 2.0.
    transition_scale : float, optional
        Soft transition: sign_smooth = tanh(S / scale), scale = transition_scale * percentile(|S|, 90).
        Larger values (0.3–0.5) → smoother transition and less high‑frequency noise. Default 0.25.
        Equal to 0 → hard sign (S/(|S|+eps)), which may produce artefacts.

    Returns
    -------
    p_down, p_up : array, same shape as p
        Downgoing and upgoing (or, in general, axis‑oriented) pressure components.
    """
    p = np.asarray(p, dtype=np.float64)
    vz = np.asarray(vz, dtype=np.float64)

    if axis == "z":
        S = p * vz  # vertical Poynting component (vx not needed here)
    elif axis == "x":
        if vx is None:
            raise ValueError("vx is required when axis='x'")
        vx = np.asarray(vx, dtype=np.float64)
        S = p * vx
    else:
        raise ValueError("axis must be 'z' or 'x'")

    # Smooth S in space (reduces high‑frequency noise in the weights)
    if sigma_smooth is not None and sigma_smooth > 0:
        sig = np.atleast_1d(sigma_smooth).astype(float)
        if sig.size == 1:
            sig = (float(sig[0]), float(sig[0]))
        else:
            sig = tuple(sig.flat)
        if S.ndim == 3:
            # (nt, nx, nz): smooth only over x and z
            if len(sig) == 2:
                sigma_apply = (0, sig[0], sig[1])
            else:
                sigma_apply = (sig[0], sig[1], sig[2])
        else:
            sigma_apply = (sig[0], sig[1]) if len(sig) >= 2 else (sig[0], sig[0])
        S = gaussian_filter(S, sigma=sigma_apply, mode="nearest")

    S_abs = np.abs(S)
    scale = np.percentile(S_abs, 90) + 1e-30

    if transition_scale is None or transition_scale <= 0:
        # Hard sign (legacy behaviour)
        if eps is None:
            eps = 1e-10 * (np.max(S_abs) + 1e-30)
        sign_smooth = S / (S_abs + eps)
    else:
        # Soft transition tanh(S / scale) — fewer high‑frequency artefacts
        scale *= transition_scale
        sign_smooth = np.tanh(S / scale)

    w_down = 0.5 * (1.0 + sign_smooth)
    w_up = 0.5 * (1.0 - sign_smooth)

    p_down = (p * w_down).astype(np.float32)
    p_up = (p * w_up).astype(np.float32)
    return p_down, p_up


def p_diff_from_poynting(
    p, vx, vz,
    sigma_smooth=2.0,
    theta0_deg=20,
    theta1_deg=60,
):
    """
    Split the field into diffracted (oblique propagation) and reflected (vertical)
    components by the angle of the Poynting vector S = (P·vx, P·vz).

    theta0_deg..theta1_deg specify the angular range (from vertical) where the
    diffraction weight smoothly increases from 0 to 1. The factor w_iso suppresses
    purely vertical energy flow (reflections).

    Returns
    -------
    p_diff, p_refl : array, same shape as p
        Diffracted and reflected pressure components.
    """
    p = np.asarray(p, dtype=np.float64)
    vx = np.asarray(vx, dtype=np.float64)
    vz = np.asarray(vz, dtype=np.float64)

    Sx = p * vx
    Sz = p * vz

    if sigma_smooth:
        sig = np.atleast_1d(sigma_smooth).astype(float)
        if sig.size == 1:
            sig = (float(sig[0]), float(sig[0]))
        else:
            sig = tuple(sig.flat)
        if Sx.ndim == 3:
            sigma_apply = (0, sig[0], sig[1]) if len(sig) == 2 else (0, sig[0], sig[1])
        else:
            sigma_apply = sig if len(sig) >= 2 else (sig[0], sig[0])
        Sx = gaussian_filter(Sx, sigma=sigma_apply, mode="nearest")
        Sz = gaussian_filter(Sz, sigma=sigma_apply, mode="nearest")

    theta = np.arctan2(np.abs(Sx), np.abs(Sz) + 1e-30)

    theta0 = np.deg2rad(theta0_deg)
    theta1 = np.deg2rad(theta1_deg)

    w_ang = np.clip((theta - theta0) / (theta1 - theta0), 0, 1)

    anisotropy = np.abs(Sz) / (np.abs(Sx) + np.abs(Sz) + 1e-30)
    w_iso = 1.0 - anisotropy

    w = w_ang * w_iso

    S_mag = np.sqrt(Sx**2 + Sz**2)
    w *= (S_mag > np.percentile(S_mag, 70))

    p_diff = (p * w).astype(np.float32)
    p_refl = (p * (1 - w)).astype(np.float32)

    return p_diff, p_refl


def scattering_angle_weight(
    p_fwd, vx_fwd, vz_fwd,
    p_rev, vx_rev, vz_rev,
    theta_min_deg=110,
):
    """
    Scattering‑angle weight between Poynting vectors of forward and reverse fields.

    cos(theta) = -(S_fwd · S_rev) / (|S_fwd| |S_rev|), theta = arccos(...).
    Large theta (close to π) → diffraction, small theta → reflection.
    w = clip((theta - theta_min) / (π - theta_min), 0, 1): w→1 for diffractions, w→0 for reflectors.

    Parameters
    ----------
    p_fwd, vx_fwd, vz_fwd : array (nt, nx, nz)
        Pressure and velocities of the forward field (already P_down).
    p_rev, vx_rev, vz_rev : array (nt, nx, nz)
        Pressure and velocities of the reverse field, time‑aligned with p_fwd
        (i.e. p_rev[t] corresponds to the same time as p_fwd[t]).
    theta_min_deg : float
        Minimum angle (degrees); below it weight is 0 (reflector), above it smoothly
        increases towards 1 (diffraction).

    Returns
    -------
    w : (nt, nx, nz) float32
        Diffraction weight (0 = reflector, 1 = diffractor).
    """
    p_fwd = np.asarray(p_fwd, dtype=np.float64)
    vx_fwd = np.asarray(vx_fwd, dtype=np.float64)
    vz_fwd = np.asarray(vz_fwd, dtype=np.float64)
    p_rev = np.asarray(p_rev, dtype=np.float64)
    vx_rev = np.asarray(vx_rev, dtype=np.float64)
    vz_rev = np.asarray(vz_rev, dtype=np.float64)

    Sfx = p_fwd * vx_fwd
    Sfz = p_fwd * vz_fwd
    Sbx = p_rev * vx_rev
    Sbz = p_rev * vz_rev

    num = -(Sfx * Sbx + Sfz * Sbz)
    den = np.sqrt(Sfx ** 2 + Sfz ** 2) * np.sqrt(Sbx ** 2 + Sbz ** 2) + 1e-30
    cos_theta = np.clip(num / den, -1, 1)

    theta = np.arccos(cos_theta)
    theta_min = np.deg2rad(theta_min_deg)

    w = np.clip((theta - theta_min) / (np.pi - theta_min), 0, 1)
    return w.astype(np.float32)
