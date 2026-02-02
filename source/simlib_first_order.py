"""
Симуляция акустики 1-го порядка (система скорость–давление).
Схема Вирьё (Staggered Grid): 2-й порядок по времени, 2-й по пространству.

Уравнения:
  ∂vx/∂t = - (1/ρ) ∂P/∂x
  ∂vz/∂t = - (1/ρ) ∂P/∂z
  ∂P/∂t  = - ρ c² (∂vx/∂x + ∂vz/∂z)
"""

from numba import jit
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


def ricker(f_hz, nt, dt, t0=None):
    """Вейвлет Рикера; f_hz — доминантная частота (Гц), nt — число отсчётов, dt — шаг по времени (с)."""
    t = np.arange(nt, dtype=np.float32) * dt
    if t0 is None:
        t0 = 1.0 / f_hz
    s = (1 - 2 * (np.pi * f_hz * (t - t0)) ** 2) * np.exp(-(np.pi * f_hz * (t - t0)) ** 2)
    return s.astype(np.float32)


def prepare_migration_velocity(vp_true, sigma_meters, dx, dz):
    """
    Сглаживает скоростную модель 2D gaussian filter.
    vp_true: (nx, nz). sigma_meters: радиус сглаживания в метрах.
    Возвращает vp_mig той же формы (nx, nz).
    """
    sigma_x = sigma_meters / dx
    sigma_z = sigma_meters / dz
    vp_mig = gaussian_filter(vp_true, sigma=[sigma_x, sigma_z], mode="nearest")
    return vp_mig


def _inject_record_interp(p_it, record_vals, xrec, zrec, dx, dz, n_absorb, scale):
    """
    Добавляет к полю давления p_it запись с позиций приёмников (xrec, zrec).
    При коллинеарных точках — инжекция в ближайшие узлы; иначе — griddata (linear).
    scale: множитель (для 1-го порядка используем dt).
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
    Обновление полей на разнесённой сетке (Staggered Grid), 2-й порядок по пространству.
    P хранится в [i, j],
    Vx — в [i+0.5, j] (сдвиг по X), обновляется для i = 0..nx-2,
    Vz — в [i, j+0.5] (сдвиг по Z), обновляется для j = 0..nz-2.
    Массивы p, vx, vz имеют форму (nx, nz); вне зоны обновления vx/vz не трогаем.
    """
    # 1. Обновление скоростей (Vx, Vz) по давлению P
    # Vx: dP/dx = (P[i+1] - P[i]) / dx  (2-й порядок)
    for i in range(nx - 1):
        for j in range(nz):
            rho_x = 0.5 * (rho[i + 1, j] + rho[i, j])
            vx[i, j] = vx[i, j] - (dt / rho_x) * (p[i + 1, j] - p[i, j]) / dx

    # Vz: dP/dz = (P[i,j+1] - P[i,j]) / dz
    for i in range(nx):
        for j in range(nz - 1):
            rho_z = 0.5 * (rho[i, j + 1] + rho[i, j])
            vz[i, j] = vz[i, j] - (dt / rho_z) * (p[i, j + 1] - p[i, j]) / dz

    # 2. Обновление давления P по дивергенции скорости
    for i in range(1, nx - 1):
        for j in range(1, nz - 1):
            k_modulus = rho[i, j] * vp[i, j] ** 2
            div_v = (vx[i, j] - vx[i - 1, j]) / dx + (vz[i, j] - vz[i, j - 1]) / dz
            p[i, j] = p[i, j] - dt * k_modulus * div_v

    return p, vx, vz


@jit(nopython=True)
def update_velocity_stress_staggered_4th(p, vx, vz, vp, rho, dt, dx, dz, nx, nz):
    """
    То же, но 4-й порядок по пространству в интерьере — меньше численной дисперсии.
    Стенсил первой производной на полуцелых узлах:
    f'_{i+1/2} = (9(f_{i+1}-f_i) - (f_{i+2}-f_{i-1})) / (8*dx).
    В приграничной полосе (1 точка с краёв) — 2-й порядок, чтобы вся сетка обновлялась.
    """
    idx = 1.0 / dx
    idz = 1.0 / dz
    c_x  =  9.0 /  8.0 * idx      # это остаётся как было — правильно
    c_x2 = -1.0 / 24.0 * idx      # ← здесь главное изменение: 8 → 24
    c_z  =  9.0 /  8.0 * idz      # остаётся
    c_z2 = -1.0 / 24.0 * idz

    # 1. Vx: в интерьере i=1..nx-3 — 4-й порядок; i=0 и i=nx-2 — 2-й порядок
    for i in range(nx - 1):
        for j in range(nz):
            rho_x = 0.5 * (rho[i + 1, j] + rho[i, j])
            if 1 <= i <= nx - 3:
                dp_dx = c_x * (p[i + 1, j] - p[i, j]) + c_x2 * (p[i + 2, j] - p[i - 1, j])
            else:
                dp_dx = (p[i + 1, j] - p[i, j]) * idx
            vx[i, j] = vx[i, j] - (dt / rho_x) * dp_dx

    # Vz: в интерьере j=1..nz-3 — 4-й порядок; края — 2-й порядок
    for i in range(nx):
        for j in range(nz - 1):
            rho_z = 0.5 * (rho[i, j + 1] + rho[i, j])
            if 1 <= j <= nz - 3:
                dp_dz = c_z * (p[i, j + 1] - p[i, j]) + c_z2 * (p[i, j + 2] - p[i, j - 1])
            else:
                dp_dz = (p[i, j + 1] - p[i, j]) * idz
            vz[i, j] = vz[i, j] - (dt / rho_z) * dp_dz

    # 2. P: в интерьере (2..nx-3, 2..nz-3) — 4-й порядок; остальное — 2-й порядок
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
    """Коэффициенты поглощающего слоя (как в simlib)."""
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
):
    """
    Прямое моделирование 2D акустики в форме 1-го порядка (скорость–давление),
    схема Вирьё (staggered grid).

    Параметры
    ---------
    src : (nt,) array
        Временная функция источника.
    vp : (nx, nz) array
        Скорость P-волны [м/с].
    nt, dt, dx, dz : int, float
        Число шагов по времени, шаг по времени и по осям.
    xsrc, zsrc : float
        Координаты источника в метрах.
    rho : (nx, nz) array или float, optional
        Плотность. Если None, используется 1.0.
    n_absorb : int
        Толщина поглощающего слоя в узлах.
    save_every : int
        Сохранять снимки полей каждые save_every шагов (1 = каждый шаг).
    return_vz : bool
        Возвращать ли историю vz (нужно для P-up/P-down разделения).
    return_vx : bool
        Возвращать ли историю vx (нужно для p_diff_from_poynting). При True возвращается (p, vx, vz).
    order : int, 2 или 4
        Порядок аппроксимации по пространству. По умолчанию 4 (меньше численной дисперсии).
    progress_callback : callable, optional
        Если задан, вызывается как progress_callback(current, total) на каждой итерации по времени
        (для прогресс-бара в GUI, например PyQt5).

    Возвращает
    ----------
    При return_vx=True: (p_cropped, vx_cropped, vz_cropped).
    Иначе при return_vz=True: (p_cropped, vz_cropped). Иначе: (p_cropped, None).
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
    n_save = (nt + save_every - 1) // save_every
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

        # Источник в давление (взрыв)
        p[isrc, jsrc] += src[it] * dt

        p *= absorb_coeff
        vx *= absorb_coeff
        vz *= absorb_coeff

        if it % save_every == 0 and save_idx < n_save:
            p_history[save_idx] = p[i0:i1, j0:j1]
            if return_vx:
                vx_history[save_idx] = vx[i0:i1, j0:j1]
            if return_vz:
                vz_history[save_idx] = vz[i0:i1, j0:j1]
            save_idx += 1

        if progress_callback is not None:
            progress_callback(it + 1, nt)

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
):
    """
    Обратное распространение для RTM: инжекция записей в обратном времени,
    та же схема 1-го порядка (скорость–давление), что и в fd2d_forward_first_order.

    Параметры
    ---------
    record : (nt, nrec) array
        Записи давления на приёмниках (forward time). nrec — число приёмников.
    vp : (nx, nz) array
        Скорость P-волны [м/с].
    nt, dt, dx, dz : int, float
        Число шагов по времени и шаги сетки.
    xrec, zrec : (nrec,) array
        Координаты приёмников в метрах.
    rho : (nx, nz) array или float, optional
        Плотность. Если None — 1.0.
    n_absorb : int
        Толщина поглощающего слоя.
    save_every : int
        Сохранять снимки каждые save_every шагов.
    return_vz : bool
        Возвращать ли историю vz.
    return_vx : bool
        Возвращать ли историю vx (для p_diff_from_poynting). При True возвращается (p, vx, vz).
    order : int, 2 или 4
        Порядок по пространству. По умолчанию 4 (должен совпадать с прямым прогоном для RTM).
    progress_callback : callable, optional
        Если задан, вызывается как progress_callback(current, total) на каждой итерации.

    Возвращает
    ----------
    При return_vx=True: (p_cropped, vx_cropped, vz_cropped).
    Иначе при return_vz=True: (p_cropped, vz_cropped). Иначе: (p_cropped, None).
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
    n_save = (nt + save_every - 1) // save_every
    p_history = np.zeros((n_save, nx_crop, nz_crop), dtype=np.float32)
    vx_history = np.zeros((n_save, nx_crop, nz_crop), dtype=np.float32) if return_vx else None
    vz_history = (
        np.zeros((n_save, nx_crop, nz_crop), dtype=np.float32) if return_vz else None
    )

    absorb_coeff = absorb(nx, nz, n_absorb)
    inject_scale = float(dt)

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
            p_history[save_idx] = p[i0:i1, j0:j1]
            if return_vx:
                vx_history[save_idx] = vx[i0:i1, j0:j1]
            if return_vz:
                vz_history[save_idx] = vz[i0:i1, j0:j1]
            save_idx += 1

        if progress_callback is not None:
            progress_callback(it + 1, nt)

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
    Разделение на нисходящую и восходящую волны по полям P и Vz (импедансный способ):
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
    Разделение на нисходящую и восходящую волны по вектору Пойнтинга S = P·v.

    Направление потока энергии задаётся знаком вертикальной компоненты S_z = P·vz:
    S_z > 0 — энергия вниз (падающая), S_z < 0 — вверх (отражённая).
    Веса w_down и w_up строятся по сглаженному sign(S), затем p_down = P·w_down,
    p_up = P·w_up, так что p_down + p_up = P.

    Для уменьшения высокочастотных артефактов: сглаживание S по пространству
    (sigma_smooth) и мягкий переход tanh(S/scale) вместо резкого sign (transition_scale).

    Параметры
    ---------
    p : (..., nx, nz) array
        Давление.
    vx : array той же формы или None
        Горизонтальная скорость. Для axis='z' можно передать None (не используется).
    vz : array той же формы
        Вертикальная скорость частиц.
    axis : str, 'z' или 'x'
        Ось для разделения: 'z' — вверх/вниз (S_z = P*vz), 'x' — по горизонтали (S_x = P*vx).
    eps : float, optional
        Регуляризация (используется только при transition_scale=0). Если None — авто.
    sigma_smooth : float или tuple, optional
        Сглаживание S по пространству перед знаком (в узлах сетки).
        Для 2D: (sigma_x, sigma_z); для 3D: (0, sigma_x, sigma_z) — по времени не сглаживаем.
        Один float — то же по x и z (для 3D: (0, sigma, sigma)). По умолчанию 2.0.
    transition_scale : float, optional
        Мягкий переход: sign_smooth = tanh(S / scale), scale = transition_scale * percentile(|S|, 90).
        Больше значение (0.3–0.5) — плавнее переход, меньше высокочастотной грязи. По умолчанию 0.25.
        Равен 0 — жёсткий sign (S/(|S|+eps)), возможны артефакты.

    Возвращает
    ----------
    p_down, p_up : array той же формы, что p
        Падающая и восходящая (или по axis) компоненты давления.
    """
    p = np.asarray(p, dtype=np.float64)
    vz = np.asarray(vz, dtype=np.float64)

    if axis == "z":
        S = p * vz  # вертикальная компонента Пойнтинга (vx не нужен)
    elif axis == "x":
        if vx is None:
            raise ValueError("для axis='x' нужен vx")
        vx = np.asarray(vx, dtype=np.float64)
        S = p * vx
    else:
        raise ValueError("axis должен быть 'z' или 'x'")

    # Сглаживание S по пространству (убирает высокочастотную грязь в весах)
    if sigma_smooth is not None and sigma_smooth > 0:
        sig = np.atleast_1d(sigma_smooth).astype(float)
        if sig.size == 1:
            sig = (float(sig[0]), float(sig[0]))
        else:
            sig = tuple(sig.flat)
        if S.ndim == 3:
            # (nt, nx, nz): сглаживаем только по x, z
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
        # Жёсткий знак (как раньше)
        if eps is None:
            eps = 1e-10 * (np.max(S_abs) + 1e-30)
        sign_smooth = S / (S_abs + eps)
    else:
        # Мягкий переход tanh(S / scale) — меньше высокочастотных артефактов
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
    Разделение поля на дифракционную (наклонное распространение) и отражённую
    (вертикальное) составляющие по углу вектора Пойнтинга S = (P·vx, P·vz).

    theta0_deg..theta1_deg — диапазон углов (от вертикали), в котором вес дифракций
    плавно растёт от 0 до 1. w_iso ослабляет вертикальный поток (отражения).

    Возвращает
    ----------
    p_diff, p_refl : array той же формы, что p
        Дифракционная и отражённая компоненты давления.
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
    Вес по углу рассеяния между векторами Пойнтинга прямого и обратного полей.

    cos(theta) = -(S_fwd · S_rev) / (|S_fwd| |S_rev|), theta = arccos(...).
    Большой theta (близко к π) — дифракция, малый theta — отражение.
    w = clip((theta - theta_min) / (π - theta_min), 0, 1): w→1 для дифракций, w→0 для рефлекторов.

    Параметры
    ---------
    p_fwd, vx_fwd, vz_fwd : array (nt, nx, nz)
        Давление и скорости прямого (forward) поля (уже P_down).
    p_rev, vx_rev, vz_rev : array (nt, nx, nz)
        Давление и скорости обратного (reverse) поля, выровненного по времени с p_fwd
        (т.е. p_rev[t] соответствует тому же моменту времени, что и p_fwd[t]).
    theta_min_deg : float
        Минимальный угол (градусы); ниже — вес 0 (рефлектор), выше — плавный переход к 1 (дифракция).

    Возвращает
    ----------
    w : (nt, nx, nz) float32
        Вес дифракции (0 = рефлектор, 1 = дифрактор).
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
