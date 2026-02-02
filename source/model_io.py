"""
Загрузка скоростной модели из SEG-Y для GUI.
dz = f.bin[segyio.BinField.Interval] / 1000000
dx = шаг по X из атрибутов CDP_X (медиана разностей между уникальными X).
"""
import numpy as np
import segyio


def load_velocity_from_segy(path):
    """
    Загружает скоростную модель из SEG-Y.
    path: путь к .sgy файлу.

    dz = f.bin[segyio.BinField.Interval] / 1000000
    dx = медиана разностей между соседними уникальными CDP_X (шаг сетки по X).

    Возвращает (vp, dx, dz) или (None, None, None) при ошибке.
    vp: (nz, nx), float64; ось 0 — глубина Z, ось 1 — X.
    """
    try:
        with segyio.open(path, "r", ignore_geometry=True) as f:
            # dz из бинарного заголовка (интервал сэмпла), в м после деления на 1e6
            interval = f.bin[segyio.BinField.Interval]
            dz = float(interval) / 1_000.0

            # dx — шаг по X: медиана разностей между уникальными CDP_X
            # f.attributes возвращает генератор — приводим к списку
            cdpx = np.array(list(f.attributes(segyio.TraceField.CDP_X)), dtype=np.float64)
            ux = np.unique(np.sort(cdpx))
            if len(ux) > 1:
                diffs = np.diff(ux)
                dx = float(np.median(diffs))
            else:
                dx = float(np.median(cdpx)) if len(cdpx) > 0 else 1.0

            # Данные: трассы — столбцы по X, сэмплы — по глубине Z
            # f.trace[:] в segyio — генератор, приводим к списку
            raw = np.asarray(list(f.trace[:]), dtype=np.float64)
            vp = raw.T  # (n_samples, n_traces) = (nz, nx)
        return vp, dx, dz
    except Exception as e:
        raise RuntimeError("Ошибка загрузки SEG-Y: {}: {}".format(type(e).__name__, e)) from e
