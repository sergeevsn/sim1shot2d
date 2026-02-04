# -*- coding: utf-8 -*-
"""
Точка входа GUI: PyQt5, меню File, Model, Survey, Simulation.
"""
import sys
import os
import gc
import numpy as np
from configparser import ConfigParser
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import resample as scipy_resample
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QFileDialog,
    QMenu, QAction, QDialog, QFormLayout, QDoubleSpinBox, QSpinBox, QPushButton,
    QDialogButtonBox, QListWidget, QListWidgetItem, QGroupBox, QRadioButton,
    QHBoxLayout, QMessageBox, QScrollArea, QFrame, QSizePolicy, QComboBox,
    QCheckBox, QSlider, QGridLayout, QProgressBar, QLineEdit, QStackedWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Импорт из source: каталог с main.py в path, чтобы работало и из gui/, и из корня репо
import re
_gui_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_gui_dir)
if _gui_dir not in sys.path:
    sys.path.insert(0, _gui_dir)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)
import segyio
from source.model_io import load_velocity_from_segy
from source import snapshot_io
from source.simlib_first_order import (
    ricker,
    fd2d_forward_first_order,
    fd2d_backward_first_order,
    prepare_migration_velocity,
)

# Порядок компонентов снапшотов в комбобоксе
SNAPSHOT_COMPONENT_ORDER = ["P fwd", "Vz fwd", "Vx fwd", "P bwd", "Vz bwd", "Vx bwd"]

try:
    import psutil
except ImportError:
    psutil = None


def _get_process_memory_mb():
    """Память процесса (RSS) в МБ. Возвращает None, если psutil недоступен."""
    if psutil is None:
        return None
    try:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _get_snapshots_h5_path(base_dir, run_name, run_type="forward", project_stem=None):
    """Путь к HDF5 снапшотов: base_dir/<project_stem>_files/<fwd|bwd>_<sanitized_name>.h5"""
    safe = re.sub(r"[^\w\-]", "_", (run_name or "run").strip()) or "run"
    prefix = "fwd" if run_type == "forward" else "bwd"
    stem = (project_stem or "unsaved").strip() or "unsaved"
    snap_dir = os.path.join(base_dir, "{}_files".format(stem))
    os.makedirs(snap_dir, exist_ok=True)
    return os.path.join(snap_dir, "{}_{}.h5".format(prefix, safe))


def _get_seismogram_npz_path(snapshots_h5_path):
    """Путь к NPZ сейсмограмм: рядом с H5, имя <base>.h5 -> <base>_seismogram.npz"""
    d = os.path.dirname(snapshots_h5_path)
    base = os.path.splitext(os.path.basename(snapshots_h5_path))[0]
    return os.path.join(d, "{}_seismogram.npz".format(base))


def _get_system_memory_limit_mb(percent=0.85):
    """Рекомендуемый лимит памяти (percent от общей системной) в МБ. None если psutil недоступен."""
    if psutil is None:
        return None
    try:
        vm = psutil.virtual_memory()
        return (vm.total * percent) / (1024 * 1024)
    except Exception:
        return None


def _laplacian_filter_2d(img, order, dx=1.0, dz=1.0):
    """Лапласиан 2D: order 2 (scipy) или 4 (стецил 4-го порядка). Edge-padding, затем crop результата."""
    img = np.asarray(img, dtype=np.float64)
    pad = 1 if order == 2 else 2
    padded = np.pad(img, ((pad, pad), (pad, pad)), mode="edge")
    nz_p, nx_p = padded.shape
    if order == 2:
        lap_p = ndimage.laplace(padded, mode="edge")
        lap = lap_p[pad : pad + img.shape[0], pad : pad + img.shape[1]]
        return -lap
    c_x = 1.0 / (12.0 * dx * dx)
    c_z = 1.0 / (12.0 * dz * dz)
    lap_p = np.zeros_like(padded)
    for i in range(2, nx_p - 2):
        lap_p[:, i] = c_x * (
            -padded[:, i - 2] + 16 * padded[:, i - 1] - 30 * padded[:, i]
            + 16 * padded[:, i + 1] - padded[:, i + 2]
        )
    for j in range(2, nz_p - 2):
        lap_p[j, :] += c_z * (
            -padded[j - 2, :] + 16 * padded[j - 1, :] - 30 * padded[j, :]
            + 16 * padded[j + 1, :] - padded[j + 2, :]
        )
    lap = lap_p[pad : pad + img.shape[0], pad : pad + img.shape[1]].copy()
    return -lap


def _agc_along_z(img, dz, window_m, eps=1e-12):
    """AGC по оси Z: для каждого столбца нормализация по скользящему окну (RMS). img (nz, nx)."""
    img = np.asarray(img, dtype=np.float64)
    nz, nx = img.shape
    w = max(1, min(nz, int(round(window_m / dz))))
    out = np.zeros_like(img)
    for ix in range(nx):
        trace = img[:, ix].copy()
        for iz in range(nz):
            start = max(0, iz - w // 2)
            end = min(nz, iz + w // 2 + 1)
            win = trace[start:end]
            rms = np.sqrt(np.mean(win * win) + eps)
            out[iz, ix] = trace[iz] / rms
    return out


class VelocityCanvas(FigureCanvas):
    """Холст: слои Z-X (модель, сглаженная, снапшоты P/Vz/Vx, съёмка) с alpha."""
    def __init__(self, parent=None):
        self._fig = Figure(figsize=(6, 4), facecolor="#f0f0f0")
        super().__init__(self._fig)
        if parent is not None:
            self.setParent(parent)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._fig.subplots_adjust(left=0.14, right=0.92, top=0.95, bottom=0.1)
        self._ax = self._fig.add_subplot(111)
        self._cbar = None
        self._vp = None
        self._dx = self._dz = 1.0
        self._source = None
        self._receivers = []
        self._diffractors = []
        self._smoothed_vp = None
        self._snapshot_2d = None
        self._rtm_image = None
        self._layer = {
            "show_original": True, "alpha_original": 1.0,
            "show_survey": True, "alpha_survey": 1.0,
            "show_smoothed": False, "alpha_smoothed": 1.0,
            "show_snapshots": False, "alpha_snapshots": 0.5,
            "snapshot_vmin": None, "snapshot_vmax": None,
            "show_image": False, "alpha_image": 0.8,
            "image_vmin": None, "image_vmax": None,
        }

    def set_velocity(self, vp, dx, dz):
        self._vp = np.asarray(vp, dtype=np.float64) if vp is not None else None
        self._dx = float(dx) if dx is not None else 1.0
        self._dz = float(dz) if dz is not None else 1.0
        self._redraw()

    def set_source(self, x, z, freq=None):
        self._source = (float(x), float(z), freq) if x is not None and z is not None else None
        self._redraw()

    def set_receivers(self, points):
        self._receivers = list(points) if points else []
        self._redraw()

    def set_diffractors(self, diffractors):
        self._diffractors = list(diffractors) if diffractors else []
        self._redraw()

    def set_smoothed_vp(self, vp):
        self._smoothed_vp = np.asarray(vp, dtype=np.float64) if vp is not None else None
        self._redraw()

    def set_snapshot_2d(self, data):
        self._snapshot_2d = np.asarray(data, dtype=np.float64) if data is not None else None
        self._redraw()

    def set_rtm_image(self, data):
        self._rtm_image = np.asarray(data, dtype=np.float64) if data is not None else None
        self._redraw()

    def set_layer(self, **kwargs):
        self._layer.update(kwargs)
        self._redraw()

    def _redraw(self):
        if self._cbar is not None:
            try:
                self._cbar.remove()
            except Exception:
                pass
            self._cbar = None
        self._ax.clear()
        if self._vp is None or self._vp.size == 0:
            self._ax.text(0.5, 0.5, "Model not loaded", ha="center", va="center", transform=self._ax.transAxes)
            self.draw()
            return
        nz, nx = self._vp.shape
        dx, dz = self._dx, self._dz
        extent = [0, nx * dx, nz * dz, 0]
        lay = self._layer
        im = None

        if lay["show_original"]:
            im = self._ax.imshow(
                self._vp,
                cmap="seismic",
                aspect="auto",
                extent=extent,
                interpolation="nearest",
                origin="upper",
                alpha=lay["alpha_original"],
            )
        if lay["show_smoothed"] and self._smoothed_vp is not None and self._smoothed_vp.shape == self._vp.shape:
            self._ax.imshow(
                self._smoothed_vp,
                cmap="seismic",
                aspect="auto",
                extent=extent,
                interpolation="nearest",
                origin="upper",
                alpha=lay["alpha_smoothed"],
            )
        if lay["show_snapshots"] and self._snapshot_2d is not None and self._snapshot_2d.shape == self._vp.shape:
            s = self._snapshot_2d
            vmin = lay.get("snapshot_vmin")
            vmax = lay.get("snapshot_vmax")
            if vmin is None or vmax is None:
                v = max(np.abs(s.min()), np.abs(s.max())) or 1.0
                vmin, vmax = -v, v
            self._ax.imshow(
                s,
                cmap="gray",
                aspect="auto",
                extent=extent,
                interpolation="nearest",
                origin="upper",
                alpha=lay["alpha_snapshots"],
                vmin=vmin,
                vmax=vmax,
            )
        if lay["show_image"] and self._rtm_image is not None and self._rtm_image.shape == self._vp.shape:
            img = self._rtm_image
            vmin = lay.get("image_vmin")
            vmax = lay.get("image_vmax")
            if vmin is None or vmax is None:
                v = max(np.abs(img.min()), np.abs(img.max())) or 1.0
                vmin, vmax = -v, v
            self._ax.imshow(
                img,
                cmap="gray",
                aspect="auto",
                extent=extent,
                interpolation="nearest",
                origin="upper",
                alpha=lay["alpha_image"],
                vmin=vmin,
                vmax=vmax,
            )
        if im is None:
            self._ax.imshow(
                self._vp,
                cmap="seismic",
                aspect="auto",
                extent=extent,
                interpolation="nearest",
                origin="upper",
                alpha=0.0,
            )
        im = self._ax.images[0] if self._ax.images else None
        self._ax.set_xlabel("X, m")
        self._ax.set_ylabel("Z, m")
        self._ax.tick_params(axis="both", which="major", labelsize=8)
        if im is not None:
            self._cbar = self._fig.colorbar(im, ax=self._ax, label="V, m/s", shrink=0.8)

        if lay["show_survey"]:
            alpha = lay["alpha_survey"]
            for (rx, rz) in self._receivers:
                self._ax.text(rx, rz, "x", color="lime", fontsize=10, ha="center", va="center", fontweight="bold", alpha=alpha)
            if self._source is not None:
                sx, sz, _ = self._source
                self._ax.text(sx, sz, "v", color="red", fontsize=10, ha="center", va="center", fontweight="bold", alpha=alpha)

        self._ax.autoscale(False)
        self._ax.set_xlim(0, nx * dx)
        self._ax.set_ylim(nz * dz, 0)
        self.draw()


class SeismogramCanvas(FigureCanvas):
    """Холст сейсмограммы: оси X/Y зависят от Profile (X, Time) или Well (Time, Z)."""
    def __init__(self, parent=None):
        self._fig = Figure(figsize=(4, 4), facecolor="#f0f0f0")
        super().__init__(self._fig)
        if parent is not None:
            self.setParent(parent)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._ax = self._fig.add_subplot(111)
        self._fig.subplots_adjust(left=0.14, right=0.95, top=0.95, bottom=0.1)
        self._layout = "Profile"
        self._data = None       # (n_times, n_receivers) для отображения
        self._t_ms = None       # 1D, мс
        self._receivers = []    # список (x, z)
        self._percentile = 98   # для vmin, vmax = np.percentile(data, [100-p, p])
        self._redraw()

    def set_layout(self, layout):
        """layout: 'Profile' или 'Well'."""
        self._layout = layout if layout in ("Profile", "Well") else "Profile"
        self._redraw()

    def set_seismogram(self, data, t_ms, receivers, layout, dx, dz):
        """data: (n_times, n_receivers); t_ms: массив времени в мс; receivers: список (x, z)."""
        self._data = np.asarray(data, dtype=np.float64) if data is not None else None
        self._t_ms = np.asarray(t_ms, dtype=np.float64) if t_ms is not None else None
        self._receivers = list(receivers) if receivers else []
        self._layout = layout if layout in ("Profile", "Well") else "Profile"
        self._redraw()

    def set_percentile(self, p):
        """Устанавливает перцентиль для vmin, vmax: np.percentile(data, [100-p, p])."""
        self._percentile = max(0.5, min(99.5, float(p)))
        self._redraw()

    def _redraw(self):
        self._ax.clear()
        self._fig.subplots_adjust(left=0.14, right=0.95, top=0.95, bottom=0.1)
        if self._layout == "Profile":
            self._ax.set_xlabel("X, m")
            self._ax.set_ylabel("Time, ms")
        else:
            self._ax.set_xlabel("Time, ms")
            self._ax.set_ylabel("Z, m")
        if self._data is not None and self._data.size > 0 and self._t_ms is not None and len(self._receivers) > 0:
            p = self._percentile
            vmin, vmax = np.percentile(self._data, [100 - p, p])
            if vmax <= vmin:
                vmax = vmin + 1.0
            n_t, n_rec = self._data.shape
            if self._layout == "Profile":
                xs = np.array([r[0] for r in self._receivers])
                x_min, x_max = xs.min(), xs.max()
                if x_max <= x_min:
                    x_max = x_min + 1.0
                t_max = float(self._t_ms[-1]) if len(self._t_ms) > 0 else 0.0
                extent = [x_min, x_max, t_max, 0.0]
                self._ax.imshow(
                    self._data,
                    aspect="auto",
                    extent=extent,
                    interpolation="nearest",
                    origin="upper",
                    cmap="gray",
                    vmin=vmin,
                    vmax=vmax,
                )
            else:
                zs = np.array([r[1] for r in self._receivers])
                z_min, z_max = zs.min(), zs.max()
                if z_max <= z_min:
                    z_max = z_min + 1.0
                t_max = float(self._t_ms[-1]) if len(self._t_ms) > 0 else 0.0
                extent = [0.0, t_max, z_max, z_min]
                self._ax.imshow(
                    self._data.T,
                    aspect="auto",
                    extent=extent,
                    interpolation="nearest",
                    origin="upper",
                    cmap="gray",
                    vmin=vmin,
                    vmax=vmax,
                )
        else:
            self._ax.text(0.5, 0.5, "Seismogram", ha="center", va="center", transform=self._ax.transAxes, fontsize=12)
        self._ax.tick_params(axis="both", which="major", labelsize=8)
        self.draw()


def _render_snapshot_frame_agg(vp, dx, dz, receivers, source, smoothed_vp, snapshot_2d,
                                include_overlays, alpha_original, alpha_survey, alpha_smoothed, alpha_snapshots,
                                snapshot_vmin, snapshot_vmax, time_ms=0.0, component_label="P",
                                figsize=(6, 4), dpi=100):
    """
    Рисует один кадр Z-X (модель + опциональные оверлеи + снапшот), без RTM.
    Возвращает RGB array (H, W, 3) uint8 для записи в GIF.
    """
    fig = Figure(figsize=figsize, dpi=dpi, facecolor="#f0f0f0")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.14, right=0.95, top=0.92, bottom=0.1)
    nz, nx = vp.shape
    extent = [0, nx * dx, nz * dz, 0]
    im = None
    if include_overlays:
        im = ax.imshow(vp, cmap="seismic", aspect="auto", extent=extent, interpolation="nearest",
                       origin="upper", alpha=alpha_original)
        if smoothed_vp is not None and smoothed_vp.shape == vp.shape:
            ax.imshow(smoothed_vp, cmap="seismic", aspect="auto", extent=extent,
                      interpolation="nearest", origin="upper", alpha=alpha_smoothed)
    if snapshot_2d is not None and snapshot_2d.shape == vp.shape:
        s = np.asarray(snapshot_2d, dtype=np.float64)
        vmin = snapshot_vmin
        vmax = snapshot_vmax
        if vmin is None or vmax is None:
            v = max(np.abs(s.min()), np.abs(s.max())) or 1.0
            vmin, vmax = -v, v
        ax.imshow(s, cmap="gray", aspect="auto", extent=extent, interpolation="nearest",
                  origin="upper", alpha=alpha_snapshots, vmin=vmin, vmax=vmax)
    if im is None:
        im = ax.imshow(vp, cmap="seismic", aspect="auto", extent=extent,
                       interpolation="nearest", origin="upper", alpha=0.0)
    ax.set_title("t = {} ms, {}".format(int(round(time_ms)), component_label), fontsize=10)
    ax.set_xlabel("X, m")
    ax.set_ylabel("Z, m")
    ax.tick_params(axis="both", which="major", labelsize=8)
    if include_overlays and alpha_survey > 0:
        for (rx, rz) in (receivers or []):
            ax.text(rx, rz, "x", color="lime", fontsize=10, ha="center", va="center", fontweight="bold", alpha=alpha_survey)
        if source is not None:
            sx, sz = source[0], source[1]
            ax.text(sx, sz, "v", color="red", fontsize=10, ha="center", va="center", fontweight="bold", alpha=alpha_survey)
    ax.autoscale(False)
    ax.set_xlim(0, nx * dx)
    ax.set_ylim(nz * dz, 0)
    canvas.draw()
    w, h = canvas.get_width_height()
    # matplotlib 3.6+ убрал tostring_rgb(); используем buffer_rgba() и берём RGB
    buf = np.asarray(canvas.buffer_rgba()).reshape((h, w, 4))
    rgb = buf[:, :, :3].copy()
    return rgb


class _SnapshotGifRendererCtx(object):
    """
    Контекст для быстрого экспорта GIF: один раз создаётся figure/canvas,
    дальше обновляются только данные снапшота и заголовок — без пересоздания фигуры.
    """
    def __init__(self, vp, dx, dz, receivers, source, smoothed_vp,
                 include_overlays, alpha_original, alpha_survey, alpha_smoothed, alpha_snapshots,
                 snapshot_vmin, snapshot_vmax, figsize=(6, 4), dpi=100):
        nz, nx = vp.shape
        extent = [0, nx * dx, nz * dz, 0]
        self.fig = Figure(figsize=figsize, dpi=dpi, facecolor="#f0f0f0")
        self.canvas = FigureCanvasAgg(self.fig)
        ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.14, right=0.95, top=0.92, bottom=0.1)
        if include_overlays:
            ax.imshow(vp, cmap="seismic", aspect="auto", extent=extent, interpolation="nearest",
                     origin="upper", alpha=alpha_original)
            if smoothed_vp is not None and smoothed_vp.shape == vp.shape:
                ax.imshow(smoothed_vp, cmap="seismic", aspect="auto", extent=extent,
                          interpolation="nearest", origin="upper", alpha=alpha_smoothed)
        # Слой снапшота — единственный, который будем обновлять
        dummy = np.zeros_like(vp, dtype=np.float64)
        self.snapshot_im = ax.imshow(
            dummy, cmap="gray", aspect="auto", extent=extent, interpolation="nearest",
            origin="upper", alpha=alpha_snapshots, vmin=snapshot_vmin, vmax=snapshot_vmax
        )
        if not include_overlays:
            ax.imshow(vp, cmap="seismic", aspect="auto", extent=extent,
                      interpolation="nearest", origin="upper", alpha=0.0)
        ax.set_xlabel("X, m")
        ax.set_ylabel("Z, m")
        ax.tick_params(axis="both", which="major", labelsize=8)
        if include_overlays and alpha_survey > 0:
            for (rx, rz) in (receivers or []):
                ax.text(rx, rz, "x", color="lime", fontsize=10, ha="center", va="center", fontweight="bold", alpha=alpha_survey)
            if source is not None:
                sx, sz = source[0], source[1]
                ax.text(sx, sz, "v", color="red", fontsize=10, ha="center", va="center", fontweight="bold", alpha=alpha_survey)
        ax.autoscale(False)
        ax.set_xlim(0, nx * dx)
        ax.set_ylim(nz * dz, 0)
        self.ax = ax
        self._w = self._h = None

    def render_frame(self, snapshot_2d, time_ms, component_label):
        self.snapshot_im.set_data(snapshot_2d)
        self.ax.set_title("t = {} ms, {}".format(int(round(time_ms)), component_label), fontsize=10)
        self.canvas.draw()
        w, h = self.canvas.get_width_height()
        buf = np.asarray(self.canvas.buffer_rgba()).reshape((h, w, 4))
        return buf[:, :, :3].copy()


class ExportSnapshotGifDialog(QDialog):
    """Export → Snapshot Animation: настройки GIF (оверлеи, задержка, цикл, размер)."""
    def __init__(self, parent=None, include_overlays=True, delay_ms=80, loop=0,
                 frame_step=2, default_width_px=600, default_height_px=400):
        super().__init__(parent)
        self.setWindowTitle("Export — Snapshot Animation (GIF)")
        layout = QFormLayout(self)
        self._chk_overlays = QCheckBox()
        self._chk_overlays.setChecked(include_overlays)
        self._chk_overlays.setToolTip("Include model and survey overlay as on Z-X plane")
        layout.addRow("Include Overlays:", self._chk_overlays)
        self._width_spin = QSpinBox()
        self._width_spin.setRange(100, 4096)
        self._width_spin.setSuffix(" px")
        self._width_spin.setValue(max(100, min(4096, default_width_px)))
        self._width_spin.setToolTip("Export frame width (default: current Z-X plane size)")
        layout.addRow("Width:", self._width_spin)
        self._height_spin = QSpinBox()
        self._height_spin.setRange(100, 4096)
        self._height_spin.setSuffix(" px")
        self._height_spin.setValue(max(100, min(4096, default_height_px)))
        self._height_spin.setToolTip("Export frame height (default: current Z-X plane size)")
        layout.addRow("Height:", self._height_spin)
        self._frame_step_spin = QSpinBox()
        self._frame_step_spin.setRange(1, 100)
        self._frame_step_spin.setValue(max(1, min(100, int(frame_step))))
        self._frame_step_spin.setToolTip("1 = all frames, 2 = every 2nd, 3 = every 3rd, …")
        layout.addRow("Export every N-th frame:", self._frame_step_spin)
        self._delay_spin = QSpinBox()
        self._delay_spin.setRange(20, 2000)
        self._delay_spin.setSuffix(" ms")
        self._delay_spin.setValue(delay_ms)
        self._delay_spin.setToolTip("Delay between frames")
        layout.addRow("Frame delay:", self._delay_spin)
        self._loop_spin = QSpinBox()
        self._loop_spin.setRange(0, 10000)
        self._loop_spin.setValue(loop)
        self._loop_spin.setSpecialValueText("Infinite")
        self._loop_spin.setToolTip("0 = loop forever")
        layout.addRow("Loop (0=infinite):", self._loop_spin)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addRow(bb)

    def get_include_overlays(self):
        return self._chk_overlays.isChecked()

    def get_delay_ms(self):
        return self._delay_spin.value()

    def get_loop(self):
        return self._loop_spin.value()

    def get_frame_step(self):
        """Экспортировать каждый N-й кадр (1 = все)."""
        return max(1, self._frame_step_spin.value())

    def get_width_height_px(self):
        """Возвращает (width, height) в пикселях для экспорта (dpi=100 → figsize в дюймах)."""
        return (self._width_spin.value(), self._height_spin.value())


class ModelParametersDialog(QDialog):
    def __init__(self, dx, dz, has_model=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model — Sampling")
        layout = QFormLayout(self)
        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setRange(0.1, 10000)
        self.dx_spin.setDecimals(2)
        self.dx_spin.setValue(dx)
        self.dx_spin.setSuffix(" m")
        layout.addRow("dx (m):", self.dx_spin)
        self.dz_spin = QDoubleSpinBox()
        self.dz_spin.setRange(0.1, 10000)
        self.dz_spin.setDecimals(2)
        self.dz_spin.setValue(dz)
        self.dz_spin.setSuffix(" m")
        layout.addRow("dz (m):", self.dz_spin)
        self.resample_check = QCheckBox("Resample")
        self.resample_check.setChecked(False)
        self.resample_check.setEnabled(bool(has_model))
        self.resample_check.setToolTip("Сохранить текущие extents по X и Z, ресемплировать модель на новые dx, dz (2D интерполяция)")
        layout.addRow("", self.resample_check)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addRow(bb)

    def get_dx_dz(self):
        return self.dx_spin.value(), self.dz_spin.value()

    def get_resample(self):
        return self.resample_check.isChecked()


class LayeredModelDialog(QDialog):
    """Model -> Create Layered: генерация горизонтально-слоистой velocity-модели."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model — Create Layered")
        self._updating = False

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # Defaults per requirement
        self.nx_spin = QSpinBox()
        self.nx_spin.setRange(2, 1000000)
        self.nx_spin.setValue(201)
        form.addRow("NX:", self.nx_spin)

        self.nz_spin = QSpinBox()
        self.nz_spin.setRange(2, 1000000)
        self.nz_spin.setValue(201)
        form.addRow("NZ:", self.nz_spin)

        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setRange(0.01, 100000)
        self.dx_spin.setDecimals(2)
        self.dx_spin.setValue(5.0)
        self.dx_spin.setSuffix(" m")
        form.addRow("DX:", self.dx_spin)

        self.dz_spin = QDoubleSpinBox()
        self.dz_spin.setRange(0.01, 100000)
        self.dz_spin.setDecimals(2)
        self.dz_spin.setValue(5.0)
        self.dz_spin.setSuffix(" m")
        form.addRow("DZ:", self.dz_spin)

        self.x_extent_lbl = QLabel("")
        self.z_extent_lbl = QLabel("")
        form.addRow("X Extent:", self.x_extent_lbl)
        form.addRow("Z Extent:", self.z_extent_lbl)

        self.layers_spin = QSpinBox()
        self.layers_spin.setRange(1, 10000)
        self.layers_spin.setValue(3)
        form.addRow("Layers:", self.layers_spin)

        layout.addLayout(form)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Ztop", "Velocity"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, stretch=1)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._on_accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

        self.nx_spin.valueChanged.connect(self._update_extents)
        self.nz_spin.valueChanged.connect(self._update_extents)
        self.dx_spin.valueChanged.connect(self._update_extents)
        self.dz_spin.valueChanged.connect(self._update_extents)
        self.layers_spin.valueChanged.connect(self._on_layers_changed)

        self._update_extents()
        self._on_layers_changed(self.layers_spin.value(), initial=True)

        self._result = None

    def _extent_x(self):
        nx = int(self.nx_spin.value())
        dx = float(self.dx_spin.value())
        return (nx - 1) * dx

    def _extent_z(self):
        nz = int(self.nz_spin.value())
        dz = float(self.dz_spin.value())
        return (nz - 1) * dz

    def _update_extents(self):
        if self._updating:
            return
        self._updating = True
        try:
            self.x_extent_lbl.setText("{:.2f} m".format(self._extent_x()))
            self.z_extent_lbl.setText("{:.2f} m".format(self._extent_z()))
            # UX: при изменении extents поджать Ztop в таблице под новый Zmax
            self._normalize_ztops()
        finally:
            self._updating = False

    def _normalize_ztops(self):
        """Нормализация Ztop под текущий Zmax без перестановки скоростей.

        - Первая строка всегда 0
        - Остальные: clamp в [0, Zmax-DZ] и строго возрастают (шаг >= DZ)
        - Если из-за маленького Zmax строгий рост невозможен, раскладываем равномерно
        """
        n = int(self.table.rowCount() or 0)
        if n <= 0:
            return
        zmax = max(0.0, float(self._extent_z()))
        dz = float(self.dz_spin.value())
        # Минимальный шаг: 1 dz (чтобы не получать Ztop == Zmax)
        min_step = max(1e-6, dz)
        # Верхняя граница для Ztop (кроме первой строки): Zmax - dz
        ztop_cap = max(0.0, zmax - dz)

        ztops = [self._get_row_value(r, 0, 0.0) for r in range(n)]
        ztops[0] = 0.0
        prev = 0.0
        ok = True
        for i in range(1, n):
            zt = float(ztops[i])
            zt = max(0.0, min(ztop_cap, zt))
            if zt <= prev + 1e-12:
                zt = min(ztop_cap, prev + min_step)
            if zt <= prev + 1e-12 and ztop_cap > 0:
                ok = False
                break
            ztops[i] = zt
            prev = zt

        if not ok and ztop_cap > 0:
            # Раскладываем равномерно так, чтобы строго возрастало
            step = ztop_cap / float(max(1, n))
            prev = 0.0
            ztops[0] = 0.0
            for i in range(1, n):
                cand = i * step
                cand = max(0.0, min(ztop_cap, cand))
                if cand <= prev + 1e-12:
                    cand = min(ztop_cap, prev + min_step)
                ztops[i] = cand
                prev = cand

        for r in range(n):
            self._set_row_value(r, 0, 0.0 if r == 0 else ztops[r])

    def _get_row_value(self, row, col, default=0.0):
        it = self.table.item(row, col)
        if it is None:
            return float(default)
        txt = (it.text() or "").strip()
        if txt == "":
            return float(default)
        try:
            return float(txt)
        except Exception:
            return float(default)

    def _set_row_value(self, row, col, value):
        it = self.table.item(row, col)
        if it is None:
            it = QTableWidgetItem()
            self.table.setItem(row, col, it)
        if isinstance(value, (int, np.integer)):
            it.setText(str(int(value)))
        else:
            it.setText("{:.6g}".format(float(value)))

    def _default_table_for_layers(self, n_layers):
        # Requirement default example for 3 layers; for others: simple monotone guesses
        if n_layers == 1:
            return [(0.0, 2000.0)]
        if n_layers == 3:
            return [(0.0, 1800.0), (305.0, 2200.0), (605.0, 2500.0)]
        zmax = max(0.0, self._extent_z())
        dz = float(self.dz_spin.value())
        ztop_cap = max(0.0, zmax - dz)
        out = []
        for i in range(n_layers):
            ztop = 0.0 if i == 0 else min(ztop_cap, (ztop_cap * i) / max(1, n_layers))
            vel = 1800.0 + 200.0 * i
            out.append((ztop, vel))
        out[0] = (0.0, out[0][1])
        return out

    def _on_layers_changed(self, n_layers, initial=False):
        n_layers = int(n_layers)
        old_rows = self.table.rowCount()
        # preserve existing values where possible
        old = [(self._get_row_value(r, 0, 0.0), self._get_row_value(r, 1, 2000.0)) for r in range(old_rows)]
        self.table.setRowCount(n_layers)
        if initial or old_rows == 0:
            vals = self._default_table_for_layers(n_layers)
        else:
            vals = old[:n_layers]
            if n_layers > len(vals):
                # append new rows based on previous
                zmax = max(0.0, self._extent_z())
                dz = float(self.dz_spin.value())
                ztop_cap = max(0.0, zmax - dz)
                last_z = vals[-1][0] if vals else 0.0
                last_v = vals[-1][1] if vals else 2000.0
                for _ in range(n_layers - len(vals)):
                    last_z = min(ztop_cap, last_z + 300.0)
                    vals.append((last_z, last_v))
        for r in range(n_layers):
            ztop, vel = vals[r]
            self._set_row_value(r, 0, 0.0 if r == 0 else ztop)
            self._set_row_value(r, 1, vel)
        # На всякий случай сразу нормализуем под текущие extents (исключить Ztop == Zmax)
        self._normalize_ztops()

    def _on_accept(self):
        nx = int(self.nx_spin.value())
        nz = int(self.nz_spin.value())
        dx = float(self.dx_spin.value())
        dz = float(self.dz_spin.value())
        n_layers = int(self.layers_spin.value())
        zmax = (nz - 1) * dz
        ztop_cap = max(0.0, zmax - dz)

        ztops = []
        vels = []
        for r in range(n_layers):
            zt = self._get_row_value(r, 0, 0.0)
            vv = self._get_row_value(r, 1, 2000.0)
            ztops.append(float(zt))
            vels.append(float(vv))

        # validation
        if n_layers < 1:
            QMessageBox.warning(self, "Create Layered", "Layer Number must be >= 1.")
            return
        if abs(ztops[0]) > 1e-9:
            QMessageBox.warning(self, "Create Layered", "First Ztop must be 0.")
            return
        prev = -1e-9
        for i, zt in enumerate(ztops):
            if i == 0:
                lo, hi = 0.0, 0.0
            else:
                lo, hi = 0.0, ztop_cap
            if zt < lo - 1e-9 or zt > hi + 1e-9:
                QMessageBox.warning(
                    self,
                    "Create Layered",
                    "Ztop out of range at row {} ({:.2f}..{:.2f}).".format(i + 1, lo, hi),
                )
                return
            if i > 0 and (zt + 1e-9 <= prev):
                QMessageBox.warning(self, "Create Layered", "Ztop must be strictly increasing (row {}).".format(i + 1))
                return
            if i > 0 and dz > 0 and (zt < prev + dz - 1e-9) and i != 0:
                QMessageBox.warning(self, "Create Layered", "Ztop step must be >= DZ (row {}).".format(i + 1))
                return
            prev = zt
        for i, vv in enumerate(vels):
            if not np.isfinite(vv) or vv <= 0:
                QMessageBox.warning(self, "Create Layered", "Velocity must be > 0 (row {}).".format(i + 1))
                return

        self._result = {"nx": nx, "nz": nz, "dx": dx, "dz": dz, "ztops": ztops, "vels": vels}
        self.accept()

    def get_result(self):
        return self._result


class SmoothDialog(QDialog):
    """Model -> Smooth: размер сглаживания в метрах (sigma для gaussian_filter)."""
    def __init__(self, smooth_size_m=0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model — Smooth")
        layout = QFormLayout(self)
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(0, 10000)
        self.size_spin.setDecimals(1)
        self.size_spin.setValue(smooth_size_m)
        self.size_spin.setSuffix(" m")
        self.size_spin.setToolTip("Smoothing radius in meters; converted to sigma for 2D gaussian filter")
        layout.addRow("Smooth Size (m):", self.size_spin)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addRow(bb)

    def get_smooth_size_m(self):
        return self.size_spin.value()


class CropDialog(QDialog):
    """Model → Crop: обрезка модели по границам в метрах (0 = не обрезать)."""
    def __init__(self, crop_x_left=0, crop_x_right=0, crop_z_top=0, crop_z_bottom=0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model — Crop")
        layout = QFormLayout(self)
        self.x_left_spin = QDoubleSpinBox()
        self.x_left_spin.setRange(0, 1e7)
        self.x_left_spin.setDecimals(1)
        self.x_left_spin.setValue(crop_x_left)
        self.x_left_spin.setSuffix(" m")
        layout.addRow("Crop X Left:", self.x_left_spin)
        self.x_right_spin = QDoubleSpinBox()
        self.x_right_spin.setRange(0, 1e7)
        self.x_right_spin.setDecimals(1)
        self.x_right_spin.setValue(crop_x_right)
        self.x_right_spin.setSuffix(" m")
        layout.addRow("Crop X Right:", self.x_right_spin)
        self.z_top_spin = QDoubleSpinBox()
        self.z_top_spin.setRange(0, 1e7)
        self.z_top_spin.setDecimals(1)
        self.z_top_spin.setValue(crop_z_top)
        self.z_top_spin.setSuffix(" m")
        self.z_top_spin.setToolTip("Top = minimum Z (обрезать сверху)")
        layout.addRow("Crop Z Top:", self.z_top_spin)
        self.z_bottom_spin = QDoubleSpinBox()
        self.z_bottom_spin.setRange(0, 1e7)
        self.z_bottom_spin.setDecimals(1)
        self.z_bottom_spin.setValue(crop_z_bottom)
        self.z_bottom_spin.setSuffix(" m")
        self.z_bottom_spin.setToolTip("Bottom = maximum Z (обрезать снизу)")
        layout.addRow("Crop Z Bottom:", self.z_bottom_spin)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addRow(bb)

    def get_crop_m(self):
        return (
            self.x_left_spin.value(),
            self.x_right_spin.value(),
            self.z_top_spin.value(),
            self.z_bottom_spin.value(),
        )


class DiffractorDialog(QDialog):
    def __init__(self, parent=None, x=0, z=0, r=10, v=2000):
        super().__init__(parent)
        self.setWindowTitle("Add diffractor")
        layout = QFormLayout(self)
        self.x_spin = QDoubleSpinBox()
        self.x_spin.setRange(-1e6, 1e6)
        self.x_spin.setDecimals(1)
        self.x_spin.setValue(x)
        layout.addRow("X (m):", self.x_spin)
        self.z_spin = QDoubleSpinBox()
        self.z_spin.setRange(0, 1e6)
        self.z_spin.setDecimals(1)
        self.z_spin.setValue(z)
        layout.addRow("Z (m):", self.z_spin)
        self.r_spin = QDoubleSpinBox()
        self.r_spin.setRange(0.1, 1000)
        self.r_spin.setDecimals(1)
        self.r_spin.setValue(r)
        self.r_spin.setSuffix(" m")
        layout.addRow("R — radius (m):", self.r_spin)
        self.v_spin = QDoubleSpinBox()
        self.v_spin.setRange(100, 10000)
        self.v_spin.setDecimals(0)
        self.v_spin.setValue(v)
        self.v_spin.setSuffix(" m/s")
        layout.addRow("V — velocity (m/s):", self.v_spin)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addRow(bb)

    def get_params(self):
        return {"x": self.x_spin.value(), "z": self.z_spin.value(),
                "r": self.r_spin.value(), "v": self.v_spin.value()}


class DiffractorsDialog(QDialog):
    def __init__(self, diffractors, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model — Diffractors")
        self._diffractors = list(diffractors)
        self._parent = parent
        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        self._refresh_list()
        layout.addWidget(self.list_widget)
        btn_add = QPushButton("+ Add diffractor")
        btn_add.clicked.connect(self._on_add)
        layout.addWidget(btn_add)
        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.accepted.connect(self.accept)
        layout.addWidget(bb)

    def _refresh_list(self):
        self.list_widget.clear()
        for i, d in enumerate(self._diffractors):
            t = "X={:.1f} Z={:.1f} R={:.1f} V={:.0f}".format(
                d["x"], d["z"], d["r"], d["v"])
            self.list_widget.addItem(QListWidgetItem(t))

    def _on_add(self):
        x, z, r, v = 0, 100, 10, 2000
        if self._diffractors:
            last = self._diffractors[-1]
            x, z, r, v = last["x"], last["z"] + 20, last["r"], last["v"]
        dlg = DiffractorDialog(self, x=x, z=z, r=r, v=v)
        if dlg.exec_() == QDialog.Accepted:
            self._diffractors.append(dlg.get_params())
            self._refresh_list()
            if self._parent and hasattr(self._parent, "apply_diffractors_and_redraw"):
                self._parent.apply_diffractors_and_redraw(self._diffractors)

    def get_diffractors(self):
        return self._diffractors


class SourceDialog(QDialog):
    def __init__(self, x=0, z=0, freq=22, parent=None, model_xmax=None, model_zmax=None, model_dz=None):
        super().__init__(parent)
        self.setWindowTitle("Survey — Source")
        self._model_xmax = None if model_xmax is None else float(model_xmax)
        self._model_zmax = None if model_zmax is None else float(model_zmax)
        self._model_dz = None if model_dz is None else float(model_dz)
        layout = QFormLayout(self)
        self.x_spin = QDoubleSpinBox()
        if self._model_xmax is None:
            self.x_spin.setRange(-1e6, 1e6)
        else:
            # Ограничить в пределах модели
            self.x_spin.setRange(0.0, max(0.0, self._model_xmax))
        self.x_spin.setDecimals(1)
        self.x_spin.setValue(x)
        layout.addRow("X (m):", self.x_spin)
        self.z_spin = QDoubleSpinBox()
        if self._model_zmax is None:
            self.z_spin.setRange(0, 1e6)
        else:
            self.z_spin.setRange(0.0, max(0.0, self._model_zmax))
        self.z_spin.setDecimals(1)
        self.z_spin.setValue(z)
        layout.addRow("Z (m):", self.z_spin)
        self.btn_surface_center = QPushButton("Surface Center")
        self.btn_surface_center.setToolTip("Set X to center of model X extent and Z to dz")
        self.btn_surface_center.setEnabled(self._model_xmax is not None and self._model_dz is not None)
        self.btn_surface_center.clicked.connect(self._on_surface_center)
        layout.addRow("", self.btn_surface_center)
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(1, 500)
        self.freq_spin.setDecimals(1)
        self.freq_spin.setValue(freq)
        self.freq_spin.setSuffix(" Hz")
        layout.addRow("Freq — Ricker frequency (Hz):", self.freq_spin)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._on_accept)
        bb.rejected.connect(self.reject)
        layout.addRow(bb)

    def _on_surface_center(self):
        if self._model_xmax is None or self._model_dz is None:
            return
        x = 0.5 * float(self._model_xmax)
        z = float(self._model_dz)
        # clamp to allowed ranges just in case
        x = max(self.x_spin.minimum(), min(self.x_spin.maximum(), x))
        z = max(self.z_spin.minimum(), min(self.z_spin.maximum(), z))
        self.x_spin.setValue(x)
        self.z_spin.setValue(z)

    def _on_accept(self):
        x = float(self.x_spin.value())
        z = float(self.z_spin.value())
        if self._model_xmax is not None:
            if x < 0.0 - 1e-9 or x > self._model_xmax + 1e-9:
                QMessageBox.warning(self, "Survey — Source", "X is out of model extent (0..{:.2f}).".format(self._model_xmax))
                return
        if self._model_zmax is not None:
            if z < 0.0 - 1e-9 or z > self._model_zmax + 1e-9:
                QMessageBox.warning(self, "Survey — Source", "Z is out of model extent (0..{:.2f}).".format(self._model_zmax))
                return
        self.accept()

    def get_params(self):
        return self.x_spin.value(), self.z_spin.value(), self.freq_spin.value()


class ReceiversDialog(QDialog):
    def __init__(self, receivers=None, layout_name="Profile", parent=None, model_xmax=None, model_zmax=None):
        super().__init__(parent)
        self.setWindowTitle("Survey — Receivers")
        self._model_xmax = None if model_xmax is None else float(model_xmax)
        self._model_zmax = None if model_zmax is None else float(model_zmax)
        self._updating = False
        layout = QVBoxLayout(self)
        grp = QGroupBox("Receiver layout type")
        self.radio_profile = QRadioButton("Profile")
        self.radio_well = QRadioButton("Well")
        # Значения по умолчанию (если приёмников нет или не удаётся восстановить)
        p_z, p_x0, p_nx, p_xstep = 0.0, 0.0, 50, 5.0
        w_x, w_z0, w_nz, w_zstep = 500.0, 0.0, 100, 5.0
        if receivers and len(receivers) > 0:
            xs = [r[0] for r in receivers]
            zs = [r[1] for r in receivers]
            if layout_name == "Well":
                self.radio_well.setChecked(True)
                w_x = xs[0]
                w_z0 = zs[0]
                w_nz = len(receivers)
                w_zstep = (zs[-1] - zs[0]) / (w_nz - 1) if w_nz > 1 else 5.0
            else:
                self.radio_profile.setChecked(True)
                p_z = zs[0]
                p_x0 = xs[0]
                p_nx = len(receivers)
                p_xstep = (xs[-1] - xs[0]) / (p_nx - 1) if p_nx > 1 else 5.0
        else:
            self.radio_profile.setChecked(True)
        grp_layout = QVBoxLayout(grp)
        grp_layout.addWidget(self.radio_profile)
        grp_layout.addWidget(self.radio_well)
        layout.addWidget(grp)

        self.form_profile = QFrame()
        f_profile = QFormLayout(self.form_profile)
        self.p_z = QDoubleSpinBox()
        self.p_z.setRange(0, 1e6)
        self.p_z.setValue(p_z)
        f_profile.addRow("Z (m):", self.p_z)
        self.p_x0 = QDoubleSpinBox()
        self.p_x0.setRange(-1e6, 1e6)
        self.p_x0.setValue(p_x0)
        f_profile.addRow("X0 (m):", self.p_x0)
        self.p_nx = QSpinBox()
        self.p_nx.setRange(1, 10000)
        self.p_nx.setValue(p_nx)
        f_profile.addRow("NX:", self.p_nx)
        self.p_xstep = QDoubleSpinBox()
        self.p_xstep.setRange(0.1, 1000)
        self.p_xstep.setValue(p_xstep)
        f_profile.addRow("X step (m):", self.p_xstep)
        self.btn_fill_profile = QPushButton("Fill Extent")
        self.btn_fill_profile.clicked.connect(self._fill_extent_profile)
        f_profile.addRow("", self.btn_fill_profile)
        layout.addWidget(self.form_profile)

        self.form_well = QFrame()
        f_well = QFormLayout(self.form_well)
        self.w_x = QDoubleSpinBox()
        self.w_x.setRange(-1e6, 1e6)
        self.w_x.setValue(w_x)
        f_well.addRow("X (m):", self.w_x)
        self.w_z0 = QDoubleSpinBox()
        self.w_z0.setRange(0, 1e6)
        self.w_z0.setValue(w_z0)
        f_well.addRow("Z0 (m):", self.w_z0)
        self.w_nz = QSpinBox()
        self.w_nz.setRange(1, 10000)
        self.w_nz.setValue(w_nz)
        f_well.addRow("NZ:", self.w_nz)
        self.w_zstep = QDoubleSpinBox()
        self.w_zstep.setRange(0.1, 1000)
        self.w_zstep.setValue(w_zstep)
        f_well.addRow("Z step (m):", self.w_zstep)
        self.btn_fill_well = QPushButton("Fill Extent")
        self.btn_fill_well.clicked.connect(self._fill_extent_well)
        f_well.addRow("", self.btn_fill_well)
        layout.addWidget(self.form_well)
        self.form_well.setVisible(self.radio_well.isChecked())
        self.form_profile.setVisible(self.radio_profile.isChecked())

        def toggled():
            self.form_profile.setVisible(self.radio_profile.isChecked())
            self.form_well.setVisible(self.radio_well.isChecked())
            self._apply_constraints()
        self.radio_profile.toggled.connect(toggled)

        # Ограничения по модели: не позволять NX/NZ уходить за пределы extents
        self.p_x0.valueChanged.connect(self._apply_constraints)
        self.p_nx.valueChanged.connect(self._apply_constraints)
        self.p_xstep.valueChanged.connect(self._apply_constraints)
        self.w_z0.valueChanged.connect(self._apply_constraints)
        self.w_nz.valueChanged.connect(self._apply_constraints)
        self.w_zstep.valueChanged.connect(self._apply_constraints)
        self._apply_constraints()

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def _max_count_for_extent(self, start, step, maxv):
        """Максимальное N, чтобы start + (N-1)*step <= maxv."""
        if maxv is None:
            return None
        try:
            start = float(start)
            step = float(step)
            maxv = float(maxv)
        except Exception:
            return None
        if step <= 0:
            return 1
        if maxv < start:
            return 1
        return int(np.floor((maxv - start) / step)) + 1

    def _apply_constraints(self):
        if self._updating:
            return
        self._updating = True
        try:
            if self.radio_profile.isChecked():
                nmax = self._max_count_for_extent(self.p_x0.value(), self.p_xstep.value(), self._model_xmax)
                if nmax is not None:
                    nmax = max(1, nmax)
                    if self.p_nx.value() > nmax:
                        self.p_nx.setValue(nmax)
                    self.p_nx.setMaximum(max(1, nmax))
            else:
                nmax = self._max_count_for_extent(self.w_z0.value(), self.w_zstep.value(), self._model_zmax)
                if nmax is not None:
                    nmax = max(1, nmax)
                    if self.w_nz.value() > nmax:
                        self.w_nz.setValue(nmax)
                    self.w_nz.setMaximum(max(1, nmax))
        finally:
            self._updating = False

    def _fill_extent_profile(self):
        # Заполнить от 0 до Xmax с текущим шагом
        if self._model_xmax is None:
            return
        step = float(self.p_xstep.value())
        if step <= 0:
            return
        # Сначала сбросить X0, затем обновить ограничения (в т.ч. максимум NX),
        # и только потом выставлять NX — иначе setValue() может быть обрезан старым maximum.
        self._updating = True
        try:
            self.p_x0.setValue(0.0)
        finally:
            self._updating = False
        self._apply_constraints()
        nmax = self._max_count_for_extent(0.0, step, self._model_xmax)
        if nmax is None:
            return
        self.p_nx.setValue(max(1, nmax))
        self._apply_constraints()

    def _fill_extent_well(self):
        # Заполнить от 0 до Zmax с текущим шагом
        if self._model_zmax is None:
            return
        step = float(self.w_zstep.value())
        if step <= 0:
            return
        # Аналогично Profile: сперва обновить ограничения, чтобы максимум NZ был актуален
        self._updating = True
        try:
            self.w_z0.setValue(0.0)
        finally:
            self._updating = False
        self._apply_constraints()
        nmax = self._max_count_for_extent(0.0, step, self._model_zmax)
        if nmax is None:
            return
        self.w_nz.setValue(max(1, nmax))
        self._apply_constraints()

    def get_layout(self):
        """Возвращает 'Profile' или 'Well' в зависимости от выбранного типа расстановки."""
        return "Profile" if self.radio_profile.isChecked() else "Well"

    def get_receiver_points(self):
        if self.radio_profile.isChecked():
            z = self.p_z.value()
            x0 = self.p_x0.value()
            nx = self.p_nx.value()
            xstep = self.p_xstep.value()
            return [(x0 + i * xstep, z) for i in range(nx)]
        else:
            x = self.w_x.value()
            z0 = self.w_z0.value()
            nz = self.w_nz.value()
            zstep = self.w_zstep.value()
            return [(x, z0 + i * zstep) for i in range(nz)]


class SimulationSettingsDialog(QDialog):
    """Параметры симуляции: Tmax, NPML, DT, SNAPSHOT_DT, Spatial order (2nd/4th). Модель (Original/Smoothed) задаётся в диалогах Run Forward / Run Backward."""
    def __init__(self, parent=None, tmax_ms=1000, npml=50, dt_ms=None, snapshot_dt_ms=2, laplacian="4pt"):
        super().__init__(parent)
        self.setWindowTitle("Simulation — Settings")
        layout = QFormLayout(self)
        self.tmax_spin = QDoubleSpinBox()
        self.tmax_spin.setRange(1, 1e6)
        self.tmax_spin.setDecimals(0)
        self.tmax_spin.setValue(tmax_ms)
        self.tmax_spin.setSuffix(" ms")
        layout.addRow("Tmax (ms):", self.tmax_spin)
        self.npml_spin = QSpinBox()
        self.npml_spin.setRange(0, 500)
        self.npml_spin.setValue(npml)
        layout.addRow("NPML (grid nodes):", self.npml_spin)
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.01, 10)
        self.dt_spin.setDecimals(3)
        self.dt_spin.setValue(dt_ms if dt_ms is not None else 0.5)
        self.dt_spin.setSuffix(" ms")
        self.dt_spin.setToolTip("Default ~0.5 CFL")
        layout.addRow("DT (ms):", self.dt_spin)
        self.snapshot_dt_spin = QDoubleSpinBox()
        self.snapshot_dt_spin.setRange(0.1, 1000)
        self.snapshot_dt_spin.setDecimals(2)
        self.snapshot_dt_spin.setValue(snapshot_dt_ms)
        self.snapshot_dt_spin.setSuffix(" ms")
        layout.addRow("SNAPSHOT_DT (ms):", self.snapshot_dt_spin)
        self.laplacian_combo = QComboBox()
        self.laplacian_combo.addItems(["4pt", "2pt"])
        self.laplacian_combo.setCurrentText(laplacian if laplacian in ("4pt", "2pt") else "4pt")
        layout.addRow("Spatial order:", self.laplacian_combo)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addRow(bb)

    def get_params(self):
        return {
            "tmax_ms": self.tmax_spin.value(),
            "npml": self.npml_spin.value(),
            "dt_ms": self.dt_spin.value(),
            "snapshot_dt_ms": self.snapshot_dt_spin.value(),
            "laplacian": self.laplacian_combo.currentText(),
        }


class RTMSettingsDialog(QDialog):
    """RTM → Settings: выбор компонента Source (P, Vz, Vx) для кросс-корреляции."""
    def __init__(self, source="P", parent=None):
        super().__init__(parent)
        self.setWindowTitle("RTM — Settings")
        layout = QFormLayout(self)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["P", "Vz", "Vx"])
        self.source_combo.setCurrentText(source if source in ("P", "Vz", "Vx") else "P")
        layout.addRow("Source:", self.source_combo)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addRow(bb)

    def get_source(self):
        return self.source_combo.currentText()


class RTMPostProcessingDialog(QDialog):
    """RTM → Post-Processing: Laplacian Filter (порядок 2/4), AGC (размер окна по Z в м)."""
    def __init__(self, laplacian_on=False, laplacian_order=4, agc_on=False, agc_window_z_m=1000.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RTM — Post-Processing")
        layout = QFormLayout(self)
        self._chk_laplacian = QCheckBox()
        self._chk_laplacian.setChecked(laplacian_on)
        layout.addRow("Laplacian Filter:", self._chk_laplacian)
        self._laplacian_order_combo = QComboBox()
        self._laplacian_order_combo.addItems(["2", "4"])
        self._laplacian_order_combo.setCurrentText(str(laplacian_order) if str(laplacian_order) in ("2", "4") else "4")
        layout.addRow("Laplacian order:", self._laplacian_order_combo)
        self._chk_agc = QCheckBox()
        self._chk_agc.setChecked(agc_on)
        layout.addRow("AGC:", self._chk_agc)
        self._agc_window_z = QDoubleSpinBox()
        self._agc_window_z.setRange(1, 100000)
        self._agc_window_z.setDecimals(0)
        self._agc_window_z.setValue(agc_window_z_m)
        self._agc_window_z.setSuffix(" m")
        layout.addRow("Window size along Z:", self._agc_window_z)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addRow(bb)

    def get_params(self):
        return {
            "laplacian_on": self._chk_laplacian.isChecked(),
            "laplacian_order": int(self._laplacian_order_combo.currentText()),
            "agc_on": self._chk_agc.isChecked(),
            "agc_window_z_m": self._agc_window_z.value(),
        }


class ForwardRunNameDialog(QDialog):
    """Запрос имени набора и модели перед Run Forward."""
    def __init__(self, default_name="Fwd 1", model_source="Original", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Run Forward — run name and model")
        layout = QFormLayout(self)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. Fwd From Orig Model")
        self._name_edit.setText(default_name)
        layout.addRow("Run name:", self._name_edit)
        self._model_combo = QComboBox()
        self._model_combo.addItems(["Original", "Smoothed"])
        self._model_combo.setCurrentText(model_source if model_source in ("Original", "Smoothed") else "Original")
        layout.addRow("Model for simulation:", self._model_combo)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addRow(bb)

    def get_name(self):
        return self._name_edit.text().strip() or "Fwd 1"

    def get_model_source(self):
        return self._model_combo.currentText()


class BackwardRunNameDialog(QDialog):
    """Запрос имени набора, источника сейсмограммы (сейсмограмма или Residual) и модели перед Run Backward."""
    def __init__(self, default_name="Bwd 1", seismogram_names=None, model_source="Original",
                 default_source_type=None, default_residual_full=None, default_residual_smoothed=None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Run Backward — run name, seismogram and model")
        layout = QFormLayout(self)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. Bwd From Smoothed")
        self._name_edit.setText(default_name)
        layout.addRow("Run name:", self._name_edit)
        self._source_type_combo = QComboBox()
        self._source_type_combo.addItem("Seismogram")
        if (seismogram_names or []) and len(seismogram_names) >= 2:
            self._source_type_combo.addItem("Residual")
        self._source_type_combo.currentTextChanged.connect(self._on_source_type_changed)
        layout.addRow("Source type:", self._source_type_combo)
        self._stack = QStackedWidget()
        page_named = QWidget()
        form_named = QFormLayout(page_named)
        self._seismogram_combo = QComboBox()
        self._seismogram_combo.addItems(seismogram_names or [])
        form_named.addRow("Seismogram (source):", self._seismogram_combo)
        self._stack.addWidget(page_named)
        page_residual = QWidget()
        form_residual = QFormLayout(page_residual)
        self._residual_full_combo = QComboBox()
        self._residual_full_combo.addItems(seismogram_names or [])
        form_residual.addRow("Full fwd (seismogram):", self._residual_full_combo)
        self._residual_smoothed_combo = QComboBox()
        self._residual_smoothed_combo.addItems(seismogram_names or [])
        form_residual.addRow("Smoothed fwd (seismogram):", self._residual_smoothed_combo)
        self._stack.addWidget(page_residual)
        layout.addRow(self._stack)
        if default_source_type == "Residual":
            self._source_type_combo.setCurrentText("Residual")
            if default_residual_full and self._residual_full_combo.findText(default_residual_full) >= 0:
                self._residual_full_combo.setCurrentText(default_residual_full)
            if default_residual_smoothed and self._residual_smoothed_combo.findText(default_residual_smoothed) >= 0:
                self._residual_smoothed_combo.setCurrentText(default_residual_smoothed)
        self._on_source_type_changed(self._source_type_combo.currentText())
        self._model_combo = QComboBox()
        self._model_combo.addItems(["Original", "Smoothed"])
        self._model_combo.setCurrentText(model_source if model_source in ("Original", "Smoothed") else "Original")
        layout.addRow("Model for simulation:", self._model_combo)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addRow(bb)

    def _on_source_type_changed(self, text):
        if text == "Residual":
            self._stack.setCurrentIndex(1)
        else:
            self._stack.setCurrentIndex(0)

    def get_name(self):
        return self._name_edit.text().strip() or "Bwd 1"

    def get_seismogram_source_type(self):
        """Returns 'named' or 'residual'."""
        return "residual" if self._source_type_combo.currentText() == "Residual" else "named"

    def get_seismogram_source_name(self):
        """For 'named' type — seismogram run name; for 'residual' — None."""
        if self.get_seismogram_source_type() != "named":
            return None
        return self._seismogram_combo.currentText() if self._seismogram_combo.count() > 0 else None

    def get_residual_full_name(self):
        return self._residual_full_combo.currentText() if self._residual_full_combo.count() > 0 else None

    def get_residual_smoothed_name(self):
        return self._residual_smoothed_combo.currentText() if self._residual_smoothed_combo.count() > 0 else None

    def get_model_source(self):
        return self._model_combo.currentText()


class RTMBuildDialog(QDialog):
    """RTM Build: выбор источников Forward и Backward снапшотов и компонента."""
    def __init__(self, fwd_names=None, bwd_names=None, default_fwd=None, default_bwd=None, source="P", parent=None):
        super().__init__(parent)
        self.setWindowTitle("RTM — Build")
        layout = QFormLayout(self)
        self._fwd_combo = QComboBox()
        self._fwd_combo.addItems(fwd_names or [])
        if default_fwd and default_fwd in (fwd_names or []):
            self._fwd_combo.setCurrentText(default_fwd)
        layout.addRow("Forward snapshots:", self._fwd_combo)
        self._bwd_combo = QComboBox()
        self._bwd_combo.addItems(bwd_names or [])
        if default_bwd and default_bwd in (bwd_names or []):
            self._bwd_combo.setCurrentText(default_bwd)
        layout.addRow("Backward snapshots:", self._bwd_combo)
        self._source_combo = QComboBox()
        self._source_combo.addItems(["P", "Vz", "Vx"])
        self._source_combo.setCurrentText(source if source in ("P", "Vz", "Vx") else "P")
        layout.addRow("Component (Source):", self._source_combo)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addRow(bb)

    def get_fwd_name(self):
        return self._fwd_combo.currentText() if self._fwd_combo.count() > 0 else None

    def get_bwd_name(self):
        return self._bwd_combo.currentText() if self._bwd_combo.count() > 0 else None

    def get_source(self):
        return self._source_combo.currentText()


class ForwardSimulationWorker(QObject):
    """Воркер для прямого моделирования в отдельном потоке; эмитит progress(current, total) и finished(result)."""
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, src, vp, nt, dt, dx, dz, xsrc, zsrc, n_absorb, save_every, order,
                 snapshots_h5_path=None, snapshot_dt_ms=None, tmax_ms=None):
        super().__init__()
        self._src = src
        self._vp = vp
        self._nt = nt
        self._dt = dt
        self._dx = dx
        self._dz = dz
        self._xsrc = xsrc
        self._zsrc = zsrc
        self._n_absorb = n_absorb
        self._save_every = save_every
        self._order = order
        self._snapshots_h5_path = snapshots_h5_path
        self._snapshot_dt_ms = snapshot_dt_ms
        self._tmax_ms = tmax_ms

    def run(self):
        try:
            result = fd2d_forward_first_order(
                self._src,
                self._vp,
                self._nt,
                self._dt,
                self._dx,
                self._dz,
                self._xsrc,
                self._zsrc,
                rho=None,
                n_absorb=self._n_absorb,
                save_every=self._save_every,
                return_vz=True,
                return_vx=True,
                order=self._order,
                progress_callback=lambda i, n: self.progress.emit(i, n),
                snapshots_h5_path=self._snapshots_h5_path,
                snapshot_dt_ms=self._snapshot_dt_ms,
                tmax_ms=self._tmax_ms,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class BackwardSimulationWorker(QObject):
    """Воркер обратного прогона: record (nt, nrec), vp, nt, dt, xrec, zrec, ..."""
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, record, vp, nt, dt, dx, dz, xrec, zrec, n_absorb, save_every, order,
                 snapshots_h5_path=None, snapshot_dt_ms=None, tmax_ms=None, seismogram_source=None):
        super().__init__()
        self._record = record
        self._vp = vp
        self._nt = nt
        self._dt = dt
        self._dx = dx
        self._dz = dz
        self._xrec = xrec
        self._zrec = zrec
        self._n_absorb = n_absorb
        self._save_every = save_every
        self._order = order
        self._snapshots_h5_path = snapshots_h5_path
        self._snapshot_dt_ms = snapshot_dt_ms
        self._tmax_ms = tmax_ms
        self._seismogram_source = seismogram_source

    def run(self):
        try:
            result = fd2d_backward_first_order(
                self._record,
                self._vp,
                self._nt,
                self._dt,
                self._dx,
                self._dz,
                self._xrec,
                self._zrec,
                rho=None,
                n_absorb=self._n_absorb,
                save_every=self._save_every,
                return_vz=True,
                return_vx=True,
                order=self._order,
                progress_callback=lambda i, n: self.progress.emit(i, n),
                snapshots_h5_path=self._snapshots_h5_path,
                snapshot_dt_ms=self._snapshot_dt_ms,
                tmax_ms=self._tmax_ms,
                seismogram_source=self._seismogram_source,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ExportSeismogramSegyDialog(QDialog):
    """Параметры экспорта текущей сейсмограммы в SEG-Y."""
    # Байты заголовков (SEG-Y trace header)
    SX_BYTE = 73
    SZ_BYTE = 45
    RX_BYTE = 81
    RZ_BYTE = 41
    SCALER_BYTE_1 = 71
    SCALER_BYTE_2 = 69

    def __init__(self, max_time_ms=1000.0, dt_ms=2.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Seismogram to SEG-Y")
        layout = QFormLayout(self)
        self._format_combo = QComboBox()
        self._format_combo.addItems(["IEEE", "IBM"])
        layout.addRow("Format:", self._format_combo)
        self._max_time_spin = QDoubleSpinBox()
        self._max_time_spin.setRange(0.1, 1e6)
        self._max_time_spin.setDecimals(2)
        self._max_time_spin.setValue(max_time_ms)
        self._max_time_spin.setSuffix(" ms")
        layout.addRow("Max Time, ms:", self._max_time_spin)
        self._dt_spin = QDoubleSpinBox()
        self._dt_spin.setRange(0.01, 1000.0)
        self._dt_spin.setDecimals(4)
        self._dt_spin.setValue(dt_ms)
        self._dt_spin.setSuffix(" ms")
        layout.addRow("DT, ms:", self._dt_spin)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addRow(bb)

    def get_format_ieee(self):
        return self._format_combo.currentText() == "IEEE"

    def get_max_time_ms(self):
        return self._max_time_spin.value()

    def get_dt_ms(self):
        return self._dt_spin.value()


class AboutDialog(QDialog):
    """Диалог About: показывает содержимое README.md в виде текста."""

    def __init__(self, parent=None, text=""):
        super().__init__(parent)
        self.setWindowTitle("About Sim1Shot2D")
        layout = QVBoxLayout(self)
        self._text_edit = QTextEdit(self)
        self._text_edit.setReadOnly(True)
        self._text_edit.setPlainText(text or "No README.md content available.")
        layout.addWidget(self._text_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok, parent=self)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        self.resize(640, 480)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sim1Shot2D — Model & Survey")
        self._vp = None
        self._dx = self._dz = 1.0
        # SEG-Y (self._model_file_path) и Layered Model (self._layered_model_spec) — взаимоисключающие режимы
        self._layered_model_spec = None  # {"nx","nz","dx","dz","ztops","vels"} или None
        self._diffractors = []
        self._source = None  # (x, z, freq)
        self._receivers = []
        self._sim_settings = {
            "tmax_ms": 1000, "npml": 50, "dt_ms": 0.5,
            "snapshot_dt_ms": 2, "laplacian": "4pt",
        }
        self._smooth_size_m = 0.0
        self._forward_runs = []   # [{"name": str, "snapshots": {"P fwd", "Vz fwd", "Vx fwd"}, "seismograms": {"P","Vz","Vx"}, "seismogram_t_ms"}, ...]
        self._backward_runs = []  # [{"name": str, "snapshots": {"P bwd", "Vz bwd", "Vx bwd"}, "seismogram_source": str}, ...]
        self._current_fwd_name = None   # имя набора fwd для отображения снапшотов
        self._current_bwd_name = None   # имя набора bwd для отображения снапшотов
        self._current_seismogram_name = None  # имя forward-набора, чья сейсмограмма отображается
        self._snapshots = None   # эффективный dict для отображения (merge current fwd + current bwd)
        self._seismogram_data = None
        self._seismogram_t_ms = None
        self._seismogram_component = "P"
        self._pending_forward_name = None
        self._pending_backward_name = None
        self._pending_backward_seismogram_name = None
        self._rtm_image = None  # (nz, nx) RTM image (постобработанный для отображения)
        self._rtm_image_base = None  # (nz, nx) исходный Image после Build
        self._rtm_settings = {"source": "P"}
        self._rtm_postproc = {"laplacian_on": False, "laplacian_order": 4, "agc_on": False, "agc_window_z_m": 1000.0}
        self._receiver_layout = "Profile"  # "Profile" или "Well" — для осей сейсмограммы
        # Последние использованные настройки экспорта анимации снапшотов (сеансовые)
        self._snapshot_gif_settings = {
            "include_overlays": True,
            "width_px": 600,
            "height_px": 400,
            "delay_ms": 80,
            "loop": 0,
            "frame_step": 2,
        }
        self._project_path = None  # путь к текущему файлу проекта (.ini) или None
        self._model_file_path = ""  # путь к загруженному SEG-Y модели
        self._vp_full = None  # оригинальная модель до кропа (референс для Crop)
        self._crop_x_left = 0.0
        self._crop_x_right = 0.0
        self._crop_z_top = 0.0
        self._crop_z_bottom = 0.0

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        row = QHBoxLayout()
        self._layer_panel = self._build_layer_panel()
        row.addWidget(self._layer_panel)
        # Колонка: заголовок Z-X Plane + холст + Snapshot Percentile снизу
        canvas_column = QWidget()
        canvas_col_layout = QVBoxLayout(canvas_column)
        canvas_col_layout.setContentsMargins(2, 2, 2, 2)
        canvas_col_layout.setSpacing(2)
        self._canvas_title = QLabel("Z-X Plane")
        self._canvas_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
        self._canvas_title.setAlignment(Qt.AlignHCenter)
        canvas_col_layout.addWidget(self._canvas_title)
        self.canvas = VelocityCanvas(self)
        canvas_col_layout.addWidget(self.canvas, stretch=1)
        self._snapshot_percentile_row = QWidget()
        snap_pl = QHBoxLayout(self._snapshot_percentile_row)
        snap_pl.addWidget(QLabel("Snapshot Percentile:"))
        self._snapshot_percentile_spin = QDoubleSpinBox()
        self._snapshot_percentile_spin.setRange(50, 100)
        self._snapshot_percentile_spin.setDecimals(0)
        self._snapshot_percentile_spin.setValue(98)
        self._snapshot_percentile_spin.setSuffix(" %")
        self._snapshot_percentile_spin.valueChanged.connect(self._update_canvas_layers)
        snap_pl.addWidget(self._snapshot_percentile_spin)
        snap_pl.addSpacing(20)
        snap_pl.addWidget(QLabel("Image Percentile:"))
        self._image_percentile_spin = QDoubleSpinBox()
        self._image_percentile_spin.setRange(50, 100)
        self._image_percentile_spin.setDecimals(0)
        self._image_percentile_spin.setValue(98)
        self._image_percentile_spin.setSuffix(" %")
        self._image_percentile_spin.valueChanged.connect(self._update_canvas_layers)
        snap_pl.addWidget(self._image_percentile_spin)
        snap_pl.addStretch()
        canvas_col_layout.addWidget(self._snapshot_percentile_row)
        row.addWidget(canvas_column, stretch=1)
        # Колонка: заголовок Seismogram + сейсмограмма + Seismogram Percentile снизу
        seismogram_column = QWidget()
        seis_col_layout = QVBoxLayout(seismogram_column)
        seis_col_layout.setContentsMargins(2, 2, 2, 2)
        seis_col_layout.setSpacing(2)
        self._seismogram_title = QLabel("Seismogram")
        self._seismogram_title.setStyleSheet("font-weight: bold; font-size: 11pt;")
        self._seismogram_title.setAlignment(Qt.AlignHCenter)
        seis_col_layout.addWidget(self._seismogram_title)
        self.seismogram_canvas = SeismogramCanvas(self)
        self.seismogram_canvas.set_layout(self._receiver_layout)
        seis_col_layout.addWidget(self.seismogram_canvas, stretch=1)
        self._seismogram_percentile_row = QWidget()
        seis_pl = QHBoxLayout(self._seismogram_percentile_row)
        seis_pl.addWidget(QLabel("Seismogram Percentile:"))
        self._seismogram_percentile_spin = QDoubleSpinBox()
        self._seismogram_percentile_spin.setRange(50, 100)
        self._seismogram_percentile_spin.setDecimals(0)
        self._seismogram_percentile_spin.setValue(98)
        self._seismogram_percentile_spin.setSuffix(" %")
        self._seismogram_percentile_spin.valueChanged.connect(self._on_seismogram_percentile_changed)
        seis_pl.addWidget(self._seismogram_percentile_spin)
        seis_pl.addStretch()
        seis_col_layout.addWidget(self._seismogram_percentile_row)
        row.addWidget(seismogram_column, stretch=1)
        main_layout.addLayout(row)
        self._snapshot_slider_row = QWidget()
        slayout = QHBoxLayout(self._snapshot_slider_row)
        slayout.addWidget(QLabel("Snapshot time:"))
        self._snapshot_slider = QSlider(Qt.Horizontal)
        self._snapshot_slider.setMinimum(0)
        self._snapshot_slider.setMaximum(0)
        self._snapshot_slider.valueChanged.connect(self._on_snapshot_slider)
        slayout.addWidget(self._snapshot_slider, stretch=1)
        self._snapshot_time_label = QLabel("0")
        slayout.addWidget(self._snapshot_time_label)
        main_layout.addWidget(self._snapshot_slider_row)
        self._snapshot_slider_row.setVisible(False)

        self._progress_row = QWidget()
        playout = QHBoxLayout(self._progress_row)
        playout.addWidget(QLabel("Progress:"))
        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        playout.addWidget(self._progress_bar, stretch=1)
        main_layout.addWidget(self._progress_row)

        self._forward_thread = None
        self._forward_worker = None
        self._backward_thread = None
        self._backward_worker = None

        self._memory_label = QLabel("Memory: —")
        self._memory_label.setToolTip("Process memory usage (RSS). Updated every 5 s.")
        self.statusBar().addPermanentWidget(self._memory_label)
        self._memory_timer = QTimer(self)
        self._memory_timer.timeout.connect(self._update_memory_status)
        self._memory_timer.start(5000)
        self._update_memory_status()

        self._update_layer_availability()

        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        act_load = QAction("Load Project", self)
        act_load.triggered.connect(self._file_load_project)
        file_menu.addAction(act_load)
        act_save = QAction("Save Project", self)
        act_save.triggered.connect(self._file_save_project)
        file_menu.addAction(act_save)
        act_save_as = QAction("Save Project As...", self)
        act_save_as.triggered.connect(self._file_save_project_as)
        file_menu.addAction(act_save_as)
        file_menu.addSeparator()
        act_reset = QAction("Reset Project", self)
        act_reset.triggered.connect(self._file_reset_project)
        file_menu.addAction(act_reset)
        file_menu.addSeparator()
        act_exit = QAction("Exit", self)
        act_exit.triggered.connect(self._file_exit)
        file_menu.addAction(act_exit)

        model_menu = menubar.addMenu("&Model")
        act_open = QAction("Load SEG-Y", self)
        act_open.triggered.connect(self._model_open)
        model_menu.addAction(act_open)
        act_create_layered = QAction("Create Layered", self)
        act_create_layered.triggered.connect(self._model_create_layered)
        model_menu.addAction(act_create_layered)
        act_params = QAction("Sampling", self)
        act_params.triggered.connect(self._model_parameters)
        model_menu.addAction(act_params)
        act_crop = QAction("Crop", self)
        act_crop.triggered.connect(self._model_crop)
        model_menu.addAction(act_crop)
        act_diff = QAction("Diffractors", self)
        act_diff.triggered.connect(self._model_diffractors)
        model_menu.addAction(act_diff)
        act_smooth = QAction("Smooth", self)
        act_smooth.triggered.connect(self._model_smooth)
        model_menu.addAction(act_smooth)

        survey_menu = menubar.addMenu("&Survey")
        act_src = QAction("Source", self)
        act_src.triggered.connect(self._survey_source)
        survey_menu.addAction(act_src)
        act_rec = QAction("Receivers", self)
        act_rec.triggered.connect(self._survey_receivers)
        survey_menu.addAction(act_rec)

        sim_menu = menubar.addMenu("Simulation")
        act_settings = QAction("Settings", self)
        act_settings.triggered.connect(self._simulation_settings)
        sim_menu.addAction(act_settings)
        act_forward = QAction("Run Forward", self)
        act_forward.triggered.connect(self._simulation_run_forward)
        sim_menu.addAction(act_forward)
        self._act_backward = QAction("Run Backward", self)
        self._act_backward.triggered.connect(self._simulation_run_backward)
        self._act_backward.setEnabled(False)
        sim_menu.addAction(self._act_backward)

        rtm_menu = menubar.addMenu("RTM")
        act_rtm_settings = QAction("Settings", self)
        act_rtm_settings.triggered.connect(self._rtm_settings_dialog)
        rtm_menu.addAction(act_rtm_settings)
        self._act_rtm_build = QAction("Build", self)
        self._act_rtm_build.triggered.connect(self._rtm_build)
        self._act_rtm_build.setEnabled(False)
        rtm_menu.addAction(self._act_rtm_build)
        act_rtm_postproc = QAction("Post-Processing", self)
        act_rtm_postproc.triggered.connect(self._rtm_post_processing_dialog)
        rtm_menu.addAction(act_rtm_postproc)

        export_menu = menubar.addMenu("E&xport")
        act_snapshot_gif = QAction("Snapshot Animation", self)
        act_snapshot_gif.triggered.connect(self._export_snapshot_animation)
        export_menu.addAction(act_snapshot_gif)
        act_seismogram_segy = QAction("Seismogram SEG-Y", self)
        act_seismogram_segy.triggered.connect(self._export_seismogram_segy)
        export_menu.addAction(act_seismogram_segy)

        help_menu = menubar.addMenu("&Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(self._show_about_dialog)
        help_menu.addAction(act_about)

    def _show_about_dialog(self):
        """Открыть диалог About и показать содержимое README.md."""
        readme_path = os.path.join(_gui_dir, "README.md")
        text = ""
        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            text = "README.md not found.\nExpected path: {}\n\nError: {}".format(readme_path, e)
        dlg = AboutDialog(self, text=text)
        dlg.exec_()

    def _build_layer_panel(self):
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(panel)
        grp_current = QGroupBox("Current runs")
        gl_current = QFormLayout()
        self._combo_current_fwd = QComboBox()
        self._combo_current_fwd.currentTextChanged.connect(self._on_current_fwd_changed)
        gl_current.addRow("Current Fwd Snapshots:", self._combo_current_fwd)
        self._combo_current_bwd = QComboBox()
        self._combo_current_bwd.currentTextChanged.connect(self._on_current_bwd_changed)
        gl_current.addRow("Current Bwd Snapshots:", self._combo_current_bwd)
        self._combo_current_seismogram = QComboBox()
        self._combo_current_seismogram.currentTextChanged.connect(self._on_current_seismogram_changed)
        gl_current.addRow("Current Seismogram:", self._combo_current_seismogram)
        grp_current.setLayout(gl_current)
        layout.addWidget(grp_current)
        grp_seis = QGroupBox("Seismogram")
        gl_seis = QFormLayout()
        self._combo_seismogram_component = QComboBox()
        self._combo_seismogram_component.addItems(["P", "Vz", "Vx"])
        self._combo_seismogram_component.setCurrentText(getattr(self, "_seismogram_component", "P") or "P")
        self._combo_seismogram_component.currentTextChanged.connect(self._on_seismogram_component_changed)
        gl_seis.addRow("Component:", self._combo_seismogram_component)
        grp_seis.setLayout(gl_seis)
        layout.addWidget(grp_seis)
        grp = QGroupBox("Z-X layers")
        gl = QGridLayout()
        self._chk_original = QCheckBox("Original model")
        self._chk_original.setChecked(True)
        self._chk_original.stateChanged.connect(self._update_canvas_layers)
        self._alpha_original = QDoubleSpinBox()
        self._alpha_original.setRange(0, 1)
        self._alpha_original.setSingleStep(0.1)
        self._alpha_original.setValue(1.0)
        self._alpha_original.valueChanged.connect(self._update_canvas_layers)
        gl.addWidget(self._chk_original, 0, 0)
        gl.addWidget(QLabel("α"), 0, 2)
        gl.addWidget(self._alpha_original, 0, 3)

        self._chk_survey = QCheckBox("Survey")
        self._chk_survey.setChecked(True)
        self._chk_survey.stateChanged.connect(self._update_canvas_layers)
        self._alpha_survey = QDoubleSpinBox()
        self._alpha_survey.setRange(0, 1)
        self._alpha_survey.setSingleStep(0.1)
        self._alpha_survey.setValue(1.0)
        self._alpha_survey.valueChanged.connect(self._update_canvas_layers)
        gl.addWidget(self._chk_survey, 1, 0)
        gl.addWidget(QLabel("α"), 1, 2)
        gl.addWidget(self._alpha_survey, 1, 3)

        self._chk_smoothed = QCheckBox("Smoothed model")
        self._chk_smoothed.setChecked(False)
        self._chk_smoothed.stateChanged.connect(self._update_canvas_layers)
        self._alpha_smoothed = QDoubleSpinBox()
        self._alpha_smoothed.setRange(0, 1)
        self._alpha_smoothed.setSingleStep(0.1)
        self._alpha_smoothed.setValue(1.0)
        self._alpha_smoothed.valueChanged.connect(self._update_canvas_layers)
        gl.addWidget(self._chk_smoothed, 2, 0)
        gl.addWidget(QLabel("α"), 2, 2)
        gl.addWidget(self._alpha_smoothed, 2, 3)

        self._chk_snapshots = QCheckBox("Snapshots")
        self._chk_snapshots.setChecked(False)
        self._chk_snapshots.stateChanged.connect(self._update_canvas_layers)
        self._snapshot_combo = QComboBox()
        self._snapshot_combo.currentTextChanged.connect(self._update_canvas_layers)
        self._alpha_snapshots = QDoubleSpinBox()
        self._alpha_snapshots.setRange(0, 1)
        self._alpha_snapshots.setSingleStep(0.1)
        self._alpha_snapshots.setValue(0.5)
        self._alpha_snapshots.valueChanged.connect(self._update_canvas_layers)
        gl.addWidget(self._chk_snapshots, 3, 0)
        gl.addWidget(self._snapshot_combo, 3, 1)
        gl.addWidget(QLabel("α"), 3, 2)
        gl.addWidget(self._alpha_snapshots, 3, 3)
        self._chk_image = QCheckBox("Image")
        self._chk_image.setChecked(False)
        self._chk_image.stateChanged.connect(self._update_canvas_layers)
        self._alpha_image = QDoubleSpinBox()
        self._alpha_image.setRange(0, 1)
        self._alpha_image.setSingleStep(0.1)
        self._alpha_image.setValue(0.8)
        self._alpha_image.valueChanged.connect(self._update_canvas_layers)
        gl.addWidget(self._chk_image, 4, 0)
        gl.addWidget(QLabel("α"), 4, 2)
        gl.addWidget(self._alpha_image, 4, 3)
        grp.setLayout(gl)
        layout.addWidget(grp)
        layout.addStretch()
        return panel

    def _on_seismogram_percentile_changed(self, value):
        self.seismogram_canvas.set_percentile(value)

    def _get_forward_run(self, name):
        for r in self._forward_runs:
            if r["name"] == name:
                return r
        return None

    def _get_backward_run(self, name):
        for r in self._backward_runs:
            if r["name"] == name:
                return r
        return None

    def _rebuild_effective_snapshots(self):
        """Собирает _snapshots из текущих fwd и bwd наборов для отображения."""
        merged = {}
        if self._current_fwd_name:
            r = self._get_forward_run(self._current_fwd_name)
            if r and r.get("snapshots"):
                merged.update(r["snapshots"])
        if self._current_bwd_name:
            r = self._get_backward_run(self._current_bwd_name)
            if r and r.get("snapshots"):
                merged.update(r["snapshots"])
        self._snapshots = merged if merged else None
        self._update_layer_availability()
        self._update_canvas_layers()

    def _apply_current_seismogram(self):
        """Устанавливает _seismogram_data/_t_ms и холст сейсмограммы из текущего набора."""
        if not self._current_seismogram_name:
            self._seismogram_data = None
            self._seismogram_t_ms = None
            self.seismogram_canvas.set_seismogram(None, None, [], self._receiver_layout, self._dx, self._dz)
            return
        r = self._get_forward_run(self._current_seismogram_name)
        comp = getattr(self, "_seismogram_component", "P") or "P"
        seis_map = (r.get("seismograms") or {}) if r else {}
        if r and seis_map.get(comp) is not None and r.get("seismogram_t_ms") is not None:
            self._seismogram_data = seis_map.get(comp)
            self._seismogram_t_ms = r["seismogram_t_ms"]
            self.seismogram_canvas.set_seismogram(
                self._seismogram_data,
                self._seismogram_t_ms,
                self._receivers,
                self._receiver_layout,
                self._dx,
                self._dz,
            )
        else:
            self._seismogram_data = None
            self._seismogram_t_ms = None
        self._update_layer_availability()

    def _on_seismogram_component_changed(self, comp):
        comp = (comp or "P").strip()
        if comp not in ("P", "Vz", "Vx"):
            comp = "P"
        self._seismogram_component = comp
        self._apply_current_seismogram()

    def _refresh_current_combos(self):
        """Заполняет комбобоксы Current из _forward_runs и _backward_runs."""
        fwd_names = [r["name"] for r in self._forward_runs]
        bwd_names = [r["name"] for r in self._backward_runs]
        if fwd_names and not self._current_fwd_name:
            self._current_fwd_name = fwd_names[0]
        if fwd_names and not self._current_seismogram_name:
            self._current_seismogram_name = fwd_names[0]
        if bwd_names and not self._current_bwd_name:
            self._current_bwd_name = bwd_names[0]
        self._combo_current_fwd.blockSignals(True)
        self._combo_current_fwd.clear()
        self._combo_current_fwd.addItems(fwd_names)
        idx = fwd_names.index(self._current_fwd_name) if (fwd_names and self._current_fwd_name in fwd_names) else 0
        self._combo_current_fwd.setCurrentIndex(min(idx, len(fwd_names) - 1) if fwd_names else -1)
        self._combo_current_fwd.blockSignals(False)
        self._combo_current_bwd.blockSignals(True)
        self._combo_current_bwd.clear()
        self._combo_current_bwd.addItems(bwd_names)
        idx = bwd_names.index(self._current_bwd_name) if (bwd_names and self._current_bwd_name in bwd_names) else 0
        self._combo_current_bwd.setCurrentIndex(min(idx, len(bwd_names) - 1) if bwd_names else -1)
        self._combo_current_bwd.blockSignals(False)
        self._combo_current_seismogram.blockSignals(True)
        self._combo_current_seismogram.clear()
        self._combo_current_seismogram.addItems(fwd_names)
        idx = fwd_names.index(self._current_seismogram_name) if (fwd_names and self._current_seismogram_name in fwd_names) else 0
        self._combo_current_seismogram.setCurrentIndex(min(idx, len(fwd_names) - 1) if fwd_names else -1)
        self._combo_current_seismogram.blockSignals(False)

    def _on_current_fwd_changed(self, name):
        if not name:
            return
        self._current_fwd_name = name
        self._rebuild_effective_snapshots()

    def _on_current_bwd_changed(self, name):
        if not name:
            return
        self._current_bwd_name = name
        self._rebuild_effective_snapshots()

    def _on_current_seismogram_changed(self, name):
        if not name:
            return
        self._current_seismogram_name = name
        self._apply_current_seismogram()

    def _update_layer_availability(self):
        self._refresh_current_combos()
        has_model = self._vp is not None and self._vp.size > 0
        has_survey = (self._source is not None) or len(self._receivers) > 0
        has_smoothed = has_model and self._smooth_size_m > 0
        has_snapshots = (
            self._snapshots is not None
            and any(self._snapshots.get(k) is not None for k in SNAPSHOT_COMPONENT_ORDER)
        )
        has_seismogram = len(self._forward_runs) > 0 and len(self._receivers) > 0
        src_comp = self._rtm_settings.get("source", "P")
        fwd_run = self._get_forward_run(self._current_fwd_name) if self._current_fwd_name else None
        bwd_run = self._get_backward_run(self._current_bwd_name) if self._current_bwd_name else None
        has_rtm_build = (
            len(self._forward_runs) > 0
            and len(self._backward_runs) > 0
            and fwd_run is not None
            and bwd_run is not None
            and (fwd_run.get("snapshots") or {}).get(src_comp + " fwd") is not None
            and (bwd_run.get("snapshots") or {}).get(src_comp + " bwd") is not None
        )
        has_rtm_image = self._rtm_image is not None and self._rtm_image.size > 0
        self._chk_original.setEnabled(has_model)
        self._alpha_original.setEnabled(has_model)
        self._chk_survey.setEnabled(has_survey)
        self._alpha_survey.setEnabled(has_survey)
        self._chk_smoothed.setEnabled(has_smoothed)
        self._alpha_smoothed.setEnabled(has_smoothed)
        self._chk_snapshots.setEnabled(has_snapshots)
        self._snapshot_combo.setEnabled(has_snapshots)
        self._alpha_snapshots.setEnabled(has_snapshots)
        self._snapshot_percentile_spin.setEnabled(has_snapshots)
        self._snapshot_slider_row.setVisible(has_snapshots)
        self._chk_image.setEnabled(has_rtm_image)
        self._alpha_image.setEnabled(has_rtm_image)
        self._image_percentile_spin.setEnabled(has_rtm_image)
        act_bwd = getattr(self, "_act_backward", None)
        if act_bwd is not None:
            act_bwd.setEnabled(has_seismogram)
        act_rtm = getattr(self, "_act_rtm_build", None)
        if act_rtm is not None:
            act_rtm.setEnabled(has_rtm_build)
        if has_snapshots:
            self._snapshot_combo.clear()
            for k in SNAPSHOT_COMPONENT_ORDER:
                if self._snapshots.get(k) is not None:
                    self._snapshot_combo.addItem(k)
            nt = 0
            for k in SNAPSHOT_COMPONENT_ORDER:
                arr = self._snapshots.get(k)
                if arr is not None and arr.ndim >= 1:
                    nt = max(nt, arr.shape[0])
            self._snapshot_slider.setMaximum(max(0, nt - 1))
            if self._snapshot_slider.value() >= nt:
                self._snapshot_slider.setValue(0)
        self._update_canvas_layers()

    def _update_memory_status(self):
        """Обновляет подпись памяти в статус-баре (процесс RSS и лимит при наличии psutil)."""
        mb = _get_process_memory_mb()
        limit_mb = _get_system_memory_limit_mb()
        if mb is not None:
            if limit_mb is not None:
                self._memory_label.setText("Memory: {:.0f} / {:.0f} MB".format(mb, limit_mb))
            else:
                self._memory_label.setText("Memory: {:.0f} MB".format(mb))
        else:
            self._memory_label.setText("Memory: —")

    def _maybe_reduce_memory(self):
        """Если использование памяти выше лимита — предлагает удалить самые старые наборы Forward/Backward."""
        mb = _get_process_memory_mb()
        limit_mb = _get_system_memory_limit_mb()
        if mb is None or limit_mb is None or mb <= limit_mb:
            return
        reply = QMessageBox.question(
            self,
            "Memory",
            "Memory usage ({:.0f} MB) exceeds the recommended limit ({:.0f} MB).\n"
            "Remove the oldest Forward and Backward runs to free memory?".format(mb, limit_mb),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        removed_any = False
        while len(self._forward_runs) > 1:
            mb = _get_process_memory_mb()
            if mb is None or mb <= limit_mb:
                break
            old_name = self._forward_runs[0]["name"]
            self._forward_runs.pop(0)
            if self._current_fwd_name == old_name:
                self._current_fwd_name = self._forward_runs[0]["name"] if self._forward_runs else None
            if self._current_seismogram_name == old_name:
                self._current_seismogram_name = self._forward_runs[0]["name"] if self._forward_runs else None
            removed_any = True
            gc.collect()
        while len(self._backward_runs) > 1:
            mb = _get_process_memory_mb()
            if mb is None or mb <= limit_mb:
                break
            old_name = self._backward_runs[0]["name"]
            self._backward_runs.pop(0)
            if self._current_bwd_name == old_name:
                self._current_bwd_name = self._backward_runs[0]["name"] if self._backward_runs else None
            removed_any = True
            gc.collect()
        if removed_any:
            self._rebuild_effective_snapshots()
            self._apply_current_seismogram()
            self._refresh_current_combos()
            self._update_layer_availability()
            self._update_memory_status()
            QMessageBox.information(self, "Memory", "Oldest runs have been removed.")

    def _update_canvas_layers(self):
        p_snap = self._snapshot_percentile_spin.value()
        snapshot_vmin = snapshot_vmax = None
        if self._snapshots is not None:
            comp = self._snapshot_combo.currentText()
            arr = self._snapshots.get(comp)
            if arr is not None and arr.size > 0:
                # Для H5 не грузим весь массив (OOM): берём выборку кадров для percentile
                if hasattr(arr, "read_full"):
                    arr_p = snapshot_io.sample_frames_for_percentile(arr._path, arr._dset_name)
                else:
                    arr_p = arr
                snapshot_vmin, snapshot_vmax = np.percentile(arr_p, [100 - p_snap, p_snap])
                if snapshot_vmax <= snapshot_vmin:
                    snapshot_vmax = snapshot_vmin + 1.0
        p_img = self._image_percentile_spin.value()
        image_vmin = image_vmax = None
        if self._rtm_image is not None and self._rtm_image.size > 0:
            image_vmin, image_vmax = np.percentile(self._rtm_image, [100 - p_img, p_img])
            if image_vmax <= image_vmin:
                image_vmax = image_vmin + 1.0
        self.canvas.set_rtm_image(self._rtm_image)
        self.canvas.set_layer(
            show_original=self._chk_original.isChecked(),
            alpha_original=self._alpha_original.value(),
            show_survey=self._chk_survey.isChecked(),
            alpha_survey=self._alpha_survey.value(),
            show_smoothed=self._chk_smoothed.isChecked(),
            alpha_smoothed=self._alpha_smoothed.value(),
            show_snapshots=self._chk_snapshots.isChecked(),
            alpha_snapshots=self._alpha_snapshots.value(),
            snapshot_vmin=snapshot_vmin,
            snapshot_vmax=snapshot_vmax,
            show_image=self._chk_image.isChecked(),
            alpha_image=self._alpha_image.value(),
            image_vmin=image_vmin,
            image_vmax=image_vmax,
        )
        if self._snapshots is not None:
            comp = self._snapshot_combo.currentText()
            arr = self._snapshots.get(comp)
            idx = self._snapshot_slider.value()
            if arr is not None and arr.ndim == 3 and 0 <= idx < arr.shape[0]:
                # arr: (n_save, nx, nz), холст ожидает (nz, nx)
                self.canvas.set_snapshot_2d(arr[idx].T)
            else:
                self.canvas.set_snapshot_2d(None)
        else:
            self.canvas.set_snapshot_2d(None)

    def _on_snapshot_slider(self, value):
        if self._snapshots is not None:
            comp = self._snapshot_combo.currentText()
            arr = self._snapshots.get(comp)
            if arr is not None and arr.ndim == 3 and 0 <= value < arr.shape[0]:
                self.canvas.set_snapshot_2d(arr[value].T)
        self._snapshot_time_label.setText(str(value))

    def _file_load_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Project", "", "Project (*.ini);;All (*)")
        if not path:
            return
        file_name = os.path.basename(path)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage("Loading project {}...".format(file_name))
        try:
            self._load_project_from_ini(path)
            self._project_path = path
            self.setWindowTitle("Sim1Shot2D — " + file_name)
        except Exception as e:
            import traceback
            QMessageBox.warning(
                self, "Load Project",
                "Failed to load project:\n\n{}".format(e))
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor()
            self.statusBar().clearMessage()

    def _file_save_project(self):
        if self._project_path is None:
            self._file_save_project_as()
            return
        try:
            self._save_project_to_ini(self._project_path)
        except Exception as e:
            QMessageBox.warning(
                self, "Save Project",
                "Failed to save project:\n\n{}".format(e))

    def _file_save_project_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", "", "Project (*.ini);;All (*)")
        if not path:
            return
        if not path.endswith(".ini"):
            path = path + ".ini"
        try:
            self._save_project_to_ini(path)
            self._project_path = path
            self.setWindowTitle("Sim1Shot2D — " + os.path.basename(path))
        except Exception as e:
            QMessageBox.warning(
                self, "Save Project As",
                "Failed to save project:\n\n{}".format(e))

    def _file_reset_project(self):
        """Сброс проекта в начальное состояние (как при запуске программы)."""
        reply = QMessageBox.question(
            self,
            "Reset Project",
            "Reset all data (model, survey, runs, RTM) to initial state? Unsaved changes will be lost.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        # Сброс буферов runs и визуализации
        self._forward_runs = []
        self._backward_runs = []
        self._current_fwd_name = None
        self._current_bwd_name = None
        self._current_seismogram_name = None
        self._snapshots = None
        self._seismogram_data = None
        self._seismogram_t_ms = None
        self._pending_forward_name = None
        self._pending_backward_name = None
        self._pending_backward_seismogram_name = None
        self._rtm_image = None
        self._rtm_image_base = None
        self.canvas.set_snapshot_2d(None)
        self.canvas.set_rtm_image(None)
        if hasattr(self, "seismogram_canvas") and self.seismogram_canvas is not None:
            self.seismogram_canvas.set_seismogram(None, None, [], "Profile", 1.0, 1.0)
        # Модель
        self._vp = None
        self._vp_full = None
        self._dx = self._dz = 1.0
        self._model_file_path = ""
        self._layered_model_spec = None
        self._crop_x_left = 0.0
        self._crop_x_right = 0.0
        self._crop_z_top = 0.0
        self._crop_z_bottom = 0.0
        self._diffractors = []
        self._smooth_size_m = 0.0
        # Съёмка
        self._source = None
        self._receivers = []
        self._receiver_layout = "Profile"
        # Параметры симуляции и RTM — к дефолтам
        self._sim_settings = {
            "tmax_ms": 1000, "npml": 50, "dt_ms": 0.5,
            "snapshot_dt_ms": 2, "laplacian": "4pt",
        }
        self._rtm_settings = {"source": "P"}
        self._rtm_postproc = {"laplacian_on": False, "laplacian_order": 4, "agc_on": False, "agc_window_z_m": 1000.0}
        self._seismogram_component = "P"
        self._project_path = None
        self._last_model_source = "Original"
        if hasattr(self, "_combo_seismogram_component"):
            self._combo_seismogram_component.blockSignals(True)
            self._combo_seismogram_component.setCurrentText("P")
            self._combo_seismogram_component.blockSignals(False)
        if hasattr(self, "seismogram_canvas"):
            self.seismogram_canvas.set_layout("Profile")
        self._apply_velocity_to_canvas()
        self._refresh_current_combos()
        self._apply_current_seismogram()
        self._rebuild_effective_snapshots()
        self._update_layer_availability()
        self._update_memory_status()
        self.setWindowTitle("Sim1Shot2D — Model & Survey")

    def _file_exit(self):
        QApplication.quit()

    def _export_snapshot_animation(self):
        """Export → Snapshot Animation: GIF из текущих снапшотов (те же, что на Z-X plane), без RTM."""
        if self._vp is None or self._vp.size == 0:
            QMessageBox.warning(self, "Export", "Load model first.")
            return
        comp = self._snapshot_combo.currentText()
        arr = (self._snapshots or {}).get(comp) if comp else None
        if arr is None:
            QMessageBox.warning(self, "Export", "No snapshot data. Select a snapshot component on Z-X plane and ensure Snapshots are available.")
            return
        n_save = arr.shape[0] if hasattr(arr, "shape") and getattr(arr, "ndim", 0) == 3 else 0
        if n_save == 0:
            QMessageBox.warning(self, "Export", "No snapshot frames.")
            return
        include_overlays_current = self._chk_original.isChecked() or self._chk_survey.isChecked() or self._chk_smoothed.isChecked()
        cw = self.canvas.size().width() if self.canvas else 0
        ch = self.canvas.size().height() if self.canvas else 0
        default_w = max(100, cw) if cw else 600
        default_h = max(100, ch) if ch else 400
        # Подхватываем последние использованные настройки (сеансовые)
        s = getattr(self, "_snapshot_gif_settings", None) or {}
        include_overlays_def = s.get("include_overlays", include_overlays_current)
        delay_def = s.get("delay_ms", 80)
        loop_def = s.get("loop", 0)
        frame_step_def = s.get("frame_step", 2)
        width_def = s.get("width_px", default_w)
        height_def = s.get("height_px", default_h)
        dlg = ExportSnapshotGifDialog(
            self,
            include_overlays=include_overlays_def,
            delay_ms=int(delay_def),
            loop=int(loop_def),
            frame_step=int(frame_step_def),
            default_width_px=int(width_def),
            default_height_px=int(height_def),
        )
        if dlg.exec_() != QDialog.Accepted:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save GIF", "", "GIF (*.gif);;All (*)")
        if not path:
            return
        if not path.endswith(".gif"):
            path = path + ".gif"
        try:
            from PIL import Image
        except ImportError:
            QMessageBox.warning(self, "Export", "Pillow (PIL) is required for GIF export. Install: pip install Pillow")
            return
        # Percentile для снапшотов (как на Z-X)
        if hasattr(arr, "read_full"):
            arr_p = snapshot_io.sample_frames_for_percentile(arr._path, arr._dset_name)
        else:
            arr_p = np.asarray(arr, dtype=np.float64)
        p_snap = self._snapshot_percentile_spin.value()
        snapshot_vmin, snapshot_vmax = np.percentile(arr_p, [100 - p_snap, p_snap])
        if snapshot_vmax <= snapshot_vmin:
            snapshot_vmax = snapshot_vmin + 1.0
        include_overlays = dlg.get_include_overlays()
        delay_ms = dlg.get_delay_ms()
        loop = dlg.get_loop()
        frame_step = dlg.get_frame_step()
        export_w, export_h = dlg.get_width_height_px()
        # Сохраняем настройки экспорта для следующего захода в меню
        self._snapshot_gif_settings = {
            "include_overlays": bool(include_overlays),
            "width_px": int(export_w),
            "height_px": int(export_h),
            "delay_ms": int(delay_ms),
            "loop": int(loop),
            "frame_step": int(frame_step),
        }
        dpi = 100
        figsize = (export_w / dpi, export_h / dpi)
        vp = np.asarray(self._vp, dtype=np.float64)
        smoothed_vp = None
        if self.canvas._smoothed_vp is not None and self.canvas._smoothed_vp.shape == vp.shape:
            smoothed_vp = np.asarray(self.canvas._smoothed_vp, dtype=np.float64)
        alpha_orig = self._alpha_original.value() if include_overlays else 0.0
        alpha_surv = self._alpha_survey.value() if include_overlays else 0.0
        alpha_smooth = self._alpha_smoothed.value() if include_overlays else 0.0
        alpha_snap = self._alpha_snapshots.value()
        snapshot_dt_ms = self._sim_settings.get("snapshot_dt_ms", 2.0)
        comp_short = comp.replace(" fwd", "").replace(" bwd", "")
        run_name = self._current_fwd_name if " fwd" in comp else self._current_bwd_name
        component_label = "{} {}".format(comp_short, run_name or ("fwd" if " fwd" in comp else "bwd"))
        frames = []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Быстрый путь: один холст, обновляем только снапшот и заголовок (значительное ускорение)
            ctx = _SnapshotGifRendererCtx(
                vp, self._dx, self._dz, self._receivers, self._source, smoothed_vp,
                include_overlays, alpha_orig, alpha_surv, alpha_smooth, alpha_snap,
                snapshot_vmin, snapshot_vmax, figsize=figsize, dpi=dpi,
            )
            indices = list(range(0, n_save, frame_step))
            for idx, i in enumerate(indices):
                self.statusBar().showMessage("Export frame {}/{}".format(idx + 1, len(indices)))
                QApplication.processEvents()
                time_ms = i * snapshot_dt_ms
                frame_2d = arr[i]
                if hasattr(frame_2d, "shape") and frame_2d.ndim == 2:
                    snapshot_2d = np.asarray(frame_2d, dtype=np.float64).T
                else:
                    snapshot_2d = np.asarray(frame_2d, dtype=np.float64)
                    if snapshot_2d.shape != vp.shape:
                        snapshot_2d = snapshot_2d.T
                rgb = ctx.render_frame(snapshot_2d, time_ms, component_label)
                frames.append(Image.fromarray(rgb))
            if not frames:
                QMessageBox.warning(self, "Export", "No frames to export.")
                return
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=delay_ms, loop=loop)
            self.statusBar().clearMessage()
            QMessageBox.information(self, "Export", "GIF saved: {} ({} frames).".format(path, len(frames)))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.statusBar().clearMessage()
            QMessageBox.warning(self, "Export", "Export failed:\n\n{}".format(e))
        finally:
            QApplication.restoreOverrideCursor()

    def _export_seismogram_segy(self):
        """Export → Seismogram SEG-Y: текущая сейсмограмма в SEG-Y с диалогом параметров."""
        if self._seismogram_data is None or self._seismogram_data.size == 0:
            QMessageBox.warning(self, "Export", "No seismogram data. Run Forward and select current seismogram.")
            return
        if not self._receivers:
            QMessageBox.warning(self, "Export", "No receivers defined.")
            return
        t_ms = self._seismogram_t_ms
        if t_ms is None or (hasattr(t_ms, "__len__") and len(t_ms) == 0):
            QMessageBox.warning(self, "Export", "No time axis for seismogram.")
            return
        # По умолчанию Max Time = Tmax из настроек симуляции (чтобы не терять последний сэмпл)
        tmax_sim = self._sim_settings.get("tmax_ms", 1000.0)
        max_from_data = float(np.max(t_ms)) if hasattr(t_ms, "__len__") else float(t_ms)
        max_time_default = max(tmax_sim, max_from_data)
        dt_default = self._sim_settings.get("snapshot_dt_ms", 2.0)
        dlg = ExportSeismogramSegyDialog(max_time_ms=max_time_default, dt_ms=dt_default, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Seismogram SEG-Y", "", "SEG-Y (*.sgy *.segy);;All (*)")
        if not path:
            return
        if not path.lower().endswith((".sgy", ".segy")):
            path = path + ".sgy"
        use_ieee = dlg.get_format_ieee()
        max_time_ms = dlg.get_max_time_ms()
        export_dt_ms = dlg.get_dt_ms()
        scaler = 100
        sx_byte = ExportSeismogramSegyDialog.SX_BYTE
        sz_byte = ExportSeismogramSegyDialog.SZ_BYTE
        rx_byte = ExportSeismogramSegyDialog.RX_BYTE
        rz_byte = ExportSeismogramSegyDialog.RZ_BYTE
        scaler_b1 = ExportSeismogramSegyDialog.SCALER_BYTE_1
        scaler_b2 = ExportSeismogramSegyDialog.SCALER_BYTE_2

        data = np.asarray(self._seismogram_data, dtype=np.float64)
        n_traces = data.shape[1] if data.ndim >= 2 else 1
        if data.ndim == 1:
            data = data[:, np.newaxis]
        n_orig = data.shape[0]
        orig_dt = (float(np.max(t_ms)) - float(np.min(t_ms))) / (n_orig - 1) if n_orig > 1 else dt_default

        n_samples = int(round(max_time_ms / export_dt_ms)) + 1
        if abs(export_dt_ms - orig_dt) < 1e-9 * max(export_dt_ms, orig_dt):
            export_data = np.zeros((n_samples, n_traces), dtype=np.float64)
            copy_n = min(n_samples, n_orig)
            export_data[:copy_n, :] = data[:copy_n, :]
            if copy_n < n_samples:
                export_data[copy_n:, :] = 0
            time_axis_ms = np.arange(n_samples, dtype=np.float64) * export_dt_ms
        else:
            time_axis_ms = np.arange(n_samples, dtype=np.float64) * export_dt_ms
            export_data = np.zeros((n_samples, n_traces), dtype=np.float64)
            for tr in range(n_traces):
                export_data[:, tr] = scipy_resample(data[:, tr], n_samples)

        spec = segyio.spec()
        spec.format = 5 if use_ieee else 1
        spec.sorting = 0
        spec.samples = np.arange(len(time_axis_ms), dtype=np.int32)
        spec.tracecount = n_traces

        try:
            with segyio.create(path, spec) as f:
                f.bin[segyio.BinField.Interval] = int(round(export_dt_ms * 1000))
                sx = self._source[0] if self._source else 0.0
                sz = self._source[1] if self._source else 0.0
                for i in range(n_traces):
                    f.trace[i] = np.asarray(export_data[:, i], dtype=np.float32)
                    h = f.header[i]
                    h[scaler_b1] = scaler
                    h[scaler_b2] = scaler
                    h[sx_byte] = int(round(sx * scaler))
                    h[sz_byte] = int(round(sz * scaler))
                    rx = self._receivers[i][0]
                    rz = self._receivers[i][1]
                    h[rx_byte] = int(round(rx * scaler))
                    h[rz_byte] = int(round(rz * scaler))
            QMessageBox.information(self, "Export", "Seismogram saved: {} ({} traces, {} samples).".format(path, n_traces, n_samples))
        except Exception as e:
            QMessageBox.warning(self, "Export", "Export failed:\n\n{}".format(e))

    def _save_project_to_ini(self, path):
        cfg = ConfigParser()
        cfg.add_section("model")
        base_dir = os.path.dirname(os.path.abspath(path))
        layered = getattr(self, "_layered_model_spec", None)
        model_file = getattr(self, "_model_file_path", "") or ""

        if layered:
            # Layered model хранится параметрами и регенерируется при загрузке проекта.
            # Взаимоисключающе с SEG-Y.
            cfg.set("model", "type", "layered")
            cfg.set("model", "file", "")
            nx = int(layered.get("nx", (self._vp_full.shape[1] if self._vp_full is not None else 0)))
            nz = int(layered.get("nz", (self._vp_full.shape[0] if self._vp_full is not None else 0)))
            cfg.set("model", "layered_nx", str(nx))
            cfg.set("model", "layered_nz", str(nz))
            cfg.set("model", "layered_dx", str(float(layered.get("dx", self._dx))))
            cfg.set("model", "layered_dz", str(float(layered.get("dz", self._dz))))
            ztops = list(layered.get("ztops", [0.0]))
            vels = list(layered.get("vels", [2000.0]))
            n_layers = max(1, min(len(ztops), len(vels)))
            cfg.set("model", "layered_layer_count", str(n_layers))
            for i in range(n_layers):
                cfg.set("model", "layered_{}_ztop".format(i), str(float(ztops[i])))
                cfg.set("model", "layered_{}_vel".format(i), str(float(vels[i])))
        else:
            if model_file and os.path.isabs(model_file):
                try:
                    model_file = os.path.relpath(model_file, base_dir)
                except ValueError:
                    pass
            cfg.set("model", "type", "segy" if model_file else "none")
            cfg.set("model", "file", model_file)
        cfg.set("model", "dx", str(self._dx))
        cfg.set("model", "dz", str(self._dz))
        cfg.set("model", "smooth_size_m", str(self._smooth_size_m))
        cfg.set("model", "crop_x_left", str(getattr(self, "_crop_x_left", 0)))
        cfg.set("model", "crop_x_right", str(getattr(self, "_crop_x_right", 0)))
        cfg.set("model", "crop_z_top", str(getattr(self, "_crop_z_top", 0)))
        cfg.set("model", "crop_z_bottom", str(getattr(self, "_crop_z_bottom", 0)))
        cfg.set("model", "diffractor_count", str(len(self._diffractors)))
        for i, d in enumerate(self._diffractors):
            cfg.set("model", "diffractor_{}_x".format(i), str(d["x"]))
            cfg.set("model", "diffractor_{}_z".format(i), str(d["z"]))
            cfg.set("model", "diffractor_{}_r".format(i), str(d["r"]))
            cfg.set("model", "diffractor_{}_v".format(i), str(d["v"]))

        cfg.add_section("survey")
        if self._source is not None:
            cfg.set("survey", "source_x", str(self._source[0]))
            cfg.set("survey", "source_z", str(self._source[1]))
            cfg.set("survey", "source_freq", str(self._source[2]))
        else:
            cfg.set("survey", "source_x", "0")
            cfg.set("survey", "source_z", "0")
            cfg.set("survey", "source_freq", "22")
        cfg.set("survey", "receiver_type", self._receiver_layout)
        if self._receivers:
            xs = [p[0] for p in self._receivers]
            zs = [p[1] for p in self._receivers]
            if self._receiver_layout == "Profile":
                cfg.set("survey", "profile_z", str(zs[0]))
                cfg.set("survey", "profile_x0", str(xs[0]))
                cfg.set("survey", "profile_nx", str(len(self._receivers)))
                if len(xs) > 1:
                    cfg.set("survey", "profile_xstep", str(xs[1] - xs[0]))
                else:
                    cfg.set("survey", "profile_xstep", "5")
            else:
                cfg.set("survey", "well_x", str(xs[0]))
                cfg.set("survey", "well_z0", str(zs[0]))
                cfg.set("survey", "well_nz", str(len(self._receivers)))
                if len(zs) > 1:
                    cfg.set("survey", "well_zstep", str(zs[1] - zs[0]))
                else:
                    cfg.set("survey", "well_zstep", "5")
        else:
            cfg.set("survey", "profile_z", "100")
            cfg.set("survey", "profile_x0", "0")
            cfg.set("survey", "profile_nx", "50")
            cfg.set("survey", "profile_xstep", "5")

        cfg.add_section("simulation")
        for k, v in self._sim_settings.items():
            cfg.set("simulation", k, str(v))

        base_dir = os.path.dirname(os.path.abspath(path))
        fwd_with_h5 = [r for r in (self._forward_runs or []) if r.get("snapshots_path")]
        bwd_with_h5 = [r for r in (self._backward_runs or []) if r.get("snapshots_path")]
        if fwd_with_h5 or bwd_with_h5:
            cfg.add_section("runs")
            cfg.set("runs", "seismogram_component", getattr(self, "_seismogram_component", "P") or "P")
            cfg.set("runs", "forward_count", str(len(fwd_with_h5)))
            for i, r in enumerate(fwd_with_h5):
                cfg.set("runs", "forward_{}_name".format(i), r["name"])
                rel = r["snapshots_path"]
                if os.path.isabs(rel):
                    try:
                        rel = os.path.relpath(rel, base_dir)
                    except ValueError:
                        rel = os.path.basename(rel)
                cfg.set("runs", "forward_{}_path".format(i), rel)
                # Сохраняем сейсмограммы в NPZ для быстрой загрузки при открытии проекта
                if r.get("seismograms") and r.get("seismogram_t_ms") is not None:
                    npz_path = _get_seismogram_npz_path(r["snapshots_path"])
                    try:
                        np.savez_compressed(
                            npz_path,
                            P=r["seismograms"]["P"],
                            Vz=r["seismograms"]["Vz"],
                            Vx=r["seismograms"]["Vx"],
                            t_ms=r["seismogram_t_ms"],
                        )
                    except Exception:
                        pass
            cfg.set("runs", "backward_count", str(len(bwd_with_h5)))
            for i, r in enumerate(bwd_with_h5):
                cfg.set("runs", "backward_{}_name".format(i), r["name"])
                rel = r["snapshots_path"]
                if os.path.isabs(rel):
                    try:
                        rel = os.path.relpath(rel, base_dir)
                    except ValueError:
                        rel = os.path.basename(rel)
                cfg.set("runs", "backward_{}_path".format(i), rel)
                cfg.set("runs", "backward_{}_seismogram_source".format(i), r.get("seismogram_source", ""))
            if self._current_fwd_name:
                cfg.set("runs", "current_fwd_name", self._current_fwd_name)
            if self._current_bwd_name:
                cfg.set("runs", "current_bwd_name", self._current_bwd_name)
            if self._current_seismogram_name:
                cfg.set("runs", "current_seismogram_name", self._current_seismogram_name)
        if self._rtm_image is not None and self._rtm_image.size > 0:
            if not cfg.has_section("runs"):
                cfg.add_section("runs")
            project_stem = os.path.splitext(os.path.basename(path))[0].strip() or "unsaved"
            snap_dir_name = "{}_files".format(project_stem)
            snap_dir = os.path.join(base_dir, snap_dir_name)
            os.makedirs(snap_dir, exist_ok=True)
            rtm_path = os.path.join(snap_dir, "rtm_image.npz")
            np.savez_compressed(rtm_path, image=self._rtm_image.astype(np.float64))
            cfg.set("runs", "rtm_file", os.path.join(snap_dir_name, "rtm_image.npz"))

        with open(path, "w", encoding="utf-8") as f:
            cfg.write(f)

    def _load_project_from_ini(self, path):
        cfg = ConfigParser()
        cfg.read(path, encoding="utf-8")
        base_dir = os.path.dirname(os.path.abspath(path))

        # Сброс буферов сейсмограмм, снапшотов и визуализации
        self._forward_runs = []
        self._backward_runs = []
        self._current_fwd_name = None
        self._current_bwd_name = None
        self._current_seismogram_name = None
        self._snapshots = None
        self._seismogram_data = None
        self._seismogram_t_ms = None
        self._rtm_image = None
        self._rtm_image_base = None
        self.canvas.set_snapshot_2d(None)
        self.canvas.set_rtm_image(None)
        if hasattr(self, "seismogram_canvas") and self.seismogram_canvas is not None:
            self.seismogram_canvas.set_seismogram(None, None, [], getattr(self, "_receiver_layout", "Profile"), self._dx, self._dz)

        if cfg.has_section("model"):
            # Сброс модели, чтобы при отсутствии/ошибке файла не показывать старую модель с чужим кропом
            self._vp_full = None
            self._vp = None
            self._model_file_path = None
            self._layered_model_spec = None
            model_type = cfg.get("model", "type", fallback="").strip().lower()
            model_file = cfg.get("model", "file", fallback="").strip()
            has_layered_keys = cfg.has_option("model", "layered_layer_count") or cfg.has_option("model", "layered_nx")
            if model_type == "layered" or (not model_file and has_layered_keys):
                nx = cfg.getint("model", "layered_nx", fallback=201)
                nz = cfg.getint("model", "layered_nz", fallback=201)
                dx_l = cfg.getfloat("model", "layered_dx", fallback=cfg.getfloat("model", "dx", fallback=self._dx))
                dz_l = cfg.getfloat("model", "layered_dz", fallback=cfg.getfloat("model", "dz", fallback=self._dz))
                n_layers = max(1, cfg.getint("model", "layered_layer_count", fallback=1))
                ztops = []
                vels = []
                for i in range(n_layers):
                    ztops.append(cfg.getfloat("model", "layered_{}_ztop".format(i), fallback=0.0))
                    vels.append(cfg.getfloat("model", "layered_{}_vel".format(i), fallback=2000.0))
                # generate vp
                z_arr = np.arange(int(nz), dtype=np.float64) * float(dz_l)
                vp_col = np.zeros((int(nz),), dtype=np.float64)
                for i in range(len(ztops)):
                    z0 = float(ztops[i])
                    z1 = float(ztops[i + 1]) if i + 1 < len(ztops) else float("inf")
                    mask = (z_arr >= z0) & (z_arr < z1)
                    vp_col[mask] = float(vels[i])
                if np.any(vp_col == 0):
                    last_v = float(vels[-1]) if vels else 2000.0
                    vp_col[vp_col == 0] = last_v
                vp = np.repeat(vp_col[:, None], int(nx), axis=1)
                self._vp_full = np.asarray(vp, dtype=np.float64, copy=True)
                self._vp = self._vp_full
                self._model_file_path = ""
                self._layered_model_spec = {
                    "nx": int(nx), "nz": int(nz), "dx": float(dx_l), "dz": float(dz_l),
                    "ztops": list(map(float, ztops)), "vels": list(map(float, vels)),
                }
                # Зафиксировать sampling из layered, но далее ниже всё равно применится model/dx,dz
                self._dx = float(dx_l)
                self._dz = float(dz_l)
            else:
                if model_file and not os.path.isabs(model_file):
                    model_file = os.path.normpath(os.path.join(base_dir, model_file))
                if model_file and os.path.isfile(model_file):
                    vp, dx_load, dz_load = load_velocity_from_segy(model_file)
                    if vp is not None:
                        self._vp_full = np.asarray(vp, dtype=np.float64, copy=True)
                        self._vp = self._vp_full
                        self._model_file_path = os.path.abspath(model_file)
            self._dx = cfg.getfloat("model", "dx", fallback=self._dx)
            self._dz = cfg.getfloat("model", "dz", fallback=self._dz)
            self._smooth_size_m = cfg.getfloat("model", "smooth_size_m", fallback=0.0)
            self._crop_x_left = cfg.getfloat("model", "crop_x_left", fallback=0.0)
            self._crop_x_right = cfg.getfloat("model", "crop_x_right", fallback=0.0)
            self._crop_z_top = cfg.getfloat("model", "crop_z_top", fallback=0.0)
            self._crop_z_bottom = cfg.getfloat("model", "crop_z_bottom", fallback=0.0)
            # Отладка: значения кропа из project-файла
            print("[Crop load] project file crop: crop_x_left={}, crop_x_right={}, crop_z_top={}, crop_z_bottom={}".format(
                self._crop_x_left, self._crop_x_right, self._crop_z_top, self._crop_z_bottom))
            n_diff = cfg.getint("model", "diffractor_count", fallback=0)
            self._diffractors = []
            for i in range(n_diff):
                self._diffractors.append({
                    "x": cfg.getfloat("model", "diffractor_{}_x".format(i)),
                    "z": cfg.getfloat("model", "diffractor_{}_z".format(i)),
                    "r": cfg.getfloat("model", "diffractor_{}_r".format(i)),
                    "v": cfg.getfloat("model", "diffractor_{}_v".format(i)),
                })
            # Применить кроп сразу после загрузки модели (координаты survey уже в обрезанной системе)
            if self._vp_full is not None:
                nz_full, nx_full = self._vp_full.shape
                dx, dz = self._dx, self._dz
                if dx <= 0:
                    dx = 1.0
                if dz <= 0:
                    dz = 1.0
                ix0 = max(0, min(nx_full - 1, int(round(self._crop_x_left / dx))))
                ix1 = max(ix0, min(nx_full - 1, nx_full - 1 - int(round(self._crop_x_right / dx))))
                j0 = max(0, min(nz_full - 1, int(round(self._crop_z_top / dz))))
                j1 = max(j0, min(nz_full - 1, nz_full - 1 - int(round(self._crop_z_bottom / dz))))
                # Отладка: размеры и координаты оригинальной модели
                x_min_full = 0.0
                x_max_full = (nx_full - 1) * dx
                z_min_full = 0.0
                z_max_full = (nz_full - 1) * dz
                print("[Crop load] original model: nx={}, nz={}; x=[{}, {}] m, z=[{}, {}] m; dx={}, dz={}".format(
                    nx_full, nz_full, x_min_full, x_max_full, z_min_full, z_max_full, dx, dz))
                # Отладка: индексы и размеры после кропа
                nz_crop = j1 - j0 + 1
                nx_crop = ix1 - ix0 + 1
                x_min_crop = ix0 * dx
                x_max_crop = ix1 * dx
                z_min_crop = j0 * dz
                z_max_crop = j1 * dz
                print("[Crop load] after crop: ix0={}, ix1={}, j0={}, j1={}; nx={}, nz={}; x=[{}, {}] m, z=[{}, {}] m".format(
                    ix0, ix1, j0, j1, nx_crop, nz_crop, x_min_crop, x_max_crop, z_min_crop, z_max_crop))
                self._vp = np.asarray(self._vp_full[j0 : j1 + 1, ix0 : ix1 + 1], dtype=np.float64, copy=True)

        if cfg.has_section("survey"):
            sx = cfg.getfloat("survey", "source_x", fallback=0)
            sz = cfg.getfloat("survey", "source_z", fallback=0)
            sfreq = cfg.getfloat("survey", "source_freq", fallback=22)
            self._source = (sx, sz, sfreq)
            self._receiver_layout = cfg.get("survey", "receiver_type", fallback="Profile").strip()
            if self._receiver_layout not in ("Profile", "Well"):
                self._receiver_layout = "Profile"
            if self._receiver_layout == "Profile":
                pz = cfg.getfloat("survey", "profile_z", fallback=100)
                x0 = cfg.getfloat("survey", "profile_x0", fallback=0)
                nx = cfg.getint("survey", "profile_nx", fallback=50)
                xstep = cfg.getfloat("survey", "profile_xstep", fallback=5)
                self._receivers = [(x0 + i * xstep, pz) for i in range(nx)]
            else:
                wx = cfg.getfloat("survey", "well_x", fallback=500)
                z0 = cfg.getfloat("survey", "well_z0", fallback=0)
                nz = cfg.getint("survey", "well_nz", fallback=100)
                zstep = cfg.getfloat("survey", "well_zstep", fallback=5)
                self._receivers = [(wx, z0 + i * zstep) for i in range(nz)]
            self.seismogram_canvas.set_layout(self._receiver_layout)

        if cfg.has_section("simulation"):
            self._sim_settings["tmax_ms"] = cfg.getfloat("simulation", "tmax_ms", fallback=self._sim_settings["tmax_ms"])
            self._sim_settings["npml"] = cfg.getint("simulation", "npml", fallback=self._sim_settings["npml"])
            self._sim_settings["dt_ms"] = cfg.getfloat("simulation", "dt_ms", fallback=self._sim_settings["dt_ms"])
            self._sim_settings["snapshot_dt_ms"] = cfg.getfloat("simulation", "snapshot_dt_ms", fallback=self._sim_settings["snapshot_dt_ms"])
            self._sim_settings["laplacian"] = cfg.get("simulation", "laplacian", fallback=self._sim_settings["laplacian"]).strip()
            self._last_model_source = cfg.get("simulation", "model_source", fallback="Original").strip()

        # Подгрузка насчитанных снапшотов/сейсмограмм и RTM из проекта (H5 — только ссылки, первый кадр по требованию)
        if cfg.has_section("runs"):
            self._seismogram_component = cfg.get("runs", "seismogram_component", fallback=getattr(self, "_seismogram_component", "P") or "P").strip() or "P"
            if self._seismogram_component not in ("P", "Vz", "Vx"):
                self._seismogram_component = "P"
            if hasattr(self, "_combo_seismogram_component"):
                self._combo_seismogram_component.blockSignals(True)
                self._combo_seismogram_component.setCurrentText(self._seismogram_component)
                self._combo_seismogram_component.blockSignals(False)
            n_fwd = cfg.getint("runs", "forward_count", fallback=0)
            for i in range(n_fwd):
                name = cfg.get("runs", "forward_{}_name".format(i), fallback="")
                rel_path = cfg.get("runs", "forward_{}_path".format(i), fallback="").strip()
                if not name or not rel_path:
                    continue
                full_path = os.path.normpath(os.path.join(base_dir, rel_path)) if not os.path.isabs(rel_path) else rel_path
                if not os.path.isfile(full_path):
                    continue
                seismograms = {}
                seismogram_t_ms = None
                npz_path = _get_seismogram_npz_path(full_path)
                if os.path.isfile(npz_path):
                    try:
                        z = np.load(npz_path, allow_pickle=False)
                        seismograms["P"] = z["P"]
                        seismograms["Vz"] = z["Vz"]
                        seismograms["Vx"] = z["Vx"]
                        seismogram_t_ms = z["t_ms"]
                    except Exception:
                        pass
                if not seismograms or seismogram_t_ms is None:
                    for comp in ("P", "Vz", "Vx"):
                        data_c, t_c = snapshot_io.compute_seismogram_from_h5(
                            full_path, self._receivers, self._dx, self._dz,
                            snapshot_dt_ms=self._sim_settings.get("snapshot_dt_ms", 2.0),
                            component=comp,
                        )
                        seismograms[comp] = data_c
                        if seismogram_t_ms is None:
                            seismogram_t_ms = t_c
                snapshots = snapshot_io.snapshots_dict_from_h5(full_path, "forward")
                self._forward_runs.append({
                    "name": name,
                    "snapshots_path": full_path,
                    "snapshots": snapshots,
                    "seismograms": seismograms,
                    "seismogram_t_ms": seismogram_t_ms,
                })
            n_bwd = cfg.getint("runs", "backward_count", fallback=0)
            for i in range(n_bwd):
                name = cfg.get("runs", "backward_{}_name".format(i), fallback="")
                rel_path = cfg.get("runs", "backward_{}_path".format(i), fallback="").strip()
                seis_src = cfg.get("runs", "backward_{}_seismogram_source".format(i), fallback="")
                if not name or not rel_path:
                    continue
                full_path = os.path.normpath(os.path.join(base_dir, rel_path)) if not os.path.isabs(rel_path) else rel_path
                if not os.path.isfile(full_path):
                    continue
                snapshots = snapshot_io.snapshots_dict_from_h5(full_path, "backward")
                self._backward_runs.append({
                    "name": name,
                    "snapshots_path": full_path,
                    "snapshots": snapshots,
                    "seismogram_source": seis_src,
                })
            self._current_fwd_name = cfg.get("runs", "current_fwd_name", fallback="").strip() or (self._forward_runs[0]["name"] if self._forward_runs else None)
            self._current_bwd_name = cfg.get("runs", "current_bwd_name", fallback="").strip() or (self._backward_runs[0]["name"] if self._backward_runs else None)
            self._current_seismogram_name = cfg.get("runs", "current_seismogram_name", fallback="").strip() or (self._forward_runs[0]["name"] if self._forward_runs else None)
            rtm_rel = cfg.get("runs", "rtm_file", fallback="").strip()
            if rtm_rel:
                rtm_full = os.path.normpath(os.path.join(base_dir, rtm_rel))
                if os.path.isfile(rtm_full):
                    try:
                        data = np.load(rtm_full, allow_pickle=False)
                        if hasattr(data, "files") and "image" in data.files:
                            img = np.asarray(data["image"], dtype=np.float64)
                            data.close()
                        else:
                            img = np.asarray(data, dtype=np.float64)
                        self._rtm_image = img
                        self._rtm_image_base = self._rtm_image.copy()
                    except Exception:
                        pass

        self._apply_velocity_to_canvas()
        self._refresh_current_combos()
        self._apply_current_seismogram()
        self._rebuild_effective_snapshots()
        self._update_layer_availability()
        self._update_memory_status()

    def _model_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open velocity model (SEG-Y)", "", "SEG-Y (*.sgy *.segy);;All (*)")
        if not path:
            return
        try:
            vp, dx, dz = load_velocity_from_segy(path)
        except Exception as e:
            import traceback
            import sys
            print("Error loading model from file:", file=sys.stderr)
            print(str(e), file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Model load error")
            msg.setText("Failed to load model from file.")
            msg.setInformativeText(str(e))
            msg.setDetailedText(traceback.format_exc())
            msg.exec_()
            return
        if vp is None:
            QMessageBox.warning(self, "Error", "Failed to load model from file.")
            return
        self._vp_full = np.asarray(vp, dtype=np.float64, copy=True)
        self._vp = self._vp_full
        self._dx, self._dz = dx, dz
        self._model_file_path = path
        # SEG-Y и layered — взаимоисключающие
        self._layered_model_spec = None
        self._crop_x_left = self._crop_x_right = 0.0
        self._crop_z_top = self._crop_z_bottom = 0.0
        self._apply_velocity_to_canvas()

    def _model_create_layered(self):
        dlg = LayeredModelDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        res = dlg.get_result() or {}
        nx = int(res.get("nx", 201))
        nz = int(res.get("nz", 201))
        dx = float(res.get("dx", 5.0))
        dz = float(res.get("dz", 5.0))
        ztops = list(res.get("ztops", [0.0]))
        vels = list(res.get("vels", [2000.0]))

        has_model = self._vp_full is not None and getattr(self._vp_full, "size", 0) > 0
        has_runs = len(self._forward_runs) > 0 or len(self._backward_runs) > 0
        has_rtm = self._rtm_image is not None and getattr(self._rtm_image, "size", 0) > 0
        if has_model or has_runs or has_rtm:
            reply = QMessageBox.warning(
                self,
                "Create Layered",
                "Existing model, snapshots, seismograms and RTM will be removed. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        # build layered vp (nz, nx)
        z_arr = np.arange(nz, dtype=np.float64) * dz
        vp_col = np.zeros((nz,), dtype=np.float64)
        for i in range(len(ztops)):
            z0 = float(ztops[i])
            z1 = float(ztops[i + 1]) if i + 1 < len(ztops) else float("inf")
            mask = (z_arr >= z0) & (z_arr < z1)
            vp_col[mask] = float(vels[i])
        if np.any(vp_col == 0):
            # fallback: fill any gaps with last velocity
            last_v = float(vels[-1]) if vels else 2000.0
            vp_col[vp_col == 0] = last_v
        vp = np.repeat(vp_col[:, None], nx, axis=1)

        self._vp_full = np.asarray(vp, dtype=np.float64, copy=True)
        self._vp = self._vp_full
        self._dx, self._dz = dx, dz
        self._model_file_path = ""
        # SEG-Y и layered — взаимоисключающие
        self._layered_model_spec = {"nx": nx, "nz": nz, "dx": dx, "dz": dz, "ztops": list(ztops), "vels": list(vels)}
        self._crop_x_left = self._crop_x_right = 0.0
        self._crop_z_top = self._crop_z_bottom = 0.0
        # Model-related objects: reset to be safe for new extents
        self._diffractors = []
        self._source = None
        self._receivers = []

        # Reset runs/visualization buffers (same as project load)
        self._forward_runs = []
        self._backward_runs = []
        self._current_fwd_name = None
        self._current_bwd_name = None
        self._current_seismogram_name = None
        self._snapshots = None
        self._seismogram_data = None
        self._seismogram_t_ms = None
        self._rtm_image = None
        self._rtm_image_base = None
        self.canvas.set_snapshot_2d(None)
        self.canvas.set_rtm_image(None)
        if hasattr(self, "seismogram_canvas") and self.seismogram_canvas is not None:
            self.seismogram_canvas.set_seismogram(
                None, None, [], getattr(self, "_receiver_layout", "Profile"), self._dx, self._dz
            )

        self._apply_velocity_to_canvas()
        self._refresh_current_combos()
        self._apply_current_seismogram()
        self._rebuild_effective_snapshots()
        self._update_layer_availability()
        self._update_memory_status()

    def _apply_velocity_to_canvas(self):
        vp_display = self._vp
        if vp_display is not None and self._diffractors:
            vp_display = np.array(vp_display, dtype=np.float64, copy=True)
            nz, nx = vp_display.shape
            for d in self._diffractors:
                x, z, r, v = d["x"], d["z"], d["r"], d["v"]
                ix_c = int(round(x / self._dx))
                iz_c = int(round(z / self._dz))
                nr = max(1, int(round(r / min(self._dx, self._dz))))
                for di in range(-nr, nr + 1):
                    for dj in range(-nr, nr + 1):
                        ii = iz_c + di
                        jj = ix_c + dj
                        if 0 <= ii < nz and 0 <= jj < nx:
                            dist = np.sqrt((di * self._dz) ** 2 + (dj * self._dx) ** 2)
                            if dist <= r:
                                vp_display[ii, jj] = v
        self.canvas.set_velocity(vp_display, self._dx, self._dz)
        if self._source:
            self.canvas.set_source(self._source[0], self._source[1], self._source[2])
        else:
            self.canvas.set_source(None, None)
        self.canvas.set_receivers(self._receivers)
        self.canvas.set_diffractors(self._diffractors)
        # Сглаженный слой — только по загруженной модели (без дифракторов).
        vp_base = self._vp
        if self._smooth_size_m > 0 and vp_base is not None:
            smoothed = prepare_migration_velocity(
                np.array(vp_base.T, dtype=np.float64, copy=True),
                self._smooth_size_m, self._dx, self._dz
            ).T
            self.canvas.set_smoothed_vp(np.array(smoothed, dtype=np.float64, copy=True))
        else:
            self.canvas.set_smoothed_vp(None)
        self._update_layer_availability()
        self._update_canvas_layers()

    def _model_parameters(self):
        has_model = self._vp is not None and self._vp.size > 0
        dlg = ModelParametersDialog(self._dx, self._dz, has_model=has_model, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return
        has_runs = len(self._forward_runs) > 0 or len(self._backward_runs) > 0
        has_rtm = self._rtm_image is not None and self._rtm_image.size > 0
        if has_runs or has_rtm:
            reply = QMessageBox.warning(
                self,
                "Model Sampling",
                "Snapshots, seismograms and RTM will be removed. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        dx_new, dz_new = dlg.get_dx_dz()
        resample = has_model and dlg.get_resample()
        if resample and self._vp is not None and self._vp_full is not None:
            extent_x = (self._vp.shape[1] - 1) * self._dx
            extent_z = (self._vp.shape[0] - 1) * self._dz
            nx_new = max(2, int(round(extent_x / dx_new)) + 1)
            nz_new = max(2, int(round(extent_z / dz_new)) + 1)
            x_new_arr = np.linspace(0, extent_x, nx_new)
            z_new_arr = np.linspace(0, extent_z, nz_new)
            for arr, name in ((self._vp_full, "_vp_full"), (self._vp, "_vp")):
                arr = np.asarray(arr, dtype=np.float64)
                nz_old, nx_old = arr.shape
                x_old = np.linspace(0, extent_x, nx_old)
                z_old = np.linspace(0, extent_z, nz_old)
                interp = RegularGridInterpolator(
                    (x_old, z_old),
                    arr.T,
                    method="linear",
                    bounds_error=False,
                    fill_value=None,
                )
                xx, zz = np.meshgrid(x_new_arr, z_new_arr, indexing="ij")
                pts = np.column_stack([xx.ravel(), zz.ravel()])
                resampled = interp(pts).reshape(nx_new, nz_new).T
                setattr(self, name, np.asarray(resampled, dtype=np.float64))
            self._dx, self._dz = dx_new, dz_new
        else:
            self._dx, self._dz = dx_new, dz_new
        self._forward_runs = []
        self._backward_runs = []
        self._current_fwd_name = None
        self._current_bwd_name = None
        self._current_seismogram_name = None
        self._snapshots = None
        self._seismogram_data = None
        self._seismogram_t_ms = None
        self._rtm_image = None
        self._rtm_image_base = None
        self._apply_velocity_to_canvas()
        self._refresh_current_combos()
        self._apply_current_seismogram()
        self._rebuild_effective_snapshots()
        self._update_layer_availability()
        self._update_memory_status()

    def _model_diffractors(self):
        dlg = DiffractorsDialog(self._diffractors, self)
        dlg.exec_()
        self._diffractors = dlg.get_diffractors()
        self._apply_velocity_to_canvas()

    def _model_smooth(self):
        dlg = SmoothDialog(self._smooth_size_m, self)
        if dlg.exec_() == QDialog.Accepted:
            self._smooth_size_m = dlg.get_smooth_size_m()
            self._apply_velocity_to_canvas()

    def _model_crop(self):
        if self._vp_full is None or self._vp_full.size == 0:
            QMessageBox.warning(self, "Crop", "Load model first (Model → Load SEG-Y).")
            return
        dlg = CropDialog(
            self._crop_x_left, self._crop_x_right,
            self._crop_z_top, self._crop_z_bottom,
            self,
        )
        if dlg.exec_() != QDialog.Accepted:
            return
        x_left, x_right, z_top, z_bottom = dlg.get_crop_m()
        if x_left == 0 and x_right == 0 and z_top == 0 and z_bottom == 0:
            return
        has_runs = len(self._forward_runs) > 0 or len(self._backward_runs) > 0
        has_rtm = self._rtm_image is not None and self._rtm_image.size > 0
        if has_runs or has_rtm:
            reply = QMessageBox.warning(
                self,
                "Crop",
                "Snapshots, seismograms and RTM will be removed. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        nz_full, nx_full = self._vp_full.shape
        dx, dz = self._dx, self._dz
        ix0 = int(round(x_left / dx))
        ix1 = nx_full - 1 - int(round(x_right / dx))
        j0 = int(round(z_top / dz))
        j1 = nz_full - 1 - int(round(z_bottom / dz))
        if ix0 > ix1 or j0 > j1:
            QMessageBox.warning(
                self, "Crop",
                "Invalid crop: would leave no model. Reduce crop values.",
            )
            return
        self._vp = np.asarray(self._vp_full[j0 : j1 + 1, ix0 : ix1 + 1], dtype=np.float64, copy=True)
        new_shift_x = ix0 * dx
        new_shift_z = j0 * dz
        prev_ix0 = int(round(self._crop_x_left / dx))
        prev_j0 = int(round(self._crop_z_top / dz))
        prev_shift_x = prev_ix0 * dx
        prev_shift_z = prev_j0 * dz
        nx_new, nz_new = ix1 - ix0 + 1, j1 - j0 + 1
        x_max = (nx_new - 1) * dx
        z_max = (nz_new - 1) * dz
        if self._source is not None:
            sx, sz, freq = self._source[0], self._source[1], self._source[2]
            sx_new = sx + prev_shift_x - new_shift_x
            sz_new = sz + prev_shift_z - new_shift_z
            if 0 <= sx_new <= x_max and 0 <= sz_new <= z_max:
                self._source = (sx_new, sz_new, freq)
            else:
                self._source = None
        self._receivers = [
            (x + prev_shift_x - new_shift_x, z + prev_shift_z - new_shift_z)
            for x, z in self._receivers
            if 0 <= (x + prev_shift_x - new_shift_x) <= x_max and 0 <= (z + prev_shift_z - new_shift_z) <= z_max
        ]
        self._diffractors = [
            {**d, "x": d["x"] + prev_shift_x - new_shift_x, "z": d["z"] + prev_shift_z - new_shift_z}
            for d in self._diffractors
            if 0 <= (d["x"] + prev_shift_x - new_shift_x) <= x_max and 0 <= (d["z"] + prev_shift_z - new_shift_z) <= z_max
        ]
        self._crop_x_left = x_left
        self._crop_x_right = x_right
        self._crop_z_top = z_top
        self._crop_z_bottom = z_bottom
        self._forward_runs = []
        self._backward_runs = []
        self._current_fwd_name = None
        self._current_bwd_name = None
        self._current_seismogram_name = None
        self._snapshots = None
        self._seismogram_data = None
        self._seismogram_t_ms = None
        self._rtm_image = None
        self._rtm_image_base = None
        self._apply_velocity_to_canvas()
        self._refresh_current_combos()
        self._apply_current_seismogram()
        self._rebuild_effective_snapshots()
        self._update_layer_availability()
        self._update_memory_status()
        QMessageBox.information(self, "Crop", "Model cropped. NX={}, NZ={}.".format(nx_new, nz_new))

    def apply_diffractors_and_redraw(self, diffractors):
        self._diffractors = list(diffractors)
        self._apply_velocity_to_canvas()

    def _survey_source(self):
        model_xmax = model_zmax = None
        model_dz = None
        if self._vp is not None and getattr(self._vp, "size", 0) > 0:
            nz_m, nx_m = self._vp.shape
            model_xmax = (nx_m - 1) * float(self._dx)
            model_zmax = (nz_m - 1) * float(self._dz)
            model_dz = float(self._dz)
        # Defaults: X = 0, Z = model dz (surface) if available, otherwise 0
        x, z, freq = 0.0, (model_dz if model_dz is not None else 0.0), 22.0
        # If source is already defined, reuse its last position and frequency
        old_source = self._source
        if old_source is not None:
            x, z, freq = old_source[0], old_source[1], old_source[2]
        dlg = SourceDialog(x, z, freq, self, model_xmax=model_xmax, model_zmax=model_zmax, model_dz=model_dz)
        if dlg.exec_() == QDialog.Accepted:
            new_source = dlg.get_params()
            # If source position changes and there are simulation results, warn and optionally clear them.
            if old_source is not None and (
                abs(new_source[0] - old_source[0]) > 1e-9
                or abs(new_source[1] - old_source[1]) > 1e-9
                or abs(new_source[2] - old_source[2]) > 1e-9
            ):
                has_runs = len(self._forward_runs) > 0 or len(self._backward_runs) > 0
                has_rtm = self._rtm_image is not None and getattr(self._rtm_image, "size", 0) > 0
                if has_runs or has_rtm:
                    reply = QMessageBox.warning(
                        self,
                        "Survey — Source",
                        "Changing the source position will remove all Forward/Backward runs,\n"
                        "seismograms and RTM image. Continue?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No,
                    )
                    if reply != QMessageBox.Yes:
                        return
                    # Clear runs, seismograms and RTM (same reset logic as in Crop)
                    self._forward_runs = []
                    self._backward_runs = []
                    self._current_fwd_name = None
                    self._current_bwd_name = None
                    self._current_seismogram_name = None
                    self._snapshots = None
                    self._seismogram_data = None
                    self._seismogram_t_ms = None
                    self._rtm_image = None
                    self._rtm_image_base = None
            self._source = new_source
            self._apply_velocity_to_canvas()
            # Refresh combos/seismograms/snapshots state if we have just cleared runs
            self._refresh_current_combos()
            self._apply_current_seismogram()
            self._rebuild_effective_snapshots()
            self._update_layer_availability()
            self._update_memory_status()

    def _survey_receivers(self):
        # Ограничения приёмников: не выходить за пределы текущей модели
        model_xmax = model_zmax = None
        if self._vp is not None and getattr(self._vp, "size", 0) > 0:
            nz_m, nx_m = self._vp.shape
            model_xmax = (nx_m - 1) * float(self._dx)
            model_zmax = (nz_m - 1) * float(self._dz)
        dlg = ReceiversDialog(
            receivers=self._receivers,
            layout_name=self._receiver_layout,
            parent=self,
            model_xmax=model_xmax,
            model_zmax=model_zmax,
        )
        if dlg.exec_() == QDialog.Accepted:
            self._receivers = dlg.get_receiver_points()
            self._receiver_layout = dlg.get_layout()
            self.seismogram_canvas.set_layout(self._receiver_layout)
            self._apply_velocity_to_canvas()
            # Пересобрать сейсмограммы всех forward-наборов по новой конфигурации приёмников
            for r in self._forward_runs:
                snaps = r.get("snapshots") or {}
                seis = {}
                t_ms = None
                for comp in ("P", "Vz", "Vx"):
                    arr = snaps.get(comp + " fwd")
                    if arr is not None:
                        data_c, t_c = self._compute_seismogram_from_snapshots(arr)
                        if t_ms is None:
                            t_ms = t_c
                        seis[comp] = data_c
                if seis:
                    r["seismograms"] = seis
                    r["seismogram_t_ms"] = t_ms
            self._apply_current_seismogram()

    def _simulation_settings(self):
        dlg = SimulationSettingsDialog(
            self,
            tmax_ms=self._sim_settings["tmax_ms"],
            npml=self._sim_settings["npml"],
            dt_ms=self._sim_settings["dt_ms"],
            snapshot_dt_ms=self._sim_settings["snapshot_dt_ms"],
            laplacian=self._sim_settings["laplacian"],
        )
        if dlg.exec_() == QDialog.Accepted:
            self._sim_settings = dlg.get_params()

    def _simulation_run_forward(self):
        if self._vp is None or self._vp.size == 0:
            QMessageBox.warning(self, "Run Forward", "Load model first (Model → Load SEG-Y).")
            return
        if self._source is None:
            QMessageBox.warning(self, "Run Forward", "Set source (Survey → Source).")
            return
        if not self._receivers:
            reply = QMessageBox.question(
                self,
                "Run Forward",
                "No receivers are defined (Survey → Receivers).\n"
                "You will not get any seismogram data.\n\n"
                "Do you want to continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        default_name = "Fwd 1"
        existing = [r["name"] for r in self._forward_runs]
        for i in range(1, 1000):
            cand = "Fwd {}".format(i)
            if cand not in existing:
                default_name = cand
                break
        dlg = ForwardRunNameDialog(
            default_name=default_name,
            model_source=getattr(self, "_last_model_source", "Original") or "Original",
            parent=self,
        )
        if dlg.exec_() != QDialog.Accepted:
            return
        self._pending_forward_name = dlg.get_name()
        if not self._pending_forward_name:
            self._pending_forward_name = default_name
        model_source = dlg.get_model_source()
        self._last_model_source = model_source
        tmax_ms = self._sim_settings["tmax_ms"]
        dt_ms = self._sim_settings["dt_ms"]
        npml = self._sim_settings["npml"]
        snapshot_dt_ms = self._sim_settings["snapshot_dt_ms"]
        laplacian = self._sim_settings.get("laplacian", "4pt")

        nt = max(1, int(round(tmax_ms / dt_ms)))
        dt_s = dt_ms / 1000.0
        save_every = max(1, int(round(snapshot_dt_ms / dt_ms)))
        order = 4 if laplacian == "4pt" else 2

        if model_source == "Smoothed" and self._smooth_size_m > 0:
            vp_display = prepare_migration_velocity(
                self._vp.T, self._smooth_size_m, self._dx, self._dz
            ).T
        else:
            vp_display = np.array(self._vp, dtype=np.float64, copy=True)
            nz, nx = vp_display.shape
            for d in self._diffractors:
                x, z, r, v = d["x"], d["z"], d["r"], d["v"]
                ix_c = int(round(x / self._dx))
                iz_c = int(round(z / self._dz))
                nr = max(1, int(round(r / min(self._dx, self._dz))))
                for di in range(-nr, nr + 1):
                    for dj in range(-nr, nr + 1):
                        ii, jj = iz_c + di, ix_c + dj
                        if 0 <= ii < nz and 0 <= jj < nx:
                            if np.sqrt((di * self._dz) ** 2 + (dj * self._dx) ** 2) <= r:
                                vp_display[ii, jj] = v
        vp_sim = vp_display.T

        sx, sz, freq = self._source[0], self._source[1], self._source[2]
        src = ricker(freq, nt, dt_s)

        self._progress_bar.setMaximum(nt)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("%v / %m")

        base_dir = os.path.dirname(os.path.abspath(self._project_path)) if self._project_path else _root_dir
        project_stem = os.path.splitext(os.path.basename(self._project_path))[0] if self._project_path else "unsaved"
        snap_h5_path = _get_snapshots_h5_path(base_dir, self._pending_forward_name, "forward", project_stem=project_stem)

        self._forward_thread = QThread()
        self._forward_worker = ForwardSimulationWorker(
            src, vp_sim, nt, dt_s, self._dx, self._dz,
            sx, sz, npml, save_every, order,
            snapshots_h5_path=snap_h5_path,
            snapshot_dt_ms=snapshot_dt_ms,
            tmax_ms=tmax_ms,
        )
        self._forward_worker.moveToThread(self._forward_thread)
        self._forward_worker.progress.connect(self._on_forward_progress)
        self._forward_worker.finished.connect(self._on_forward_finished)
        self._forward_worker.error.connect(self._on_forward_error)
        self._forward_thread.started.connect(self._forward_worker.run)
        self._forward_thread.start()

    def _on_forward_progress(self, current, total):
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)

    def _compute_seismogram_from_snapshots(self, history):
        """
        Из снапшотов компоненты history (n_save, nx, nz) извлекает трассы
        в позициях приёмников. Возвращает (data, t_ms): data (n_save, n_receivers), t_ms в мс.
        """
        if history is None or getattr(history, "size", 0) == 0 or not self._receivers:
            return None, None
        history = np.asarray(history, dtype=np.float64)
        if history.ndim != 3:
            return None, None
        n_save, nx, nz = history.shape
        snapshot_dt_ms = self._sim_settings.get("snapshot_dt_ms", 2.0)
        t_ms = np.arange(n_save, dtype=np.float64) * snapshot_dt_ms
        data = np.zeros((n_save, len(self._receivers)), dtype=np.float64)
        dx, dz = self._dx, self._dz
        for rec_idx, (x, z) in enumerate(self._receivers):
            ix = int(round(x / dx))
            jz = int(round(z / dz))
            ix = max(0, min(nx - 1, ix))
            jz = max(0, min(nz - 1, jz))
            data[:, rec_idx] = history[:, ix, jz]
        return data, t_ms

    def _on_forward_finished(self, result):
        if self._forward_thread is not None:
            self._forward_thread.quit()
            self._forward_thread.wait()
            self._forward_thread = None
        self._forward_worker = None
        self._progress_bar.setValue(self._progress_bar.maximum())
        if result is not None and self._pending_forward_name:
            name = self._pending_forward_name
            # Результат — либо (path,) при записи в HDF5, либо (p_history, vx_history, vz_history)
            if len(result) == 1 and isinstance(result[0], str):
                h5_path = result[0]
                # Сейсмограммы (P, Vz, Vx) считаем по кадрам из HDF5, без загрузки всего массива в память (избегаем OOM)
                seismograms = {}
                seismogram_t_ms = None
                for comp in ("P", "Vz", "Vx"):
                    def _seismogram_progress(current, total, _comp=comp):
                        self._progress_bar.setMaximum(total)
                        self._progress_bar.setValue(current)
                        self._progress_bar.setFormat("Seismogram {} %v / %m".format(_comp))
                        QApplication.processEvents()
                    data_c, t_c = snapshot_io.compute_seismogram_from_h5(
                        h5_path, self._receivers, self._dx, self._dz,
                        snapshot_dt_ms=self._sim_settings.get("snapshot_dt_ms", 2.0),
                        progress_callback=_seismogram_progress,
                        component=comp,
                    )
                    seismograms[comp] = data_c
                    if seismogram_t_ms is None:
                        seismogram_t_ms = t_c
                print("[Forward] seismogram from H5 done", flush=True)
                npz_path = _get_seismogram_npz_path(h5_path)
                try:
                    np.savez_compressed(
                        npz_path,
                        P=seismograms["P"],
                        Vz=seismograms["Vz"],
                        Vx=seismograms["Vx"],
                        t_ms=seismogram_t_ms,
                    )
                except Exception:
                    pass
                snapshots = snapshot_io.snapshots_dict_from_h5(h5_path, "forward")
                print("[Forward] snapshots_dict_from_h5 done", flush=True)
                self._forward_runs = [r for r in self._forward_runs if r["name"] != name]
                self._forward_runs.append({
                    "name": name,
                    "snapshots_path": h5_path,
                    "snapshots": snapshots,
                    "seismograms": seismograms,
                    "seismogram_t_ms": seismogram_t_ms,
                })
                print("[Forward] run appended", flush=True)
            else:
                p_history, vx_history, vz_history = result
                seismograms = {}
                seismogram_t_ms = None
                for comp, arr in (("P", p_history), ("Vx", vx_history), ("Vz", vz_history)):
                    data_c, t_c = self._compute_seismogram_from_snapshots(arr)
                    seismograms[comp] = data_c
                    if seismogram_t_ms is None:
                        seismogram_t_ms = t_c
                self._forward_runs = [r for r in self._forward_runs if r["name"] != name]
                self._forward_runs.append({
                    "name": name,
                    "snapshots": {"P fwd": p_history, "Vz fwd": vz_history, "Vx fwd": vx_history},
                    "seismograms": seismograms,
                    "seismogram_t_ms": seismogram_t_ms,
                })
                print("[Forward] run appended (in-memory)", flush=True)
            self._current_fwd_name = name
            self._current_seismogram_name = name
            self._pending_forward_name = None
            print("[Forward] rebuild_effective_snapshots...", flush=True)
            self._rebuild_effective_snapshots()
            print("[Forward] apply_current_seismogram...", flush=True)
            self._apply_current_seismogram()
            print("[Forward] update_layer_availability...", flush=True)
            self._snapshot_slider.setValue(0)
            self._chk_snapshots.setChecked(True)
            self._update_layer_availability()
            print("[Forward] update_memory_status...", flush=True)
            self._update_memory_status()
            print("[Forward] maybe_reduce_memory...", flush=True)
            self._maybe_reduce_memory()
            print("[Forward] done, showing dialog", flush=True)
        QMessageBox.information(self, "Run Forward", "Forward run completed.")

    def _on_forward_error(self, err_msg):
        if self._forward_thread is not None:
            self._forward_thread.quit()
            self._forward_thread.wait()
            self._forward_thread = None
        self._forward_worker = None
        self._progress_bar.setValue(0)
        QMessageBox.warning(self, "Run Forward", "Error:\n\n" + err_msg)

    def _simulation_run_backward(self):
        if len(self._forward_runs) == 0:
            QMessageBox.warning(self, "Run Backward", "Run forward first (Run Forward).")
            return
        if not self._receivers:
            QMessageBox.warning(self, "Run Backward", "Set receivers (Survey → Receivers).")
            return
        default_name = "Bwd 1"
        existing = [r["name"] for r in self._backward_runs]
        for i in range(1, 1000):
            cand = "Bwd {}".format(i)
            if cand not in existing:
                default_name = cand
                break
        seismogram_names = [r["name"] for r in self._forward_runs]
        default_source_type = None
        default_residual_full = None
        default_residual_smoothed = None
        if len(seismogram_names) >= 2:
            default_source_type = "Residual"
            default_residual_full = seismogram_names[-2]
            default_residual_smoothed = seismogram_names[-1]
        dlg = BackwardRunNameDialog(
            default_name=default_name,
            seismogram_names=seismogram_names,
            model_source=getattr(self, "_last_model_source", "Original") or "Original",
            default_source_type=default_source_type,
            default_residual_full=default_residual_full,
            default_residual_smoothed=default_residual_smoothed,
            parent=self,
        )
        if dlg.exec_() != QDialog.Accepted:
            return
        self._pending_backward_name = dlg.get_name() or default_name
        source_type = dlg.get_seismogram_source_type()
        comp = getattr(self, "_seismogram_component", "P") or "P"
        if comp not in ("P", "Vz", "Vx"):
            comp = "P"
        if source_type == "named":
            self._pending_backward_seismogram_name = dlg.get_seismogram_source_name()
            if not self._pending_backward_seismogram_name:
                QMessageBox.warning(self, "Run Backward", "Select seismogram source.")
                return
            fwd_run = self._get_forward_run(self._pending_backward_seismogram_name)
            seis_map = (fwd_run.get("seismograms") or {}) if fwd_run else {}
            if not fwd_run or seis_map.get(comp) is None:
                QMessageBox.warning(self, "Run Backward", "Selected run has no {} seismogram.".format(comp))
                return
            seismogram_data = np.array(seis_map.get(comp), dtype=np.float64, copy=True)
            seismogram_t_ms = fwd_run["seismogram_t_ms"]
        else:
            full_name = dlg.get_residual_full_name()
            smoothed_name = dlg.get_residual_smoothed_name()
            if not full_name or not smoothed_name:
                QMessageBox.warning(self, "Run Backward", "Select Full fwd and Smoothed fwd for Residual.")
                return
            if full_name == smoothed_name:
                QMessageBox.warning(self, "Run Backward", "Full fwd and Smoothed fwd must be different runs.")
                return
            fwd_full = self._get_forward_run(full_name)
            fwd_smooth = self._get_forward_run(smoothed_name)
            full_map = (fwd_full.get("seismograms") or {}) if fwd_full else {}
            smooth_map = (fwd_smooth.get("seismograms") or {}) if fwd_smooth else {}
            if not fwd_full or full_map.get(comp) is None:
                QMessageBox.warning(self, "Run Backward", "Run «{}» has no {} seismogram.".format(full_name, comp))
                return
            if not fwd_smooth or smooth_map.get(comp) is None:
                QMessageBox.warning(self, "Run Backward", "Run «{}» has no {} seismogram.".format(smoothed_name, comp))
                return
            full_data = np.asarray(full_map.get(comp), dtype=np.float64)
            smooth_data = np.asarray(smooth_map.get(comp), dtype=np.float64)
            t_full = np.asarray(fwd_full["seismogram_t_ms"], dtype=np.float64)
            t_smooth = np.asarray(fwd_smooth["seismogram_t_ms"], dtype=np.float64)
            n_save_full, n_rec = full_data.shape
            n_save_smooth, n_rec_s = smooth_data.shape
            if n_rec != n_rec_s:
                QMessageBox.warning(
                    self, "Run Backward",
                    "Receiver count must match between Full and Smoothed."
                )
                return
            if len(t_full) != n_save_full or len(t_smooth) != n_save_smooth:
                QMessageBox.warning(self, "Run Backward", "Inconsistent seismogram dimensions.")
                return
            smooth_on_full = np.zeros_like(full_data)
            for rec in range(n_rec):
                smooth_on_full[:, rec] = np.interp(t_full, t_smooth, smooth_data[:, rec])
            seismogram_data = full_data - smooth_on_full
            seismogram_t_ms = t_full
            self._pending_backward_seismogram_name = "Residual (Full: {}, Smoothed: {})".format(full_name, smoothed_name)
        model_source = dlg.get_model_source()
        self._last_model_source = model_source
        tmax_ms = self._sim_settings["tmax_ms"]
        dt_ms = self._sim_settings["dt_ms"]
        npml = self._sim_settings["npml"]
        snapshot_dt_ms = self._sim_settings["snapshot_dt_ms"]
        laplacian = self._sim_settings.get("laplacian", "4pt")

        nt = max(1, int(round(tmax_ms / dt_ms)))
        dt_s = dt_ms / 1000.0
        save_every = max(1, int(round(snapshot_dt_ms / dt_ms)))
        order = 4 if laplacian == "4pt" else 2

        if model_source == "Smoothed" and self._smooth_size_m > 0:
            vp_display = prepare_migration_velocity(
                self._vp.T, self._smooth_size_m, self._dx, self._dz
            ).T
        else:
            vp_display = np.array(self._vp, dtype=np.float64, copy=True)
            nz, nx = vp_display.shape
            for d in self._diffractors:
                x, z, r, v = d["x"], d["z"], d["r"], d["v"]
                ix_c = int(round(x / self._dx))
                iz_c = int(round(z / self._dz))
                nr = max(1, int(round(r / min(self._dx, self._dz))))
                for di in range(-nr, nr + 1):
                    for dj in range(-nr, nr + 1):
                        ii, jj = iz_c + di, ix_c + dj
                        if 0 <= ii < nz and 0 <= jj < nx:
                            if np.sqrt((di * self._dz) ** 2 + (dj * self._dx) ** 2) <= r:
                                vp_display[ii, jj] = v
        vp_sim = vp_display.T

        n_save, n_rec = seismogram_data.shape
        new_t_ms = np.arange(nt, dtype=np.float64) * dt_ms
        record = np.zeros((nt, n_rec), dtype=np.float32)
        for rec in range(n_rec):
            record[:, rec] = np.interp(
                new_t_ms, seismogram_t_ms, seismogram_data[:, rec]
            ).astype(np.float32)
        xrec = np.array([r[0] for r in self._receivers], dtype=np.float64)
        zrec = np.array([r[1] for r in self._receivers], dtype=np.float64)

        self._progress_bar.setMaximum(nt)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("%v / %m")

        base_dir = os.path.dirname(os.path.abspath(self._project_path)) if self._project_path else _root_dir
        project_stem = os.path.splitext(os.path.basename(self._project_path))[0] if self._project_path else "unsaved"
        snap_h5_path = _get_snapshots_h5_path(base_dir, self._pending_backward_name, "backward", project_stem=project_stem)

        self._backward_thread = QThread()
        self._backward_worker = BackwardSimulationWorker(
            record, vp_sim, nt, dt_s, self._dx, self._dz,
            xrec, zrec, npml, save_every, order,
            snapshots_h5_path=snap_h5_path,
            snapshot_dt_ms=snapshot_dt_ms,
            tmax_ms=tmax_ms,
            seismogram_source=self._pending_backward_seismogram_name,
        )
        self._backward_worker.moveToThread(self._backward_thread)
        self._backward_worker.progress.connect(self._on_forward_progress)
        self._backward_worker.finished.connect(self._on_backward_finished)
        self._backward_worker.error.connect(self._on_backward_error)
        self._backward_thread.started.connect(self._backward_worker.run)
        self._backward_thread.start()

    def _on_backward_finished(self, result):
        if self._backward_thread is not None:
            self._backward_thread.quit()
            self._backward_thread.wait()
            self._backward_thread = None
        self._backward_worker = None
        self._progress_bar.setValue(self._progress_bar.maximum())
        if result is not None and self._pending_backward_name:
            name = self._pending_backward_name
            seismogram_src = self._pending_backward_seismogram_name
            if len(result) == 1 and isinstance(result[0], str):
                h5_path = result[0]
                snapshots = snapshot_io.snapshots_dict_from_h5(h5_path, "backward")
                self._backward_runs = [r for r in self._backward_runs if r["name"] != name]
                self._backward_runs.append({
                    "name": name,
                    "snapshots_path": h5_path,
                    "snapshots": snapshots,
                    "seismogram_source": seismogram_src,
                })
            else:
                p_history, vx_history, vz_history = result
                self._backward_runs = [r for r in self._backward_runs if r["name"] != name]
                self._backward_runs.append({
                    "name": name,
                    "snapshots": {"P bwd": p_history, "Vz bwd": vz_history, "Vx bwd": vx_history},
                    "seismogram_source": seismogram_src,
                })
            self._current_bwd_name = name
            self._pending_backward_name = None
            self._pending_backward_seismogram_name = None
            self._rebuild_effective_snapshots()
            self._chk_snapshots.setChecked(True)
            self._update_layer_availability()
            self._update_memory_status()
            self._maybe_reduce_memory()
        QMessageBox.information(self, "Run Backward", "Backward run completed.")

    def _on_backward_error(self, err_msg):
        if self._backward_thread is not None:
            self._backward_thread.quit()
            self._backward_thread.wait()
            self._backward_thread = None
        self._backward_worker = None
        self._progress_bar.setValue(0)
        QMessageBox.warning(self, "Run Backward", "Error:\n\n" + err_msg)

    def _rtm_settings_dialog(self):
        dlg = RTMSettingsDialog(
            source=self._rtm_settings.get("source", "P"),
            parent=self,
        )
        if dlg.exec_() == QDialog.Accepted:
            self._rtm_settings["source"] = dlg.get_source()
            self._update_layer_availability()

    def _rtm_build(self):
        if len(self._forward_runs) == 0 or len(self._backward_runs) == 0:
            QMessageBox.warning(
                self, "RTM Build",
                "At least one forward and one backward run required. Run Forward and Run Backward first.",
            )
            return
        fwd_names = [r["name"] for r in self._forward_runs]
        bwd_names = [r["name"] for r in self._backward_runs]
        dlg = RTMBuildDialog(
            fwd_names=fwd_names,
            bwd_names=bwd_names,
            default_fwd=self._current_fwd_name,
            default_bwd=self._current_bwd_name,
            source=self._rtm_settings.get("source", "P"),
            parent=self,
        )
        if dlg.exec_() != QDialog.Accepted:
            return
        fwd_name = dlg.get_fwd_name()
        bwd_name = dlg.get_bwd_name()
        src = dlg.get_source()
        self._rtm_settings["source"] = src
        if not fwd_name or not bwd_name:
            QMessageBox.warning(self, "RTM Build", "Select Forward and Backward sources.")
            return
        fwd_run = self._get_forward_run(fwd_name)
        bwd_run = self._get_backward_run(bwd_name)
        if not fwd_run or not bwd_run:
            QMessageBox.warning(self, "RTM Build", "Run not found.")
            return
        fwd = (fwd_run.get("snapshots") or {}).get(src + " fwd")
        bwd = (bwd_run.get("snapshots") or {}).get(src + " bwd")
        if fwd is None or bwd is None:
            QMessageBox.warning(
                self, "RTM Build",
                "No «{}» snapshots in selected runs.".format(src),
            )
            return
        fwd_source = (fwd._path, fwd._dset_name) if hasattr(fwd, "read_full") else np.asarray(fwd, dtype=np.float64)
        bwd_source = (bwd._path, bwd._dset_name) if hasattr(bwd, "read_full") else np.asarray(bwd, dtype=np.float64)

        def _source_shape(src):
            if isinstance(src, np.ndarray):
                return src.shape
            meta = snapshot_io.read_snapshots_metadata(src[0])
            return (meta["n_save"], meta["nx"], meta["nz"])

        if _source_shape(fwd_source) != _source_shape(bwd_source):
            QMessageBox.warning(
                self, "RTM Build",
                "Forward and backward snapshot dimensions do not match.",
            )
            return

        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("RTM Build %v / %m")

        def _rtm_progress(current, total):
            self._progress_bar.setMaximum(total)
            self._progress_bar.setValue(current)
            QApplication.processEvents()

        image_xy = snapshot_io.build_rtm_image(fwd_source, bwd_source, progress_callback=_rtm_progress)
        if image_xy.ndim != 2:
            QMessageBox.warning(self, "RTM Build", "Expected 2D slice.")
            return
        self._progress_bar.setValue(self._progress_bar.maximum())
        self._rtm_image_base = image_xy.T.copy()
        self._rtm_image = image_xy.T.copy()
        self.canvas.set_rtm_image(self._rtm_image)
        self._update_layer_availability()
        self._chk_image.setChecked(True)
        self._update_canvas_layers()
        QMessageBox.information(
            self, "RTM Build",
            "Image built: Forward «{}», Backward «{}», component «{}».".format(fwd_name, bwd_name, src),
        )

    def _rtm_post_processing_dialog(self):
        if self._rtm_image is None or self._rtm_image.size == 0:
            QMessageBox.warning(
                self, "RTM Post-Processing",
                "Build Image first (RTM → Build).",
            )
            return
        dlg = RTMPostProcessingDialog(
            laplacian_on=self._rtm_postproc.get("laplacian_on", False),
            laplacian_order=self._rtm_postproc.get("laplacian_order", 4),
            agc_on=self._rtm_postproc.get("agc_on", False),
            agc_window_z_m=self._rtm_postproc.get("agc_window_z_m", 1000.0),
            parent=self,
        )
        if dlg.exec_() == QDialog.Accepted:
            self._rtm_postproc = dlg.get_params()
            self._rtm_apply_post_processing()

    def _rtm_apply_post_processing(self):
        """Применяет постобработку к _rtm_image_base и записывает результат в _rtm_image."""
        base = self._rtm_image_base if self._rtm_image_base is not None else self._rtm_image
        if base is None or base.size == 0:
            return
        img = np.asarray(base, dtype=np.float64).copy()
        p = self._rtm_postproc
        if p.get("laplacian_on"):
            order = p.get("laplacian_order", 4)
            img = _laplacian_filter_2d(img, order, self._dx, self._dz)
        if p.get("agc_on"):
            window_z_m = p.get("agc_window_z_m", 1000.0)
            img = _agc_along_z(img, self._dz, window_z_m)
        self._rtm_image = img
        self.canvas.set_rtm_image(self._rtm_image)
        self._update_canvas_layers()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1400, 680)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
