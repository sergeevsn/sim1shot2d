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
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QFileDialog,
    QMenu, QAction, QDialog, QFormLayout, QDoubleSpinBox, QSpinBox, QPushButton,
    QDialogButtonBox, QListWidget, QListWidgetItem, QGroupBox, QRadioButton,
    QHBoxLayout, QMessageBox, QScrollArea, QFrame, QSizePolicy, QComboBox,
    QCheckBox, QSlider, QGridLayout, QProgressBar, QLineEdit, QStackedWidget,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Импорт из gui/source: добавляем корень проекта в path
_gui_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_gui_dir)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)
from gui.source.model_io import load_velocity_from_segy
from gui.source.simlib_first_order import (
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
            self._ax.text(0.5, 0.5, "Модель не загружена", ha="center", va="center", transform=self._ax.transAxes)
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
        self._ax.set_xlabel("X, м")
        self._ax.set_ylabel("Z, м")
        self._ax.tick_params(axis="both", which="major", labelsize=8)
        if im is not None:
            self._cbar = self._fig.colorbar(im, ax=self._ax, label="V, м/с", shrink=0.8)

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
        if self._layout == "Profile":
            self._ax.set_xlabel("X, м")
            self._ax.set_ylabel("Time, мс")
        else:
            self._ax.set_xlabel("Time, мс")
            self._ax.set_ylabel("Z, м")
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
            self._ax.text(0.5, 0.5, "Сейсмограмма", ha="center", va="center", transform=self._ax.transAxes, fontsize=12)
        self._ax.tick_params(axis="both", which="major", labelsize=8)
        self.draw()


class ModelParametersDialog(QDialog):
    def __init__(self, dx, dz, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model — Parameters")
        layout = QFormLayout(self)
        self.dx_spin = QDoubleSpinBox()
        self.dx_spin.setRange(0.1, 10000)
        self.dx_spin.setDecimals(2)
        self.dx_spin.setValue(dx)
        self.dx_spin.setSuffix(" m")
        layout.addRow("dx (м):", self.dx_spin)
        self.dz_spin = QDoubleSpinBox()
        self.dz_spin.setRange(0.1, 10000)
        self.dz_spin.setDecimals(2)
        self.dz_spin.setValue(dz)
        self.dz_spin.setSuffix(" m")
        layout.addRow("dz (м):", self.dz_spin)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addRow(bb)

    def get_dx_dz(self):
        return self.dx_spin.value(), self.dz_spin.value()


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
        self.size_spin.setSuffix(" м")
        self.size_spin.setToolTip("Радиус сглаживания в метрах; пересчитывается в sigma для 2D gaussian filter")
        layout.addRow("Smooth Size (м):", self.size_spin)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addRow(bb)

    def get_smooth_size_m(self):
        return self.size_spin.value()


class DiffractorDialog(QDialog):
    def __init__(self, parent=None, x=0, z=0, r=10, v=2000):
        super().__init__(parent)
        self.setWindowTitle("Добавить дифрактор")
        layout = QFormLayout(self)
        self.x_spin = QDoubleSpinBox()
        self.x_spin.setRange(-1e6, 1e6)
        self.x_spin.setDecimals(1)
        self.x_spin.setValue(x)
        layout.addRow("X (м):", self.x_spin)
        self.z_spin = QDoubleSpinBox()
        self.z_spin.setRange(0, 1e6)
        self.z_spin.setDecimals(1)
        self.z_spin.setValue(z)
        layout.addRow("Z (м):", self.z_spin)
        self.r_spin = QDoubleSpinBox()
        self.r_spin.setRange(0.1, 1000)
        self.r_spin.setDecimals(1)
        self.r_spin.setValue(r)
        self.r_spin.setSuffix(" m")
        layout.addRow("R — радиус (м):", self.r_spin)
        self.v_spin = QDoubleSpinBox()
        self.v_spin.setRange(100, 10000)
        self.v_spin.setDecimals(0)
        self.v_spin.setValue(v)
        self.v_spin.setSuffix(" m/s")
        layout.addRow("V — скорость (м/с):", self.v_spin)
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
        btn_add = QPushButton("+ Добавить дифрактор")
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
    def __init__(self, x=0, z=0, freq=22, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Survey — Source")
        layout = QFormLayout(self)
        self.x_spin = QDoubleSpinBox()
        self.x_spin.setRange(-1e6, 1e6)
        self.x_spin.setDecimals(1)
        self.x_spin.setValue(x)
        layout.addRow("X (м):", self.x_spin)
        self.z_spin = QDoubleSpinBox()
        self.z_spin.setRange(0, 1e6)
        self.z_spin.setDecimals(1)
        self.z_spin.setValue(z)
        layout.addRow("Z (м):", self.z_spin)
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(1, 500)
        self.freq_spin.setDecimals(1)
        self.freq_spin.setValue(freq)
        self.freq_spin.setSuffix(" Hz")
        layout.addRow("Freq — частота Рикера (Hz):", self.freq_spin)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addRow(bb)

    def get_params(self):
        return self.x_spin.value(), self.z_spin.value(), self.freq_spin.value()


class ReceiversDialog(QDialog):
    def __init__(self, receivers=None, layout_name="Profile", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Survey — Receivers")
        layout = QVBoxLayout(self)
        grp = QGroupBox("Тип расстановки")
        self.radio_profile = QRadioButton("Profile")
        self.radio_well = QRadioButton("Well")
        # Значения по умолчанию (если приёмников нет или не удаётся восстановить)
        p_z, p_x0, p_nx, p_xstep = 100.0, 0.0, 50, 5.0
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
        f_profile.addRow("Z (м):", self.p_z)
        self.p_x0 = QDoubleSpinBox()
        self.p_x0.setRange(-1e6, 1e6)
        self.p_x0.setValue(p_x0)
        f_profile.addRow("X0 (м):", self.p_x0)
        self.p_nx = QSpinBox()
        self.p_nx.setRange(1, 10000)
        self.p_nx.setValue(p_nx)
        f_profile.addRow("NX:", self.p_nx)
        self.p_xstep = QDoubleSpinBox()
        self.p_xstep.setRange(0.1, 1000)
        self.p_xstep.setValue(p_xstep)
        f_profile.addRow("X step (м):", self.p_xstep)
        layout.addWidget(self.form_profile)

        self.form_well = QFrame()
        f_well = QFormLayout(self.form_well)
        self.w_x = QDoubleSpinBox()
        self.w_x.setRange(-1e6, 1e6)
        self.w_x.setValue(w_x)
        f_well.addRow("X (м):", self.w_x)
        self.w_z0 = QDoubleSpinBox()
        self.w_z0.setRange(0, 1e6)
        self.w_z0.setValue(w_z0)
        f_well.addRow("Z0 (м):", self.w_z0)
        self.w_nz = QSpinBox()
        self.w_nz.setRange(1, 10000)
        self.w_nz.setValue(w_nz)
        f_well.addRow("NZ:", self.w_nz)
        self.w_zstep = QDoubleSpinBox()
        self.w_zstep.setRange(0.1, 1000)
        self.w_zstep.setValue(w_zstep)
        f_well.addRow("Z step (м):", self.w_zstep)
        layout.addWidget(self.form_well)
        self.form_well.setVisible(self.radio_well.isChecked())
        self.form_profile.setVisible(self.radio_profile.isChecked())

        def toggled():
            self.form_profile.setVisible(self.radio_profile.isChecked())
            self.form_well.setVisible(self.radio_well.isChecked())
        self.radio_profile.toggled.connect(toggled)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

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
    """Параметры симуляции: Tmax, NPML, DT, SNAPSHOT_DT, SEISMOGRAM_DT, Laplacian scheme. Модель (Original/Smoothed) задаётся в диалогах Run Forward / Run Backward."""
    def __init__(self, parent=None, tmax_ms=1000, npml=50, dt_ms=None, snapshot_dt_ms=2, seismogram_dt_ms=2, laplacian="4pt"):
        super().__init__(parent)
        self.setWindowTitle("Simulation — Settings")
        layout = QFormLayout(self)
        self.tmax_spin = QDoubleSpinBox()
        self.tmax_spin.setRange(1, 1e6)
        self.tmax_spin.setDecimals(0)
        self.tmax_spin.setValue(tmax_ms)
        self.tmax_spin.setSuffix(" мс")
        layout.addRow("Tmax (мс):", self.tmax_spin)
        self.npml_spin = QSpinBox()
        self.npml_spin.setRange(0, 500)
        self.npml_spin.setValue(npml)
        layout.addRow("NPML (узлов сетки):", self.npml_spin)
        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.01, 10)
        self.dt_spin.setDecimals(3)
        self.dt_spin.setValue(dt_ms if dt_ms is not None else 0.5)
        self.dt_spin.setSuffix(" мс")
        self.dt_spin.setToolTip("По умолчанию ~0.5 CFL")
        layout.addRow("DT (мс):", self.dt_spin)
        self.snapshot_dt_spin = QDoubleSpinBox()
        self.snapshot_dt_spin.setRange(0.1, 1000)
        self.snapshot_dt_spin.setDecimals(2)
        self.snapshot_dt_spin.setValue(snapshot_dt_ms)
        self.snapshot_dt_spin.setSuffix(" мс")
        layout.addRow("SNAPSHOT_DT (мс):", self.snapshot_dt_spin)
        self.seismogram_dt_spin = QDoubleSpinBox()
        self.seismogram_dt_spin.setRange(0.1, 1000)
        self.seismogram_dt_spin.setDecimals(2)
        self.seismogram_dt_spin.setValue(seismogram_dt_ms)
        self.seismogram_dt_spin.setSuffix(" мс")
        layout.addRow("SEISMOGRAM_DT (мс):", self.seismogram_dt_spin)
        self.laplacian_combo = QComboBox()
        self.laplacian_combo.addItems(["4pt", "2pt"])
        self.laplacian_combo.setCurrentText(laplacian if laplacian in ("4pt", "2pt") else "4pt")
        layout.addRow("Laplacian scheme:", self.laplacian_combo)
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
            "seismogram_dt_ms": self.seismogram_dt_spin.value(),
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
        layout.addRow("Порядок лапласиана:", self._laplacian_order_combo)
        self._chk_agc = QCheckBox()
        self._chk_agc.setChecked(agc_on)
        layout.addRow("AGC:", self._chk_agc)
        self._agc_window_z = QDoubleSpinBox()
        self._agc_window_z.setRange(1, 100000)
        self._agc_window_z.setDecimals(0)
        self._agc_window_z.setValue(agc_window_z_m)
        self._agc_window_z.setSuffix(" м")
        layout.addRow("Размер окна по Z:", self._agc_window_z)
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
        self.setWindowTitle("Run Forward — имя набора и модель")
        layout = QFormLayout(self)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Например: Fwd From Orig Model")
        self._name_edit.setText(default_name)
        layout.addRow("Имя набора:", self._name_edit)
        self._model_combo = QComboBox()
        self._model_combo.addItems(["Original", "Smoothed"])
        self._model_combo.setCurrentText(model_source if model_source in ("Original", "Smoothed") else "Original")
        layout.addRow("Модель для расчёта:", self._model_combo)
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
    def __init__(self, default_name="Bwd 1", seismogram_names=None, model_source="Original", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Run Backward — имя набора, сейсмограмма и модель")
        layout = QFormLayout(self)
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Например: Bwd From Smoothed")
        self._name_edit.setText(default_name)
        layout.addRow("Имя набора:", self._name_edit)
        self._source_type_combo = QComboBox()
        self._source_type_combo.addItems(["Сейсмограмма", "Residual"])
        self._source_type_combo.currentTextChanged.connect(self._on_source_type_changed)
        layout.addRow("Тип источника:", self._source_type_combo)
        self._stack = QStackedWidget()
        page_named = QWidget()
        form_named = QFormLayout(page_named)
        self._seismogram_combo = QComboBox()
        self._seismogram_combo.addItems(seismogram_names or [])
        form_named.addRow("Сейсмограмма (источник):", self._seismogram_combo)
        self._stack.addWidget(page_named)
        page_residual = QWidget()
        form_residual = QFormLayout(page_residual)
        self._residual_full_combo = QComboBox()
        self._residual_full_combo.addItems(seismogram_names or [])
        form_residual.addRow("Full fwd (сейсмограмма):", self._residual_full_combo)
        self._residual_smoothed_combo = QComboBox()
        self._residual_smoothed_combo.addItems(seismogram_names or [])
        form_residual.addRow("Smoothed fwd (сейсмограмма):", self._residual_smoothed_combo)
        self._stack.addWidget(page_residual)
        layout.addRow(self._stack)
        self._on_source_type_changed(self._source_type_combo.currentText())
        self._model_combo = QComboBox()
        self._model_combo.addItems(["Original", "Smoothed"])
        self._model_combo.setCurrentText(model_source if model_source in ("Original", "Smoothed") else "Original")
        layout.addRow("Модель для расчёта:", self._model_combo)
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
        """Возвращает 'named' или 'residual'."""
        return "residual" if self._source_type_combo.currentText() == "Residual" else "named"

    def get_seismogram_source_name(self):
        """Для типа 'named' — имя набора сейсмограммы; для 'residual' — None."""
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
        layout.addRow("Компонент (Source):", self._source_combo)
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

    def __init__(self, src, vp, nt, dt, dx, dz, xsrc, zsrc, n_absorb, save_every, order):
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
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class BackwardSimulationWorker(QObject):
    """Воркер обратного прогона: record (nt, nrec), vp, nt, dt, xrec, zrec, ..."""
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, record, vp, nt, dt, dx, dz, xrec, zrec, n_absorb, save_every, order):
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
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VSP Sim — Model & Survey")
        self._vp = None
        self._dx = self._dz = 1.0
        self._diffractors = []
        self._source = None  # (x, z, freq)
        self._receivers = []
        self._sim_settings = {
            "tmax_ms": 1000, "npml": 50, "dt_ms": 0.5,
            "snapshot_dt_ms": 2, "seismogram_dt_ms": 2, "laplacian": "4pt",
        }
        self._smooth_size_m = 0.0
        self._forward_runs = []   # [{"name": str, "snapshots": {"P fwd", "Vz fwd", "Vx fwd"}, "seismogram_data", "seismogram_t_ms"}, ...]
        self._backward_runs = []  # [{"name": str, "snapshots": {"P bwd", "Vz bwd", "Vx bwd"}, "seismogram_source": str}, ...]
        self._current_fwd_name = None   # имя набора fwd для отображения снапшотов
        self._current_bwd_name = None   # имя набора bwd для отображения снапшотов
        self._current_seismogram_name = None  # имя forward-набора, чья сейсмограмма отображается
        self._snapshots = None   # эффективный dict для отображения (merge current fwd + current bwd)
        self._seismogram_data = None
        self._seismogram_t_ms = None
        self._pending_forward_name = None
        self._pending_backward_name = None
        self._pending_backward_seismogram_name = None
        self._rtm_image = None  # (nz, nx) RTM image (постобработанный для отображения)
        self._rtm_image_base = None  # (nz, nx) исходный Image после Build
        self._rtm_settings = {"source": "P"}
        self._rtm_postproc = {"laplacian_on": False, "laplacian_order": 4, "agc_on": False, "agc_window_z_m": 1000.0}
        self._receiver_layout = "Profile"  # "Profile" или "Well" — для осей сейсмограммы
        self._project_path = None  # путь к текущему файлу проекта (.ini) или None
        self._model_file_path = ""  # путь к загруженному SEG-Y модели

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        row = QHBoxLayout()
        self._layer_panel = self._build_layer_panel()
        row.addWidget(self._layer_panel)
        # Колонка: холст Z-X + Snapshot Percentile снизу
        canvas_column = QWidget()
        canvas_col_layout = QVBoxLayout(canvas_column)
        canvas_col_layout.setContentsMargins(0, 0, 0, 0)
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
        # Колонка: сейсмограмма + Seismogram Percentile снизу
        seismogram_column = QWidget()
        seis_col_layout = QVBoxLayout(seismogram_column)
        seis_col_layout.setContentsMargins(0, 0, 0, 0)
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
        slayout.addWidget(QLabel("Время снапшота:"))
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
        playout.addWidget(QLabel("Прогресс:"))
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

        self._memory_label = QLabel("Память: —")
        self._memory_label.setToolTip("Использование памяти процессом (RSS). Обновляется каждые 5 с.")
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
        act_exit = QAction("Exit", self)
        act_exit.triggered.connect(self._file_exit)
        file_menu.addAction(act_exit)

        model_menu = menubar.addMenu("&Model")
        act_open = QAction("Open", self)
        act_open.triggered.connect(self._model_open)
        model_menu.addAction(act_open)
        act_params = QAction("Parameters", self)
        act_params.triggered.connect(self._model_parameters)
        model_menu.addAction(act_params)
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

    def _build_layer_panel(self):
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(panel)
        grp_current = QGroupBox("Текущие наборы")
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
        grp = QGroupBox("Слои Z-X")
        gl = QGridLayout()
        self._chk_original = QCheckBox("Оригинальная модель")
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

        self._chk_survey = QCheckBox("Съёмка")
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

        self._chk_smoothed = QCheckBox("Сглаженная модель")
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
        if r and r.get("seismogram_data") is not None and r.get("seismogram_t_ms") is not None:
            self._seismogram_data = r["seismogram_data"]
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
                self._memory_label.setText("Память: {:.0f} / {:.0f} МБ".format(mb, limit_mb))
            else:
                self._memory_label.setText("Память: {:.0f} МБ".format(mb))
        else:
            self._memory_label.setText("Память: —")

    def _maybe_reduce_memory(self):
        """Если использование памяти выше лимита — предлагает удалить самые старые наборы Forward/Backward."""
        mb = _get_process_memory_mb()
        limit_mb = _get_system_memory_limit_mb()
        if mb is None or limit_mb is None or mb <= limit_mb:
            return
        reply = QMessageBox.question(
            self,
            "Память",
            "Использование памяти ({:.0f} МБ) превышает рекомендуемый лимит ({:.0f} МБ).\n"
            "Удалить самые старые наборы Forward и Backward для освобождения памяти?".format(mb, limit_mb),
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
            QMessageBox.information(self, "Память", "Самые старые наборы удалены.")

    def _update_canvas_layers(self):
        p_snap = self._snapshot_percentile_spin.value()
        snapshot_vmin = snapshot_vmax = None
        if self._snapshots is not None:
            comp = self._snapshot_combo.currentText()
            arr = self._snapshots.get(comp)
            if arr is not None and arr.size > 0:
                snapshot_vmin, snapshot_vmax = np.percentile(arr, [100 - p_snap, p_snap])
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
        try:
            self._load_project_from_ini(path)
            self._project_path = path
            self.setWindowTitle("VSP Sim — " + os.path.basename(path))
        except Exception as e:
            import traceback
            QMessageBox.warning(
                self, "Load Project",
                "Не удалось загрузить проект:\n\n{}".format(e))
            traceback.print_exc()

    def _file_save_project(self):
        if self._project_path is None:
            self._file_save_project_as()
            return
        try:
            self._save_project_to_ini(self._project_path)
        except Exception as e:
            QMessageBox.warning(
                self, "Save Project",
                "Не удалось сохранить проект:\n\n{}".format(e))

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
            self.setWindowTitle("VSP Sim — " + os.path.basename(path))
        except Exception as e:
            QMessageBox.warning(
                self, "Save Project As",
                "Не удалось сохранить проект:\n\n{}".format(e))

    def _file_exit(self):
        QApplication.quit()

    def _save_project_to_ini(self, path):
        cfg = ConfigParser()
        cfg.add_section("model")
        model_file = getattr(self, "_model_file_path", "") or ""
        if model_file and os.path.isabs(model_file):
            base_dir = os.path.dirname(os.path.abspath(path))
            try:
                model_file = os.path.relpath(model_file, base_dir)
            except ValueError:
                pass
        cfg.set("model", "file", model_file)
        cfg.set("model", "dx", str(self._dx))
        cfg.set("model", "dz", str(self._dz))
        cfg.set("model", "smooth_size_m", str(self._smooth_size_m))
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

        with open(path, "w", encoding="utf-8") as f:
            cfg.write(f)

    def _load_project_from_ini(self, path):
        cfg = ConfigParser()
        cfg.read(path, encoding="utf-8")
        base_dir = os.path.dirname(os.path.abspath(path))

        if cfg.has_section("model"):
            model_file = cfg.get("model", "file", fallback="").strip()
            if model_file and not os.path.isabs(model_file):
                model_file = os.path.normpath(os.path.join(base_dir, model_file))
            if model_file and os.path.isfile(model_file):
                vp, dx_load, dz_load = load_velocity_from_segy(model_file)
                if vp is not None:
                    self._vp = vp
                    self._model_file_path = os.path.abspath(model_file)
            self._dx = cfg.getfloat("model", "dx", fallback=self._dx)
            self._dz = cfg.getfloat("model", "dz", fallback=self._dz)
            self._smooth_size_m = cfg.getfloat("model", "smooth_size_m", fallback=0.0)
            n_diff = cfg.getint("model", "diffractor_count", fallback=0)
            self._diffractors = []
            for i in range(n_diff):
                self._diffractors.append({
                    "x": cfg.getfloat("model", "diffractor_{}_x".format(i)),
                    "z": cfg.getfloat("model", "diffractor_{}_z".format(i)),
                    "r": cfg.getfloat("model", "diffractor_{}_r".format(i)),
                    "v": cfg.getfloat("model", "diffractor_{}_v".format(i)),
                })

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
            self._sim_settings["seismogram_dt_ms"] = cfg.getfloat("simulation", "seismogram_dt_ms", fallback=self._sim_settings["seismogram_dt_ms"])
            self._sim_settings["laplacian"] = cfg.get("simulation", "laplacian", fallback=self._sim_settings["laplacian"]).strip()
            self._last_model_source = cfg.get("simulation", "model_source", fallback="Original").strip()

        self._apply_velocity_to_canvas()

    def _model_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть скоростную модель (SEG-Y)", "", "SEG-Y (*.sgy *.segy);;All (*)")
        if not path:
            return
        try:
            vp, dx, dz = load_velocity_from_segy(path)
        except Exception as e:
            import traceback
            import sys
            print("Ошибка загрузки модели из файла:", file=sys.stderr)
            print(str(e), file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Ошибка загрузки модели")
            msg.setText("Не удалось загрузить модель из файла.")
            msg.setInformativeText(str(e))
            msg.setDetailedText(traceback.format_exc())
            msg.exec_()
            return
        if vp is None:
            QMessageBox.warning(self, "Ошибка", "Не удалось загрузить модель из файла.")
            return
        self._vp = vp
        self._dx, self._dz = dx, dz
        self._model_file_path = path
        self._apply_velocity_to_canvas()

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
        if self._smooth_size_m > 0 and vp_display is not None:
            smoothed = prepare_migration_velocity(
                vp_display.T, self._smooth_size_m, self._dx, self._dz
            ).T
            self.canvas.set_smoothed_vp(smoothed)
        else:
            self.canvas.set_smoothed_vp(None)
        self._update_layer_availability()
        self._update_canvas_layers()

    def _model_parameters(self):
        dlg = ModelParametersDialog(self._dx, self._dz, self)
        if dlg.exec_() == QDialog.Accepted:
            self._dx, self._dz = dlg.get_dx_dz()
            self._apply_velocity_to_canvas()

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

    def apply_diffractors_and_redraw(self, diffractors):
        self._diffractors = list(diffractors)
        self._apply_velocity_to_canvas()

    def _survey_source(self):
        x, z, freq = 0, 0, 22
        if self._source is not None:
            x, z, freq = self._source[0], self._source[1], self._source[2]
        dlg = SourceDialog(x, z, freq, self)
        if dlg.exec_() == QDialog.Accepted:
            self._source = dlg.get_params()
            self._apply_velocity_to_canvas()

    def _survey_receivers(self):
        dlg = ReceiversDialog(
            receivers=self._receivers,
            layout_name=self._receiver_layout,
            parent=self,
        )
        if dlg.exec_() == QDialog.Accepted:
            self._receivers = dlg.get_receiver_points()
            self._receiver_layout = dlg.get_layout()
            self.seismogram_canvas.set_layout(self._receiver_layout)
            self._apply_velocity_to_canvas()
            # Пересобрать сейсмограммы всех forward-наборов по новой конфигурации приёмников
            for r in self._forward_runs:
                p_fwd = (r.get("snapshots") or {}).get("P fwd")
                if p_fwd is not None:
                    data, t_ms = self._compute_seismogram_from_snapshots(p_fwd)
                    r["seismogram_data"] = data
                    r["seismogram_t_ms"] = t_ms
            self._apply_current_seismogram()

    def _simulation_settings(self):
        dlg = SimulationSettingsDialog(
            self,
            tmax_ms=self._sim_settings["tmax_ms"],
            npml=self._sim_settings["npml"],
            dt_ms=self._sim_settings["dt_ms"],
            snapshot_dt_ms=self._sim_settings["snapshot_dt_ms"],
            seismogram_dt_ms=self._sim_settings["seismogram_dt_ms"],
            laplacian=self._sim_settings["laplacian"],
        )
        if dlg.exec_() == QDialog.Accepted:
            self._sim_settings = dlg.get_params()

    def _simulation_run_forward(self):
        if self._vp is None or self._vp.size == 0:
            QMessageBox.warning(self, "Run Forward", "Загрузите модель (Model → Open).")
            return
        if self._source is None:
            QMessageBox.warning(self, "Run Forward", "Задайте источник (Survey → Source).")
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
        if model_source == "Smoothed" and self._smooth_size_m > 0:
            vp_display = prepare_migration_velocity(
                vp_display.T, self._smooth_size_m, self._dx, self._dz
            ).T
        vp_sim = vp_display.T

        sx, sz, freq = self._source[0], self._source[1], self._source[2]
        src = ricker(freq, nt, dt_s)

        self._progress_bar.setMaximum(nt)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("%v / %m")

        self._forward_thread = QThread()
        self._forward_worker = ForwardSimulationWorker(
            src, vp_sim, nt, dt_s, self._dx, self._dz,
            sx, sz, npml, save_every, order,
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

    def _compute_seismogram_from_snapshots(self, p_history):
        """
        Из снапшотов давления p_history (n_save, nx, nz) извлекает трассы
        в позициях приёмников. Возвращает (data, t_ms): data (n_save, n_receivers), t_ms в мс.
        """
        if p_history is None or p_history.size == 0 or not self._receivers:
            return None, None
        p_history = np.asarray(p_history, dtype=np.float64)
        if p_history.ndim != 3:
            return None, None
        n_save, nx, nz = p_history.shape
        snapshot_dt_ms = self._sim_settings.get("snapshot_dt_ms", 2.0)
        t_ms = np.arange(n_save, dtype=np.float64) * snapshot_dt_ms
        data = np.zeros((n_save, len(self._receivers)), dtype=np.float64)
        dx, dz = self._dx, self._dz
        for rec_idx, (x, z) in enumerate(self._receivers):
            ix = int(round(x / dx))
            jz = int(round(z / dz))
            ix = max(0, min(nx - 1, ix))
            jz = max(0, min(nz - 1, jz))
            data[:, rec_idx] = p_history[:, ix, jz]
        return data, t_ms

    def _on_forward_finished(self, result):
        if self._forward_thread is not None:
            self._forward_thread.quit()
            self._forward_thread.wait()
            self._forward_thread = None
        self._forward_worker = None
        self._progress_bar.setValue(self._progress_bar.maximum())
        if result is not None and self._pending_forward_name:
            p_history, vx_history, vz_history = result
            seismogram_data, seismogram_t_ms = self._compute_seismogram_from_snapshots(p_history)
            name = self._pending_forward_name
            self._forward_runs = [r for r in self._forward_runs if r["name"] != name]
            self._forward_runs.append({
                "name": name,
                "snapshots": {"P fwd": p_history, "Vz fwd": vz_history, "Vx fwd": vx_history},
                "seismogram_data": seismogram_data,
                "seismogram_t_ms": seismogram_t_ms,
            })
            self._current_fwd_name = name
            self._current_seismogram_name = name
            self._pending_forward_name = None
            self._rebuild_effective_snapshots()
            self._apply_current_seismogram()
            self._snapshot_slider.setValue(0)
            self._chk_snapshots.setChecked(True)
            self._update_layer_availability()
            self._update_memory_status()
            self._maybe_reduce_memory()
        QMessageBox.information(self, "Run Forward", "Прямой прогон завершён.")

    def _on_forward_error(self, err_msg):
        if self._forward_thread is not None:
            self._forward_thread.quit()
            self._forward_thread.wait()
            self._forward_thread = None
        self._forward_worker = None
        self._progress_bar.setValue(0)
        QMessageBox.warning(self, "Run Forward", "Ошибка:\n\n" + err_msg)

    def _simulation_run_backward(self):
        if len(self._forward_runs) == 0:
            QMessageBox.warning(self, "Run Backward", "Сначала выполните прямой прогон (Run Forward).")
            return
        if not self._receivers:
            QMessageBox.warning(self, "Run Backward", "Задайте приёмники (Survey → Receivers).")
            return
        default_name = "Bwd 1"
        existing = [r["name"] for r in self._backward_runs]
        for i in range(1, 1000):
            cand = "Bwd {}".format(i)
            if cand not in existing:
                default_name = cand
                break
        seismogram_names = [r["name"] for r in self._forward_runs]
        dlg = BackwardRunNameDialog(
            default_name=default_name,
            seismogram_names=seismogram_names,
            model_source=getattr(self, "_last_model_source", "Original") or "Original",
            parent=self,
        )
        if dlg.exec_() != QDialog.Accepted:
            return
        self._pending_backward_name = dlg.get_name() or default_name
        source_type = dlg.get_seismogram_source_type()
        if source_type == "named":
            self._pending_backward_seismogram_name = dlg.get_seismogram_source_name()
            if not self._pending_backward_seismogram_name:
                QMessageBox.warning(self, "Run Backward", "Выберите источник сейсмограммы.")
                return
            fwd_run = self._get_forward_run(self._pending_backward_seismogram_name)
            if not fwd_run or fwd_run.get("seismogram_data") is None:
                QMessageBox.warning(self, "Run Backward", "У выбранного набора нет сейсмограммы.")
                return
            seismogram_data = np.array(fwd_run["seismogram_data"], dtype=np.float64, copy=True)
            seismogram_t_ms = fwd_run["seismogram_t_ms"]
        else:
            full_name = dlg.get_residual_full_name()
            smoothed_name = dlg.get_residual_smoothed_name()
            if not full_name or not smoothed_name:
                QMessageBox.warning(self, "Run Backward", "Выберите Full fwd и Smoothed fwd для Residual.")
                return
            if full_name == smoothed_name:
                QMessageBox.warning(self, "Run Backward", "Full fwd и Smoothed fwd должны быть разными наборами.")
                return
            fwd_full = self._get_forward_run(full_name)
            fwd_smooth = self._get_forward_run(smoothed_name)
            if not fwd_full or fwd_full.get("seismogram_data") is None:
                QMessageBox.warning(self, "Run Backward", "У набора «{}» нет сейсмограммы.".format(full_name))
                return
            if not fwd_smooth or fwd_smooth.get("seismogram_data") is None:
                QMessageBox.warning(self, "Run Backward", "У набора «{}» нет сейсмограммы.".format(smoothed_name))
                return
            full_data = np.asarray(fwd_full["seismogram_data"], dtype=np.float64)
            smooth_data = np.asarray(fwd_smooth["seismogram_data"], dtype=np.float64)
            t_full = np.asarray(fwd_full["seismogram_t_ms"], dtype=np.float64)
            t_smooth = np.asarray(fwd_smooth["seismogram_t_ms"], dtype=np.float64)
            n_save_full, n_rec = full_data.shape
            n_save_smooth, n_rec_s = smooth_data.shape
            if n_rec != n_rec_s:
                QMessageBox.warning(
                    self, "Run Backward",
                    "Число приёмников в Full и Smoothed должно совпадать."
                )
                return
            if len(t_full) != n_save_full or len(t_smooth) != n_save_smooth:
                QMessageBox.warning(self, "Run Backward", "Несогласованные размеры сейсмограмм.")
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
        if model_source == "Smoothed" and self._smooth_size_m > 0:
            vp_display = prepare_migration_velocity(
                vp_display.T, self._smooth_size_m, self._dx, self._dz
            ).T
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

        self._backward_thread = QThread()
        self._backward_worker = BackwardSimulationWorker(
            record, vp_sim, nt, dt_s, self._dx, self._dz,
            xrec, zrec, npml, save_every, order,
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
            p_history, vx_history, vz_history = result
            name = self._pending_backward_name
            self._backward_runs = [r for r in self._backward_runs if r["name"] != name]
            self._backward_runs.append({
                "name": name,
                "snapshots": {"P bwd": p_history, "Vz bwd": vz_history, "Vx bwd": vx_history},
                "seismogram_source": self._pending_backward_seismogram_name,
            })
            self._current_bwd_name = name
            self._pending_backward_name = None
            self._pending_backward_seismogram_name = None
            self._rebuild_effective_snapshots()
            self._chk_snapshots.setChecked(True)
            self._update_layer_availability()
            self._update_memory_status()
            self._maybe_reduce_memory()
        QMessageBox.information(self, "Run Backward", "Обратный прогон завершён.")

    def _on_backward_error(self, err_msg):
        if self._backward_thread is not None:
            self._backward_thread.quit()
            self._backward_thread.wait()
            self._backward_thread = None
        self._backward_worker = None
        self._progress_bar.setValue(0)
        QMessageBox.warning(self, "Run Backward", "Ошибка:\n\n" + err_msg)

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
                "Нужны хотя бы один forward и один backward набор. Выполните Run Forward и Run Backward.",
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
            QMessageBox.warning(self, "RTM Build", "Выберите источники Forward и Backward.")
            return
        fwd_run = self._get_forward_run(fwd_name)
        bwd_run = self._get_backward_run(bwd_name)
        if not fwd_run or not bwd_run:
            QMessageBox.warning(self, "RTM Build", "Набор не найден.")
            return
        fwd = (fwd_run.get("snapshots") or {}).get(src + " fwd")
        bwd = (bwd_run.get("snapshots") or {}).get(src + " bwd")
        if fwd is None or bwd is None:
            QMessageBox.warning(
                self, "RTM Build",
                "Нет снапшотов «{}» в выбранных наборах.".format(src),
            )
            return
        fwd = np.asarray(fwd, dtype=np.float64)
        bwd = np.asarray(bwd, dtype=np.float64)
        if fwd.shape != bwd.shape:
            QMessageBox.warning(
                self, "RTM Build",
                "Размеры снапшотов forward и backward не совпадают.",
            )
            return
        # Backward снапшоты в обратном времени — переворачиваем по оси t перед перемножением
        bwd = bwd[::-1]
        # Кросс-корреляция по времени (zero-lag): image_xy = sum_t fwd[t] * bwd[t]
        image_xy = np.sum(fwd * bwd, axis=0)
        if image_xy.ndim != 2:
            QMessageBox.warning(self, "RTM Build", "Ожидается 2D срез.")
            return
        self._rtm_image_base = image_xy.T.copy()
        self._rtm_image = image_xy.T.copy()
        self.canvas.set_rtm_image(self._rtm_image)
        self._update_layer_availability()
        self._chk_image.setChecked(True)
        self._update_canvas_layers()
        QMessageBox.information(
            self, "RTM Build",
            "Image построен: Forward «{}», Backward «{}», компонент «{}».".format(fwd_name, bwd_name, src),
        )

    def _rtm_post_processing_dialog(self):
        if self._rtm_image is None or self._rtm_image.size == 0:
            QMessageBox.warning(
                self, "RTM Post-Processing",
                "Сначала постройте Image (RTM → Build).",
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
