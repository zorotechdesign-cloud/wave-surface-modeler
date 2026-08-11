"""Generatore interattivo di superfici ondulate ed esportatore STL."""

import datetime
import os
import queue
import struct
import threading

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import opensimplex
from matplotlib.colors import LightSource
from matplotlib.widgets import Button, RadioButtons, Slider


FIELD_SIZE_X = 200.0   # mm
FIELD_SIZE_Y = 200.0   # mm
RES_PREVIEW = 200      # punti per lato nella preview interattiva
# 598 = (RES_PREVIEW - 1) * 3 + 1: ogni vertice della preview è presente
# esattamente anche nella mesh esportata, con tre suddivisioni per intervallo.
RES_EXPORT = 598       # punti per lato nello STL ad alta risoluzione
SEED = 42              # mantiene identica la forma tra preview ed export

PRESETS = {
    "1 – Morbido": {"frequency": 0.012, "wrinkles": 1.0, "wave_size": 6.0},
    "2 – Medio": {"frequency": 0.020, "wrinkles": 2.0, "wave_size": 10.0},
    "3 – Dettagliato": {"frequency": 0.035, "wrinkles": 3.5, "wave_size": 14.0},
    "4 – Estremo": {"frequency": 0.060, "wrinkles": 5.0, "wave_size": 18.0},
}
PRESET_NAMES = list(PRESETS)


def compute_surface(res, frequency, wrinkles, wave_size, seed):
    """Calcola la superficie deterministica X/Y/Z in millimetri."""
    x_vals = np.linspace(0.0, FIELD_SIZE_X, res)
    y_vals = np.linspace(0.0, FIELD_SIZE_Y, res)
    X, Y = np.meshgrid(x_vals, y_vals)

    # L'istanza locale è sicura anche durante l'esportazione in background.
    generatore = opensimplex.OpenSimplex(seed)
    noise = generatore.noise2array(x_vals * frequency, y_vals * frequency)
    Z = wave_size * np.sin(noise * 3.0 * wrinkles) * ((0.5 + 0.5 * noise) ** 2)
    return X, Y, Z


def write_stl_binary(filepath, X, Y, Z):
    """Scrive due triangoli per cella in formato STL binario."""
    v00 = np.stack([X[:-1, :-1], Y[:-1, :-1], Z[:-1, :-1]], axis=-1)
    v10 = np.stack([X[:-1, 1:], Y[:-1, 1:], Z[:-1, 1:]], axis=-1)
    v01 = np.stack([X[1:, :-1], Y[1:, :-1], Z[1:, :-1]], axis=-1)
    v11 = np.stack([X[1:, 1:], Y[1:, 1:], Z[1:, 1:]], axis=-1)

    def normals(a, b, c):
        normal = np.cross(b - a, c - a)
        magnitude = np.linalg.norm(normal, axis=-1, keepdims=True)
        return (normal / np.where(magnitude > 1e-12, magnitude, 1.0)).astype(np.float32)

    n1 = normals(v00, v10, v01)
    n2 = normals(v10, v11, v01)
    cells = (Z.shape[0] - 1) * (Z.shape[1] - 1)
    triangle_count = cells * 2
    dtype = np.dtype([
        ("normal", np.float32, (3,)),
        ("v1", np.float32, (3,)),
        ("v2", np.float32, (3,)),
        ("v3", np.float32, (3,)),
        ("attr", np.uint16),
    ])
    data = np.zeros(triangle_count, dtype=dtype)
    shape = (cells, 3)

    data["normal"][:cells] = n1.reshape(shape)
    data["v1"][:cells] = v00.reshape(shape).astype(np.float32)
    data["v2"][:cells] = v10.reshape(shape).astype(np.float32)
    data["v3"][:cells] = v01.reshape(shape).astype(np.float32)
    data["normal"][cells:] = n2.reshape(shape)
    data["v1"][cells:] = v10.reshape(shape).astype(np.float32)
    data["v2"][cells:] = v11.reshape(shape).astype(np.float32)
    data["v3"][cells:] = v01.reshape(shape).astype(np.float32)

    header_text = b"STL generato da onde_generator.py"
    with open(filepath, "wb") as file:
        file.write(header_text + b"\x00" * (80 - len(header_text)))
        file.write(struct.pack("<I", triangle_count))
        file.write(data.tobytes())


class OndeApp:
    def __init__(self):
        initial = PRESETS[PRESET_NAMES[0]]
        self.frequency = initial["frequency"]
        self.wrinkles = initial["wrinkles"]
        self.wave_size = initial["wave_size"]
        self.seed = SEED
        self.res_preview = RES_PREVIEW
        self.res_export = RES_EXPORT
        self._update_timer = None
        self._export_timer = None
        self._reset_timer = None
        self._export_results = queue.Queue()
        self._exporting = False
        self._build_ui()
        self._update_surface(None)
        plt.show()

    def _build_ui(self):
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.patch.set_facecolor("#1e1e2e")
        self.fig.canvas.manager.set_window_title("Generatore Superfici Ondulate 3D")

        self.ax3d = self.fig.add_axes([0.02, 0.08, 0.62, 0.88], projection="3d")

        title_ax = self.fig.add_axes([0.675, 0.93, 0.31, 0.05])
        title_ax.axis("off")
        title_ax.text(0.5, 0.5, "Controlli", ha="center", va="center",
                      color="white", fontsize=12, fontweight="bold")

        ax_freq = self.fig.add_axes([0.69, 0.78, 0.28, 0.03], facecolor="#2e2e4e")
        self.sl_freq = Slider(ax_freq, "Densità dettagli", 0.005, 0.10,
                              valinit=self.frequency, color="#7788ff")
        self._style_slider(self.sl_freq)
        self.sl_freq.on_changed(self._schedule_update)
        self._add_help(0.742, "Quanto sono ravvicinati i rilievi sulla superficie.")

        ax_wri = self.fig.add_axes([0.69, 0.68, 0.28, 0.03], facecolor="#2e2e4e")
        self.sl_wri = Slider(ax_wri, "Complessità onde", 0.5, 8.0,
                             valinit=self.wrinkles, color="#77ffbb")
        self._style_slider(self.sl_wri)
        self.sl_wri.on_changed(self._schedule_update)
        self._add_help(0.642, "Numero di pieghe e variazioni presenti nel rilievo.")

        ax_wave = self.fig.add_axes([0.69, 0.58, 0.28, 0.03], facecolor="#2e2e4e")
        self.sl_wave = Slider(ax_wave, "Altezza rilievo (mm)", 1.0, 30.0,
                              valinit=self.wave_size, color="#ffbb77")
        self._style_slider(self.sl_wave)
        self.sl_wave.on_changed(self._schedule_update)
        self._add_help(0.542, "Altezza verticale massima delle onde, in millimetri.")

        ax_radio = self.fig.add_axes([0.675, 0.30, 0.31, 0.20], facecolor="#2e2e4e")
        self.radio = RadioButtons(ax_radio, PRESET_NAMES, activecolor="#7788ff")
        ax_radio.set_title("Configurazioni predefinite", color="white", fontsize=9, pad=4)
        for label in self.radio.labels:
            label.set_color("white")
            label.set_fontsize(8)
        self.radio.on_clicked(self._apply_preset)

        ax_home = self.fig.add_axes([0.69, 0.255, 0.28, 0.04])
        self.btn_home = Button(ax_home, "↶ Torna alla vista iniziale",
                               color="#3d3d5c", hovercolor="#56567a")
        self.btn_home.label.set_color("white")
        self.btn_home.label.set_fontsize(9)
        self.btn_home.on_clicked(self._reset_view)

        ax_save = self.fig.add_axes([0.69, 0.17, 0.28, 0.07])
        self.btn_save = Button(ax_save, "Salva STL ad alta risoluzione",
                               color="#3355aa", hovercolor="#5577cc")
        self.btn_save.label.set_color("white")
        self.btn_save.label.set_fontsize(10)
        self.btn_save.on_clicked(self._save_stl)

        info_ax = self.fig.add_axes([0.675, 0.08, 0.31, 0.07])
        info_ax.axis("off")
        triangles = 2 * (self.res_export - 1) ** 2
        info_ax.text(
            0.5, 0.5,
            f"Anteprima: {self.res_preview}×{self.res_preview}   •   "
            f"STL: {self.res_export}×{self.res_export}\n"
            f"{triangles:,} triangoli".replace(",", "."),
            ha="center", va="center", color="#aaaacc", fontsize=8,
        )

    @staticmethod
    def _style_slider(slider):
        slider.label.set_color("white")
        slider.valtext.set_color("white")

    def _add_help(self, y, text):
        ax = self.fig.add_axes([0.69, y, 0.28, 0.025])
        ax.axis("off")
        ax.text(0, 0.5, text, ha="left", va="center", color="#aaaacc", fontsize=7)

    def _read_sliders(self):
        self.frequency = float(self.sl_freq.val)
        self.wrinkles = float(self.sl_wri.val)
        self.wave_size = float(self.sl_wave.val)

    def _schedule_update(self, _value):
        """Raggruppa i movimenti rapidi degli slider in un solo ridisegno."""
        if self._update_timer is not None:
            self._update_timer.stop()
        self._update_timer = self.fig.canvas.new_timer(interval=140)
        self._update_timer.single_shot = True
        self._update_timer.add_callback(self._update_surface, None)
        self._update_timer.start()

    def _update_surface(self, _value):
        self._read_sliders()
        X, Y, Z = compute_surface(
            self.res_preview, self.frequency, self.wrinkles, self.wave_size, self.seed
        )
        self.ax3d.cla()
        self.ax3d.set_facecolor("#1e1e2e")
        self.ax3d.tick_params(colors="white", labelsize=7)
        for pane in (self.ax3d.xaxis.pane, self.ax3d.yaxis.pane, self.ax3d.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#444466")

        light = LightSource(azdeg=225, altdeg=50)
        base = np.ones((*Z.shape, 3)) * np.array([0.45, 0.72, 0.52])
        colors = light.shade_rgb(base, Z, vert_exag=1.0)
        self.ax3d.plot_surface(
            X, Y, Z, facecolors=colors, linewidth=0, antialiased=True,
            rcount=self.res_preview, ccount=self.res_preview,
        )

        # Scala geometrica reale: la preview non amplifica più artificialmente Z.
        z_range = max(float(np.ptp(Z)), 1.0)
        self.ax3d.set_box_aspect([FIELD_SIZE_X, FIELD_SIZE_Y, z_range])
        margin = max(z_range * 0.04, 0.25)
        self.ax3d.set_zlim(float(Z.min()) - margin, float(Z.max()) + margin)
        self.ax3d.set_xlim(0, FIELD_SIZE_X)
        self.ax3d.set_ylim(0, FIELD_SIZE_Y)
        self._home_limits = (
            (0.0, FIELD_SIZE_X),
            (0.0, FIELD_SIZE_Y),
            (float(Z.min()) - margin, float(Z.max()) + margin),
        )
        self.ax3d.set_xlabel("Larghezza X (mm)", color="white", fontsize=8)
        self.ax3d.set_ylabel("Profondità Y (mm)", color="white", fontsize=8)
        self.ax3d.set_zlabel("Altezza Z (mm)", color="white", fontsize=8)
        self.ax3d.set_title("Anteprima 3D in proporzioni reali", color="white", fontsize=10)
        self.fig.canvas.draw_idle()

    def _reset_view(self, _event):
        """Ripristina orientamento e zoom iniziali senza cambiare la superficie."""
        self.ax3d.view_init(elev=30, azim=-60, roll=0)
        if hasattr(self, "_home_limits"):
            x_limits, y_limits, z_limits = self._home_limits
            self.ax3d.set_xlim(*x_limits)
            self.ax3d.set_ylim(*y_limits)
            self.ax3d.set_zlim(*z_limits)
        self.fig.canvas.draw_idle()

    def _apply_preset(self, label):
        preset = PRESETS[label]
        self.sl_freq.set_val(preset["frequency"])
        self.sl_wri.set_val(preset["wrinkles"])
        self.sl_wave.set_val(preset["wave_size"])

    def _save_stl(self, _event):
        if self._exporting:
            return

        # Se uno slider è appena stato mosso, aggiorna prima la preview: ciò che
        # l'utente vede e i parametri esportati restano sempre sincronizzati.
        if self._update_timer is not None:
            self._update_timer.stop()
        self._update_surface(None)
        parameters = (
            self.res_export, self.frequency, self.wrinkles, self.wave_size, self.seed
        )
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                f"onde_{timestamp}.stl")

        print("\n" + "─" * 42)
        print("  Generazione STL in corso…")
        print(f"  Risoluzione:       {self.res_export}×{self.res_export}")
        print(f"  Densità dettagli:  {self.frequency:.4f}")
        print(f"  Complessità onde:  {self.wrinkles:.2f}")
        print(f"  Altezza rilievo:   {self.wave_size:.2f} mm")

        self._exporting = True
        self.btn_save.set_active(False)
        self.btn_save.label.set_text("Generazione in corso…")
        self.fig.canvas.draw_idle()

        def export_worker():
            try:
                X, Y, Z = compute_surface(*parameters)
                write_stl_binary(filepath, X, Y, Z)
                self._export_results.put((True, filepath, os.path.getsize(filepath)))
            except Exception as error:
                self._export_results.put((False, filepath, error))

        threading.Thread(target=export_worker, daemon=True).start()
        self._poll_export()

    def _poll_export(self):
        """Aggiorna Tk soltanto dal thread grafico principale."""
        try:
            success, filepath, detail = self._export_results.get_nowait()
        except queue.Empty:
            self._export_timer = self.fig.canvas.new_timer(interval=100)
            self._export_timer.single_shot = True
            self._export_timer.add_callback(self._poll_export)
            self._export_timer.start()
            return

        self._exporting = False
        self.btn_save.set_active(True)
        if success:
            print(f"\n  File salvato: {filepath}")
            print(f"  Dimensione: {detail / (1024 * 1024):.1f} MB")
            print("─" * 42 + "\n")
            self.btn_save.label.set_text("Salvato!")
        else:
            print(f"\n  Errore durante il salvataggio: {detail}\n")
            self.btn_save.label.set_text("Errore")
        self.fig.canvas.draw_idle()

        self._reset_timer = self.fig.canvas.new_timer(interval=2000)
        self._reset_timer.single_shot = True
        self._reset_timer.add_callback(self._reset_save_button)
        self._reset_timer.start()

    def _reset_save_button(self):
        if not self._exporting:
            self.btn_save.label.set_text("Salva STL ad alta risoluzione")
            self.fig.canvas.draw_idle()


if __name__ == "__main__":
    print("╔══════════════════════════════════════╗")
    print("║  Generatore Superfici Ondulate 3D   ║")
    print("║  Avvio in corso…                    ║")
    print("╚══════════════════════════════════════╝")
    OndeApp()
