# onde_generator.py
# Generatore di superfici ondulate 3D con Simplex Noise
# Richiede: numpy, matplotlib, opensimplex
# Installazione: pip install numpy matplotlib opensimplex

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
import opensimplex
import os
import struct
import datetime
import threading
from matplotlib.colors import LightSource

# ─────────────────────────────────────────────
# PARAMETRI MODIFICABILI
# ─────────────────────────────────────────────
FIELD_SIZE_X = 160.0   # mm
FIELD_SIZE_Y = 160.0   # mm
RES_PREVIEW  = 100     # punti per lato nella preview
RES_EXPORT   = 300     # punti per lato nello STL esportato
SEED         = 42      # seme random per il noise

# ─────────────────────────────────────────────
# PRESET
# ─────────────────────────────────────────────
PRESETS = {
    "1 – Morbido":     {"frequency": 0.012, "wrinkles": 1.0, "wave_size": 6.0},
    "2 – Medio":       {"frequency": 0.020, "wrinkles": 2.0, "wave_size": 10.0},
    "3 – Dettagliato": {"frequency": 0.035, "wrinkles": 3.5, "wave_size": 14.0},
    "4 – Estremo":     {"frequency": 0.060, "wrinkles": 5.0, "wave_size": 18.0},
}
PRESET_NAMES = list(PRESETS.keys())

# ─────────────────────────────────────────────
# FUNZIONE PRINCIPALE: calcola la superficie
# ─────────────────────────────────────────────
def compute_surface(res, frequency, wrinkles, wave_size, seed):
    """
    Genera la griglia Z usando Simplex Noise.
    Restituisce X, Y, Z come array 2D di shape (res, res). Unità: mm.
    """
    opensimplex.seed(seed)
    x_vals = np.linspace(0.0, FIELD_SIZE_X, res)
    y_vals = np.linspace(0.0, FIELD_SIZE_Y, res)
    X, Y = np.meshgrid(x_vals, y_vals)

    # noise2array è l'API vettorizzata nativa: calcola noise su griglia (res x res)
    N = opensimplex.noise2array(x_vals * frequency, y_vals * frequency)

    # Formula originale dell'autore
    Z = wave_size * np.sin(N * 3.0 * wrinkles) * ((0.5 + 0.5 * N) ** 2)

    return X, Y, Z

# ─────────────────────────────────────────────
# SCRITTURA FILE STL BINARIO (numpy vettorizzato)
# ─────────────────────────────────────────────
def write_stl_binary(filepath, X, Y, Z):
    """
    Scrive un file STL binario standard.
    Ogni cella della griglia genera 2 triangoli.
    Usa numpy per la massima velocità e correttezza.
    """
    # Vertici dei 4 angoli di ogni cella
    v00 = np.stack([X[:-1, :-1], Y[:-1, :-1], Z[:-1, :-1]], axis=-1)
    v10 = np.stack([X[:-1, 1:],  Y[:-1, 1:],  Z[:-1, 1:] ], axis=-1)
    v01 = np.stack([X[1:,  :-1], Y[1:,  :-1], Z[1:,  :-1]], axis=-1)
    v11 = np.stack([X[1:,  1:],  Y[1:,  1:],  Z[1:,  1:] ], axis=-1)

    def normali(a, b, c):
        e1 = b - a
        e2 = c - a
        n = np.cross(e1, e2)
        mag = np.linalg.norm(n, axis=-1, keepdims=True)
        mag = np.where(mag > 1e-12, mag, 1.0)
        return (n / mag).astype(np.float32)

    n1 = normali(v00, v10, v01)   # triangolo 1
    n2 = normali(v10, v11, v01)   # triangolo 2

    rows, cols = Z.shape
    num_cells = (rows - 1) * (cols - 1)
    num_triangles = num_cells * 2

    # Struttura STL binario: 12 float32 + 1 uint16 = 50 byte per triangolo
    dtype = np.dtype([
        ('normal', np.float32, (3,)),
        ('v1',     np.float32, (3,)),
        ('v2',     np.float32, (3,)),
        ('v3',     np.float32, (3,)),
        ('attr',   np.uint16),
    ])

    data = np.zeros(num_triangles, dtype=dtype)
    s = (num_cells, 3)

    # Triangoli 1
    data['normal'][:num_cells] = n1.reshape(s)
    data['v1']    [:num_cells] = v00.reshape(s).astype(np.float32)
    data['v2']    [:num_cells] = v10.reshape(s).astype(np.float32)
    data['v3']    [:num_cells] = v01.reshape(s).astype(np.float32)

    # Triangoli 2
    data['normal'][num_cells:] = n2.reshape(s)
    data['v1']    [num_cells:] = v10.reshape(s).astype(np.float32)
    data['v2']    [num_cells:] = v11.reshape(s).astype(np.float32)
    data['v3']    [num_cells:] = v01.reshape(s).astype(np.float32)

    # Header STL: esattamente 80 byte
    header_str = b"STL generato da onde_generator.py"
    header = header_str + b"\x00" * (80 - len(header_str))  # padding a 80 byte esatti

    with open(filepath, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", num_triangles))
        f.write(data.tobytes())

# ─────────────────────────────────────────────
# INTERFACCIA GRAFICA
# ─────────────────────────────────────────────
class OndeApp:
    def __init__(self):
        self.frequency   = PRESETS["1 – Morbido"]["frequency"]
        self.wrinkles    = PRESETS["1 – Morbido"]["wrinkles"]
        self.wave_size   = PRESETS["1 – Morbido"]["wave_size"]
        self.seed        = SEED
        self.res_preview = RES_PREVIEW
        self.res_export  = RES_EXPORT
        self._build_ui()
        self._update_surface(None)
        plt.show()

    def _build_ui(self):
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.patch.set_facecolor("#1e1e2e")
        self.fig.canvas.manager.set_window_title("Generatore Superfici Ondulate 3D")

        # Area 3D
        self.ax3d = self.fig.add_axes([0.02, 0.08, 0.62, 0.88], projection='3d')
        self.ax3d.set_facecolor("#1e1e2e")
        self.ax3d.tick_params(colors='white', labelsize=7)
        for pane in [self.ax3d.xaxis.pane, self.ax3d.yaxis.pane, self.ax3d.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#444466")

        # Slider Rugosità
        ax_freq = self.fig.add_axes([0.69, 0.78, 0.28, 0.03], facecolor="#2e2e4e")
        self.sl_freq = Slider(ax_freq, "Rugosità", 0.005, 0.10,
                              valinit=self.frequency, color="#7788ff")
        self.sl_freq.label.set_color("white")
        self.sl_freq.valtext.set_color("white")
        self.sl_freq.on_changed(self._update_surface)

        # Slider Frequenza
        ax_wri = self.fig.add_axes([0.69, 0.68, 0.28, 0.03], facecolor="#2e2e4e")
        self.sl_wri = Slider(ax_wri, "Frequenza", 0.5, 8.0,
                             valinit=self.wrinkles, color="#77ffbb")
        self.sl_wri.label.set_color("white")
        self.sl_wri.valtext.set_color("white")
        self.sl_wri.on_changed(self._update_surface)

        # Slider Ampiezza onda
        ax_wave = self.fig.add_axes([0.69, 0.58, 0.28, 0.03], facecolor="#2e2e4e")
        self.sl_wave = Slider(ax_wave, "Ampiezza onda", 1.0, 30.0,
                              valinit=self.wave_size, color="#ffbb77")
        self.sl_wave.label.set_color("white")
        self.sl_wave.valtext.set_color("white")
        self.sl_wave.on_changed(self._update_surface)

        # Preset
        ax_radio = self.fig.add_axes([0.675, 0.30, 0.31, 0.22], facecolor="#2e2e4e")
        self.radio = RadioButtons(ax_radio, PRESET_NAMES, activecolor="#7788ff")
        ax_radio.set_title("Preset", color="white", fontsize=9, pad=4)
        for label in self.radio.labels:
            label.set_color("white")
            label.set_fontsize(8)
        self.radio.on_clicked(self._apply_preset)

        # Bottone Salva STL
        ax_save = self.fig.add_axes([0.69, 0.17, 0.28, 0.07])
        self.btn_save = Button(ax_save, "Salva STL",
                               color="#3355aa", hovercolor="#5577cc")
        self.btn_save.label.set_color("white")
        self.btn_save.label.set_fontsize(11)
        self.btn_save.on_clicked(self._save_stl)

        # Info risoluzione
        self.info_ax = self.fig.add_axes([0.675, 0.09, 0.31, 0.06])
        self.info_ax.axis("off")
        self.info_ax.text(
            0.5, 0.5,
            f"Preview: {self.res_preview}×{self.res_preview}   STL: {self.res_export}×{self.res_export}",
            ha="center", va="center", color="#aaaacc", fontsize=8,
            transform=self.info_ax.transAxes
        )

        # Titolo pannello
        title_ax = self.fig.add_axes([0.675, 0.93, 0.31, 0.05])
        title_ax.axis("off")
        title_ax.text(0.5, 0.5, "Controlli",
                      ha="center", va="center", color="white",
                      fontsize=12, fontweight="bold",
                      transform=title_ax.transAxes)

    def _read_sliders(self):
        self.frequency = float(self.sl_freq.val)
        self.wrinkles  = float(self.sl_wri.val)
        self.wave_size = float(self.sl_wave.val)

    def _update_surface(self, val):
        self._read_sliders()
        X, Y, Z = compute_surface(
            self.res_preview, self.frequency,
            self.wrinkles, self.wave_size, self.seed
        )
        self.ax3d.cla()
        self.ax3d.set_facecolor("#1e1e2e")
        self.ax3d.tick_params(colors='white', labelsize=7)
        for pane in [self.ax3d.xaxis.pane, self.ax3d.yaxis.pane, self.ax3d.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#444466")

        ls = LightSource(azdeg=225, altdeg=50)
        base = np.ones((*Z.shape, 3)) * np.array([0.45, 0.72, 0.52])  # verde metallo
        rgb = ls.shade_rgb(base, Z, vert_exag=0.25)
        self.ax3d.plot_surface(
            X, Y, Z,
            facecolors=rgb,
            linewidth=0,
            antialiased=True,
            rcount=self.res_preview,
            ccount=self.res_preview,
        )

        # Proporzioni reali: base 160x160 mm, Z scala reale con lieve enfasi
        z_range = max(float(Z.max() - Z.min()), 1.0)
        self.ax3d.set_box_aspect([FIELD_SIZE_X, FIELD_SIZE_Y, z_range * 3])
        self.ax3d.set_zlim(float(Z.min()) - 1, float(Z.max()) + 1)

        self.ax3d.set_xlabel("X (mm)", color="white", fontsize=8, labelpad=4)
        self.ax3d.set_ylabel("Y (mm)", color="white", fontsize=8, labelpad=4)
        self.ax3d.set_zlabel("Z (mm)", color="white", fontsize=8, labelpad=4)
        self.ax3d.set_title("Superficie Ondulata – Preview 3D", color="white", fontsize=10)
        self.fig.canvas.draw_idle()

    def _apply_preset(self, label):
        p = PRESETS[label]
        self.sl_freq.set_val(p["frequency"])
        self.sl_wri.set_val(p["wrinkles"])
        self.sl_wave.set_val(p["wave_size"])

    def _save_stl(self, event):
        self._read_sliders()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"onde_{timestamp}.stl"
        save_dir  = os.path.dirname(os.path.abspath(__file__))
        filepath  = os.path.join(save_dir, filename)

        print("\n─────────────────────────────────────────")
        print("  Generazione STL in corso...")
        print(f"  Risoluzione export: {self.res_export}×{self.res_export}")
        print(f"  Rugosita':  {self.frequency:.4f}")
        print(f"  Frequenza:  {self.wrinkles:.2f}")
        print(f"  Ampiezza:   {self.wave_size:.2f} mm")
        print(f"  Campo:      {FIELD_SIZE_X:.0f} x {FIELD_SIZE_Y:.0f} mm")

        X, Y, Z = compute_surface(
            self.res_export, self.frequency,
            self.wrinkles, self.wave_size, self.seed
        )
        write_stl_binary(filepath, X, Y, Z)

        size_kb = os.path.getsize(filepath) / 1024
        print(f"\n  File STL salvato:")
        print(f"  {filepath}")
        print(f"  Dimensione: {size_kb:.1f} KB")
        print("─────────────────────────────────────────\n")

        self.btn_save.label.set_text("Salvato!")
        self.fig.canvas.draw_idle()

        def reset():
            import time; time.sleep(2)
            self.btn_save.label.set_text("Salva STL")
            self.fig.canvas.draw_idle()
        threading.Thread(target=reset, daemon=True).start()

# ─────────────────────────────────────────────
# AVVIO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("╔═══════════════════════════════════════╗")
    print("║  Generatore Superfici Ondulate 3D     ║")
    print("║  Avvio in corso...                    ║")
    print("╚═══════════════════════════════════════╝")
    app = OndeApp()
