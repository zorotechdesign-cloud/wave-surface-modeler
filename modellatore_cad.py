"""Applicazione STEP/IGES per applicare un rilievo locale a una faccia CAD.

Il programma originale ``onde_generator.py`` rimane indipendente. Questo modulo
usa OpenCASCADE tramite Gmsh per leggere il CAD, triangolarlo e riconoscere le
singole facce. Il risultato lavorato viene esportato come STL binario.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import opensimplex
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    import gmsh
except ImportError as exc:  # messaggio più utile di un traceback all'avvio
    raise SystemExit(
        "Manca Gmsh. Installa le dipendenze con: "
        "uv pip install --python .venv\\Scripts\\python.exe -r requirements.txt"
    ) from exc


SEED = 42


@dataclass
class FaceMesh:
    tag: int
    triangles: np.ndarray
    interior_indices: np.ndarray
    vertex_indices: np.ndarray
    uv: np.ndarray
    cad_normals: np.ndarray
    metric_u: np.ndarray
    metric_v: np.ndarray
    uv_min: np.ndarray
    uv_max: np.ndarray
    periodic_u: bool
    periodic_v: bool
    boundary_distance: np.ndarray
    mesh_size: float
    center: np.ndarray
    area: float


def _write_binary_stl(path: str | os.PathLike, vertices: np.ndarray,
                      triangles: np.ndarray) -> None:
    """Esporta una mesh triangolare indicizzata come STL binario."""
    points = vertices[triangles].astype(np.float32)
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    magnitudes = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = (normals / np.where(magnitudes > 1e-12, magnitudes, 1.0)).astype(np.float32)
    dtype = np.dtype([
        ("normal", np.float32, (3,)),
        ("v1", np.float32, (3,)),
        ("v2", np.float32, (3,)),
        ("v3", np.float32, (3,)),
        ("attr", np.uint16),
    ])
    data = np.zeros(len(triangles), dtype=dtype)
    data["normal"] = normals
    data["v1"], data["v2"], data["v3"] = points[:, 0], points[:, 1], points[:, 2]
    header = b"STL lavorato da modellatore_cad.py"
    with open(path, "wb") as output:
        output.write(header + b"\0" * (80 - len(header)))
        output.write(struct.pack("<I", len(triangles)))
        output.write(data.tobytes())


class CadSurfaceModel:
    """Modello CAD triangolato, con facce ancora identificabili separatamente."""

    def __init__(self) -> None:
        self.filepath: str | None = None
        self.node_tags = np.empty(0, dtype=np.int64)
        self.vertices = np.empty((0, 3), dtype=float)
        self.triangles = np.empty((0, 3), dtype=np.int64)
        self.faces: dict[int, FaceMesh] = {}
        self.bounds = np.zeros((2, 3), dtype=float)
        self._ensure_gmsh()

    @staticmethod
    def _ensure_gmsh() -> None:
        if not gmsh.isInitialized():
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)

    def close(self) -> None:
        if gmsh.isInitialized():
            gmsh.clear()

    def load(self, filepath: str | os.PathLike, mesh_size: float) -> None:
        """Importa STEP/IGES e genera una mesh lineare con facce separate."""
        path = Path(filepath).resolve()
        if path.suffix.lower() not in {".step", ".stp", ".iges", ".igs"}:
            raise ValueError("Formato non supportato: scegliere STEP, STP, IGES o IGS.")
        if not path.is_file():
            raise FileNotFoundError(path)
        if mesh_size <= 0:
            raise ValueError("La risoluzione della mesh deve essere maggiore di zero.")

        self._ensure_gmsh()
        gmsh.clear()
        gmsh.model.add("modello_importato")
        imported = gmsh.model.occ.importShapes(str(path), highestDimOnly=False)
        if not imported:
            raise ValueError("Il file non contiene geometria CAD leggibile.")
        gmsh.model.occ.synchronize()
        surfaces = gmsh.model.getEntities(2)
        if not surfaces:
            raise ValueError("Nel file non sono state trovate superfici.")

        gmsh.option.setNumber("Mesh.MeshSizeMin", float(mesh_size))
        gmsh.option.setNumber("Mesh.MeshSizeMax", float(mesh_size))
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.model.mesh.generate(2)

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        self.node_tags = np.asarray(node_tags, dtype=np.int64)
        self.vertices = np.asarray(coords, dtype=float).reshape(-1, 3)
        tag_to_index = {int(tag): index for index, tag in enumerate(self.node_tags)}
        faces: dict[int, FaceMesh] = {}
        all_triangles: list[np.ndarray] = []

        for _, face_tag in surfaces:
            face_triangles = self._face_triangles(face_tag, tag_to_index)
            if not len(face_triangles):
                continue
            interior_tags, _, _ = gmsh.model.mesh.getNodes(2, face_tag, includeBoundary=False)
            interior = np.fromiter(
                (tag_to_index[int(tag)] for tag in interior_tags), dtype=np.int64,
                count=len(interior_tags),
            )
            # Una curva ripetuta due volte nel contorno della stessa faccia è
            # una seam parametrica (tipica di cilindri e coni), non un bordo
            # fisico. I suoi nodi devono seguire il rilievo senza raccordo a zero.
            from collections import Counter
            boundary_entities = gmsh.model.getBoundary(
                [(2, face_tag)], combined=False, oriented=False, recursive=False
            )
            curve_counts = Counter(tag for dim, tag in boundary_entities if dim == 1)
            seam_curves = [tag for tag, count in curve_counts.items() if count > 1]
            true_boundary_curves = [tag for tag, count in curve_counts.items() if count == 1]

            def curve_node_indices(curve_tags):
                indices = []
                for curve_tag in curve_tags:
                    tags, _, _ = gmsh.model.mesh.getNodes(
                        1, curve_tag, includeBoundary=True
                    )
                    indices.extend(tag_to_index[int(tag)] for tag in tags)
                return np.unique(indices).astype(np.int64) if indices else np.empty(0, dtype=np.int64)

            seam_indices = curve_node_indices(seam_curves)
            true_boundary_indices = curve_node_indices(true_boundary_curves)
            movable_indices = np.unique(np.concatenate((interior, seam_indices)))
            vertex_indices = np.unique(face_triangles)
            face_points = self.vertices[vertex_indices]
            uv = np.asarray(
                gmsh.model.getParametrization(2, face_tag, face_points.ravel()),
                dtype=float,
            ).reshape(-1, 2)
            derivatives = np.asarray(
                gmsh.model.getDerivative(2, face_tag, uv.ravel()), dtype=float
            ).reshape(-1, 6)
            metric_u = np.linalg.norm(derivatives[:, :3], axis=1)
            metric_v = np.linalg.norm(derivatives[:, 3:], axis=1)
            cad_normals = np.asarray(
                gmsh.model.getNormal(face_tag, uv.ravel()), dtype=float
            ).reshape(-1, 3)
            normal_lengths = np.linalg.norm(cad_normals, axis=1, keepdims=True)
            cad_normals /= np.where(normal_lengths > 1e-12, normal_lengths, 1.0)
            uv_min, uv_max = gmsh.model.getParametrizationBounds(2, face_tag)
            surface_type = gmsh.model.getType(2, face_tag).lower()
            periodic_u = any(name in surface_type for name in (
                "cylinder", "cone", "torus", "revolution"
            ))
            periodic_v = "torus" in surface_type

            if len(true_boundary_indices):
                from scipy.spatial import cKDTree
                boundary_distance = cKDTree(self.vertices[true_boundary_indices]).query(
                    face_points, workers=-1
                )[0]
            else:
                boundary_distance = np.full(len(vertex_indices), np.inf)
            center = np.asarray(gmsh.model.occ.getCenterOfMass(2, face_tag), dtype=float)
            area = float(gmsh.model.occ.getMass(2, face_tag))
            faces[int(face_tag)] = FaceMesh(
                int(face_tag), face_triangles, movable_indices, vertex_indices, uv,
                cad_normals, metric_u, metric_v,
                np.asarray(uv_min, dtype=float), np.asarray(uv_max, dtype=float),
                periodic_u, periodic_v, boundary_distance, float(mesh_size),
                center, area,
            )
            all_triangles.append(face_triangles)

        if not faces:
            raise ValueError("Non è stato possibile triangolare le superfici del file.")
        self.faces = faces
        self.triangles = np.vstack(all_triangles)
        self.bounds = np.vstack((self.vertices.min(axis=0), self.vertices.max(axis=0)))
        self.filepath = str(path)

    @staticmethod
    def _face_triangles(face_tag: int, tag_to_index: dict[int, int]) -> np.ndarray:
        element_types, _, node_groups = gmsh.model.mesh.getElements(2, face_tag)
        groups: list[np.ndarray] = []
        for element_type, node_group in zip(element_types, node_groups):
            _, dimension, _, node_count, _, primary_count = gmsh.model.mesh.getElementProperties(
                element_type
            )
            if dimension != 2:
                continue
            nodes = np.asarray(node_group, dtype=np.int64).reshape(-1, node_count)
            nodes = nodes[:, :primary_count]
            indices = np.vectorize(tag_to_index.__getitem__, otypes=[np.int64])(nodes)
            if primary_count == 3:
                groups.append(indices)
            elif primary_count == 4:
                groups.extend((indices[:, [0, 1, 2]], indices[:, [0, 2, 3]]))
        return np.vstack(groups) if groups else np.empty((0, 3), dtype=np.int64)

    def face_vertex_normals(self, face_tag: int) -> np.ndarray:
        """Calcola normali mediate sui vertici di una singola faccia."""
        face = self.faces[face_tag]
        tri = face.triangles
        points = self.vertices[tri]
        triangle_normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
        normals = np.zeros_like(self.vertices)
        for corner in range(3):
            np.add.at(normals, tri[:, corner], triangle_normals)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        return normals / np.where(lengths > 1e-12, lengths, 1.0)

    def apply_texture(self, face_tag: int, center: np.ndarray | None, radius: float,
                      frequency: float, complexity: float, height: float,
                      whole_face: bool = False, seed: int = SEED
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Restituisce vertici deformati, maschera interessata e spostamenti.

        Solo i nodi interni della faccia vengono spostati. La zona locale usa
        una dissolvenza morbida che arriva a zero sul bordo del raggio scelto.
        """
        if face_tag not in self.faces:
            raise ValueError("Selezionare prima una faccia valida.")
        if radius <= 0 or height < 0:
            raise ValueError("Raggio e altezza devono essere valori positivi.")

        face = self.faces[face_tag]
        face_points = self.vertices[face.vertex_indices]
        requested_origin = face.center if center is None else np.asarray(center, dtype=float)
        origin_local = int(np.argmin(np.linalg.norm(
            face_points - requested_origin, axis=1
        )))
        origin = face_points[origin_local]

        # Distanza intrinseca nelle coordinate UV della vera superficie CAD.
        delta_uv = face.uv - face.uv[origin_local]
        spans = face.uv_max - face.uv_min
        if face.periodic_u and spans[0] > 1e-12:
            delta_uv[:, 0] = (delta_uv[:, 0] + spans[0] / 2) % spans[0] - spans[0] / 2
        if face.periodic_v and spans[1] > 1e-12:
            delta_uv[:, 1] = (delta_uv[:, 1] + spans[1] / 2) % spans[1] - spans[1] / 2
        scale_u = 0.5 * (face.metric_u + face.metric_u[origin_local])
        scale_v = 0.5 * (face.metric_v + face.metric_v[origin_local])
        surface_u = delta_uv[:, 0] * scale_u
        surface_v = delta_uv[:, 1] * scale_v
        surface_distance = np.hypot(surface_u, surface_v)

        eligible_local = np.isin(face.vertex_indices, face.interior_indices)
        if whole_face:
            fade_local = eligible_local.astype(float)
        else:
            normalized = np.clip(surface_distance / radius, 0.0, 1.0)
            # Curva C1: valore e pendenza nulli sul bordo della selezione.
            fade_local = (1.0 - normalized * normalized) ** 2
            fade_local[surface_distance >= radius] = 0.0
            fade_local *= eligible_local

        # Raccordo dolce con tutte le facce adiacenti: elimina cuciture e picchi.
        edge_width = max(height * 2.5, face.mesh_size * 4.0)
        edge_t = np.clip(face.boundary_distance / edge_width, 0.0, 1.0)
        edge_fade = edge_t * edge_t * (3.0 - 2.0 * edge_t)
        fade_local *= edge_fade

        # La fantasia è campionata come campo 3D sulla superficie: non esiste
        # più un foglio piano da avvolgere, né una discontinuità sulla cucitura.
        relative = face_points - origin
        generator = opensimplex.OpenSimplex(seed)
        active_local = np.flatnonzero(fade_local > 0)
        noise_local = np.zeros(len(face.vertex_indices), dtype=float)
        noise_local[active_local] = np.fromiter(
            (generator.noise3(
                float(relative[index, 0] * frequency),
                float(relative[index, 1] * frequency),
                float(relative[index, 2] * frequency),
            ) for index in active_local),
            dtype=float, count=len(active_local),
        )
        wave_local = (
            height * np.sin(noise_local * 3.0 * complexity)
            * ((0.5 + 0.5 * noise_local) ** 2)
        )
        displacement_local = wave_local * fade_local

        # Allinea le normali CAD all'orientamento esterno della triangolazione.
        mesh_normals = self.face_vertex_normals(face_tag)[face.vertex_indices]
        cad_normals = face.cad_normals.copy()
        alignment = np.sum(mesh_normals * cad_normals, axis=1)
        valid_alignment = alignment[np.linalg.norm(mesh_normals, axis=1) > 1e-9]
        if len(valid_alignment) and np.median(valid_alignment) < 0:
            cad_normals *= -1.0

        result = self.vertices.copy()
        result[face.vertex_indices] += cad_normals * displacement_local[:, None]
        affected = np.zeros(len(self.vertices), dtype=bool)
        displacement = np.zeros(len(self.vertices), dtype=float)
        affected[face.vertex_indices] = fade_local > 0
        displacement[face.vertex_indices] = displacement_local
        return result, affected, displacement

    def export_stl(self, filepath: str | os.PathLike, vertices: np.ndarray) -> None:
        if vertices.shape != self.vertices.shape:
            raise ValueError("La mesh elaborata non appartiene al modello caricato.")
        _write_binary_stl(filepath, vertices, self.triangles)


class CadTextureApp:
    """Interfaccia per importare, scegliere e lavorare una zona CAD."""

    BG = "#1e1e2e"
    PANEL = "#2e2e4e"

    def __init__(self) -> None:
        self.model = CadSurfaceModel()
        self.selected_face: int | None = None
        self.selection_center: np.ndarray | None = None
        self.deformed_vertices: np.ndarray | None = None
        self.affected_mask: np.ndarray | None = None
        self.face_collections: dict[Poly3DCollection, int] = {}
        # Matplotlib conserva solo riferimenti deboli ai callback dei widget:
        # senza questa lista i pulsanti restano visibili ma smettono di reagire.
        self.buttons: list[Button] = []
        self._build_ui()
        self._show_empty_state()
        self._setup_drag_drop()
        plt.show()

    def _build_ui(self) -> None:
        self.fig = plt.figure(figsize=(15, 9))
        self.fig.patch.set_facecolor(self.BG)
        self.fig.canvas.manager.set_window_title("Modellatore CAD – rilievo locale STEP/IGES")
        self.fig.canvas.mpl_connect("pick_event", self._on_pick)
        self.ax3d = self.fig.add_axes([0.02, 0.08, 0.66, 0.88], projection="3d")

        self._button([0.72, 0.925, 0.12, 0.04], "Sfoglia cartelle…", self._open_file,
                     "#3355aa")
        self._button([0.845, 0.925, 0.125, 0.04], "Incolla e carica",
                     self._paste_and_load,
                     "#3355aa")
        path_ax = self.fig.add_axes([0.72, 0.875, 0.18, 0.035], facecolor="#ffffff")
        path_ax.set_title("Percorso file – incolla qui con Ctrl+V e premi Invio",
                          loc="left", color="#ccccdd", fontsize=7, pad=3)
        self.path_box = TextBox(path_ax, "", initial="", color="#ffffff",
                                hovercolor="#eeeeff")
        self.path_box.text_disp.set_color("#111122")
        self.path_box.on_submit(self._load_from_text)
        self._button([0.905, 0.875, 0.065, 0.035], "Carica",
                     self._load_from_text, "#3355aa")

        self.status_ax = self.fig.add_axes([0.71, 0.805, 0.27, 0.055])
        self.status_ax.axis("off")
        self.status_text = self.status_ax.text(
            0, 1, "Nessun modello caricato", va="top", color="white", fontsize=8,
            wrap=True,
        )

        self.sl_mesh = self._slider(0.755, "Anteprima mesh (mm, ↓ migliore)", 0.5, 4.0, 2.0,
                                    "#aa88ff")
        self._button([0.72, 0.705, 0.12, 0.035], "← Faccia precedente",
                     self._previous_face, "#4b5274")
        self._button([0.85, 0.705, 0.12, 0.035], "Faccia successiva →",
                     self._next_face, "#4b5274")
        self.sl_radius = self._slider(0.66, "Raggio zona (mm)", 2.0, 100.0, 30.0,
                                      "#ff8888")
        self.sl_frequency = self._slider(0.565, "Densità dettagli", 0.005, 0.10, 0.020,
                                         "#7788ff")
        self.sl_complexity = self._slider(0.47, "Complessità onde", 0.5, 8.0, 2.0,
                                          "#77ffbb")
        self.sl_height = self._slider(0.375, "Altezza rilievo (mm)", 0.1, 15.0, 3.0,
                                      "#ffbb77")

        mode_ax = self.fig.add_axes([0.72, 0.245, 0.25, 0.095], facecolor=self.PANEL)
        self.mode = RadioButtons(mode_ax, ["Zona circolare", "Intera faccia"],
                                 activecolor="#ff9966")
        for label in self.mode.labels:
            label.set_color("white")
            label.set_fontsize(8)

        self._button([0.72, 0.175, 0.25, 0.052], "Applica / aggiorna anteprima",
                     self._apply_preview, "#a65c24")
        self._button([0.72, 0.11, 0.12, 0.045], "Vista iniziale",
                     self._reset_view, "#3d3d5c")
        self._button([0.85, 0.11, 0.12, 0.045], "Annulla rilievo",
                     self._clear_texture, "#3d3d5c")
        self.btn_export = self._button(
            [0.72, 0.035, 0.25, 0.058], "Esporta STL alta qualità",
            self._export, "#26734d",
        )

    def _button(self, position, label, callback, color):
        ax = self.fig.add_axes(position)
        button = Button(ax, label, color=color, hovercolor="#5577aa")
        button.label.set_color("white")
        button.label.set_fontsize(9)
        button.on_clicked(callback)
        self.buttons.append(button)
        return button

    def _slider(self, y, label, low, high, initial, color):
        ax = self.fig.add_axes([0.72, y, 0.25, 0.025], facecolor=self.PANEL)
        slider = Slider(ax, label, low, high, valinit=initial, color=color)
        slider.label.set_color("white")
        slider.valtext.set_color("white")
        slider.label.set_fontsize(8)
        return slider

    def _show_empty_state(self):
        self.ax3d.cla()
        self.ax3d.set_facecolor(self.BG)
        self.ax3d.text2D(
            0.5, 0.54, "Apri un modello STEP o IGES",
            transform=self.ax3d.transAxes, ha="center", color="white", fontsize=14,
        )
        self.ax3d.text2D(
            0.5, 0.47, "Poi clicca la faccia e il punto da lavorare",
            transform=self.ax3d.transAxes, ha="center", color="#aaaacc", fontsize=10,
        )
        self.ax3d.set_axis_off()

    def _dialog_parent(self):
        return getattr(self.fig.canvas.manager, "window", None)

    def _setup_drag_drop(self):
        """Abilita il trascinamento di file CAD sull'intera finestra."""
        try:
            from tkinterdnd2 import DND_FILES, TkinterDnD
            window = self._dialog_parent()
            if window is None:
                raise RuntimeError("Finestra grafica non disponibile")
            TkinterDnD._require(window)
            window.drop_target_register(DND_FILES)
            window.dnd_bind("<<Drop>>", self._on_file_drop)
            self.status_text.set_text(
                "Trascina qui un file STEP/IGES, oppure usa Sfoglia/Incolla"
            )
        except Exception as error:
            # Sfoglia e Incolla restano operativi anche se il sistema non
            # supporta l'estensione drag-and-drop.
            self.status_text.set_text(
                f"Usa Sfoglia cartelle o Incolla e carica ({error})"
            )
        self.fig.canvas.draw_idle()

    def _on_file_drop(self, event):
        window = self._dialog_parent()
        try:
            paths = window.tk.splitlist(event.data) if window is not None else (event.data,)
        except Exception:
            paths = (str(event.data).strip("{}"),)
        supported = next(
            (path for path in paths
             if Path(path).suffix.lower() in {".step", ".stp", ".iges", ".igs"}),
            None,
        )
        if not supported:
            self.status_text.set_text(
                "Il file trascinato non è STEP/STP/IGES/IGS."
            )
            self.fig.canvas.draw_idle()
            return
        self._set_path_text(supported)
        self._load_file(supported)

    def _paste_and_load(self, _event):
        """Legge direttamente il percorso dalla clipboard di Windows."""
        window = self._dialog_parent()
        filepath = ""
        try:
            filepath = window.clipboard_get() if window is not None else ""
        except Exception:
            # Quando si copia il file (non il testo del percorso), Explorer usa
            # CF_HDROP invece del normale testo Unicode.
            filepath = self._windows_clipboard_file() if os.name == "nt" else ""
        filepath = str(filepath).splitlines()[0].strip().strip('"').strip() \
            if filepath else ""
        if not filepath:
            self.status_text.set_text(
                "Gli appunti non contengono un file o un percorso STEP/IGES."
            )
            self.fig.canvas.draw_idle()
            return
        self._set_path_text(filepath)
        self._load_file(filepath)

    def _windows_clipboard_file(self):
        """Legge un file copiato direttamente da Esplora file (formato CF_HDROP)."""
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        shell32 = ctypes.windll.shell32
        owner = self._dialog_parent()
        owner_handle = owner.winfo_id() if owner is not None else 0
        if not user32.OpenClipboard(owner_handle):
            return ""
        try:
            user32.GetClipboardData.restype = wintypes.HANDLE
            drop_handle = user32.GetClipboardData(15)  # CF_HDROP
            if not drop_handle:
                return ""
            length = shell32.DragQueryFileW(drop_handle, 0, None, 0)
            if not length:
                return ""
            buffer = ctypes.create_unicode_buffer(length + 1)
            shell32.DragQueryFileW(drop_handle, 0, buffer, len(buffer))
            return buffer.value
        finally:
            user32.CloseClipboard()

    def _open_file(self, _event):
        filepath = self._native_file_dialog()
        if not filepath:
            return
        self._set_path_text(filepath)
        self._load_file(filepath)

    def _native_file_dialog(self):
        """Apre il selettore Explorer nativo, con fallback a Tk sugli altri OS."""
        if os.name == "nt":
            try:
                return self._windows_open_dialog()
            except Exception:
                # Il campo manuale rimane sempre disponibile anche se le API
                # native non fossero accessibili su una particolare macchina.
                pass
        from tkinter import filedialog
        return filedialog.askopenfilename(
            title="Apri modello CAD",
            filetypes=[("Modelli CAD", "*.step *.stp *.iges *.igs"),
                       ("STEP", "*.step *.stp"), ("IGES", "*.iges *.igs")],
        )

    def _windows_open_dialog(self):
        """File dialog Win32: navigazione cartelle e barra percorso standard."""
        import ctypes
        from ctypes import wintypes

        class OpenFileNameW(ctypes.Structure):
            _fields_ = [
                ("lStructSize", wintypes.DWORD), ("hwndOwner", wintypes.HWND),
                ("hInstance", wintypes.HINSTANCE), ("lpstrFilter", wintypes.LPCWSTR),
                ("lpstrCustomFilter", wintypes.LPWSTR),
                ("nMaxCustFilter", wintypes.DWORD), ("nFilterIndex", wintypes.DWORD),
                ("lpstrFile", wintypes.LPWSTR), ("nMaxFile", wintypes.DWORD),
                ("lpstrFileTitle", wintypes.LPWSTR),
                ("nMaxFileTitle", wintypes.DWORD), ("lpstrInitialDir", wintypes.LPCWSTR),
                ("lpstrTitle", wintypes.LPCWSTR), ("Flags", wintypes.DWORD),
                ("nFileOffset", wintypes.WORD), ("nFileExtension", wintypes.WORD),
                ("lpstrDefExt", wintypes.LPCWSTR), ("lCustData", wintypes.LPARAM),
                ("lpfnHook", ctypes.c_void_p), ("lpTemplateName", wintypes.LPCWSTR),
                ("pvReserved", ctypes.c_void_p), ("dwReserved", wintypes.DWORD),
                ("FlagsEx", wintypes.DWORD),
            ]

        buffer = ctypes.create_unicode_buffer(32768)
        owner = self._dialog_parent()
        owner_handle = owner.winfo_id() if owner is not None else 0
        filters = (
            "Modelli CAD (*.step;*.stp;*.iges;*.igs)\0*.step;*.stp;*.iges;*.igs\0"
            "STEP (*.step;*.stp)\0*.step;*.stp\0"
            "IGES (*.iges;*.igs)\0*.iges;*.igs\0Tutti i file\0*.*\0\0"
        )
        dialog = OpenFileNameW()
        dialog.lStructSize = ctypes.sizeof(OpenFileNameW)
        dialog.hwndOwner = owner_handle
        dialog.lpstrFilter = filters
        dialog.nFilterIndex = 1
        dialog.lpstrFile = ctypes.cast(buffer, wintypes.LPWSTR)
        dialog.nMaxFile = len(buffer)
        dialog.lpstrTitle = "Apri modello STEP o IGES"
        dialog.lpstrDefExt = "step"
        # EXPLORER | PATHMUSTEXIST | FILEMUSTEXIST | NOCHANGEDIR
        dialog.Flags = 0x00080000 | 0x00000800 | 0x00001000 | 0x00000008
        get_open = ctypes.windll.comdlg32.GetOpenFileNameW
        get_open.argtypes = [ctypes.POINTER(OpenFileNameW)]
        get_open.restype = wintypes.BOOL
        return buffer.value if get_open(ctypes.byref(dialog)) else ""

    def _load_from_text(self, _event):
        filepath = self.path_box.text.strip().strip('"').strip()
        if not filepath:
            self.status_text.set_text("Incolla un percorso STEP/IGES nel campo bianco.")
            self.fig.canvas.draw_idle()
            return
        self._set_path_text(filepath)
        self._load_file(filepath)

    def _set_path_text(self, filepath):
        """Aggiorna il campo senza attivare nuovamente l'evento Invio."""
        events_enabled = self.path_box.eventson
        self.path_box.eventson = False
        try:
            self.path_box.set_val(filepath)
        finally:
            self.path_box.eventson = events_enabled

    def _load_file(self, filepath):
        self.status_text.set_text("Importazione e triangolazione in corso…")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        try:
            self.model.load(filepath, float(self.sl_mesh.val))
        except Exception as error:
            self.status_text.set_text(f"Errore: {error}")
            self.fig.canvas.draw_idle()
            return
        self.selected_face = None
        self.selection_center = None
        self.deformed_vertices = None
        self.affected_mask = None
        diagonal = float(np.linalg.norm(np.ptp(self.model.bounds, axis=0)))
        self.sl_radius.valmax = max(diagonal, 2.0)
        self.sl_radius.ax.set_xlim(self.sl_radius.valmin, self.sl_radius.valmax)
        self.sl_radius.set_val(max(min(diagonal * 0.15, self.sl_radius.valmax), 2.0))
        self.status_text.set_text(
            f"{Path(filepath).name}\n{len(self.model.faces)} facce – "
            f"{len(self.model.triangles):,} triangoli preview\n"
            "Clicca una faccia o usa Precedente/Successiva".replace(",", ".")
        )
        self._draw_model()

    def _draw_model(self):
        if not self.model.faces:
            return
        vertices = self.model.vertices if self.deformed_vertices is None else self.deformed_vertices
        current_elev = getattr(self.ax3d, "elev", 25)
        current_azim = getattr(self.ax3d, "azim", -55)
        self.ax3d.cla()
        self.ax3d.set_facecolor(self.BG)
        self.face_collections.clear()

        palette = ("#70a58a", "#789bb8", "#8d86b8", "#78a9a4")
        for index, (tag, face) in enumerate(self.model.faces.items()):
            triangles = face.triangles
            if tag == self.selected_face:
                color = "#ff985c"
            else:
                color = palette[index % len(palette)]
            collection = Poly3DCollection(
                vertices[triangles], facecolors=color, linewidths=0, alpha=1.0,
                picker=True, shade=True,
            )
            collection.set_gid(str(tag))
            self.ax3d.add_collection3d(collection)
            self.face_collections[collection] = tag

        if self.selection_center is not None:
            self.ax3d.scatter(*self.selection_center, color="#ff3322", s=45,
                              depthshade=False)
        if (self.selected_face is not None and self.affected_mask is not None
                and self.affected_mask.any()):
            selected_triangles = self.model.faces[self.selected_face].triangles
            worked = selected_triangles[np.any(self.affected_mask[selected_triangles], axis=1)]
            overlay = Poly3DCollection(
                vertices[worked], facecolors="#cc331f", linewidths=0,
                alpha=0.34, shade=True,
            )
            self.ax3d.add_collection3d(overlay)

        low, high = self.model.bounds
        span = np.maximum(high - low, 1e-6)
        margin = span * 0.04
        self.ax3d.set_xlim(low[0] - margin[0], high[0] + margin[0])
        self.ax3d.set_ylim(low[1] - margin[1], high[1] + margin[1])
        self.ax3d.set_zlim(low[2] - margin[2], high[2] + margin[2])
        self.ax3d.set_box_aspect(span)
        self.ax3d.view_init(elev=current_elev, azim=current_azim)
        self.ax3d.set_xlabel("X (mm)", color="white")
        self.ax3d.set_ylabel("Y (mm)", color="white")
        self.ax3d.set_zlabel("Z (mm)", color="white")
        self.ax3d.tick_params(colors="white", labelsize=7)
        title = "Clicca una faccia"
        if self.selected_face is not None:
            title = f"Faccia {self.selected_face} selezionata – clicca il punto centrale della zona"
        self.ax3d.set_title(title, color="white", fontsize=10)
        self.fig.canvas.draw_idle()

    def _on_pick(self, event):
        tag = self.face_collections.get(event.artist)
        if tag is None:
            return
        face = self.model.faces[tag]
        picked = int(event.ind[0]) if len(event.ind) else 0
        picked = min(picked, len(face.triangles) - 1)
        self.selected_face = tag
        self.selection_center = self.model.vertices[face.triangles[picked]].mean(axis=0)
        self.deformed_vertices = None
        self.affected_mask = None
        x, y, z = self.selection_center
        self.status_text.set_text(
            f"Faccia {tag} selezionata\nCentro zona: X {x:.1f}, Y {y:.1f}, Z {z:.1f} mm"
        )
        self._draw_model()

    def _select_face_offset(self, offset):
        """Selezione CAD affidabile anche quando il clic 3D è ambiguo."""
        if not self.model.faces:
            self.status_text.set_text("Carica prima un modello STEP/IGES.")
            self.fig.canvas.draw_idle()
            return
        tags = sorted(self.model.faces)
        if self.selected_face in tags:
            index = (tags.index(self.selected_face) + offset) % len(tags)
        else:
            index = 0 if offset >= 0 else len(tags) - 1
        self.selected_face = tags[index]
        self.selection_center = None
        self.deformed_vertices = None
        self.affected_mask = None
        face = self.model.faces[self.selected_face]
        self.status_text.set_text(
            f"Faccia CAD {self.selected_face} ({index + 1}/{len(tags)}) – "
            f"area {face.area:.1f} mm²\n"
            "Per una zona circolare clicca il suo centro sulla faccia."
        )
        self._draw_model()

    def _previous_face(self, _event):
        self._select_face_offset(-1)

    def _next_face(self, _event):
        self._select_face_offset(1)

    def _apply_preview(self, _event):
        if self.selected_face is None:
            self.status_text.set_text("Prima clicca una faccia del modello.")
            self.fig.canvas.draw_idle()
            return
        whole_face = self.mode.value_selected == "Intera faccia"
        if not whole_face and self.selection_center is None:
            self.status_text.set_text(
                "Faccia scelta. Ora clicca sulla faccia il centro della zona circolare."
            )
            self.fig.canvas.draw_idle()
            return
        try:
            result, affected, displacement = self.model.apply_texture(
                self.selected_face, self.selection_center, float(self.sl_radius.val),
                float(self.sl_frequency.val), float(self.sl_complexity.val),
                float(self.sl_height.val), whole_face=whole_face,
            )
        except Exception as error:
            self.status_text.set_text(f"Errore: {error}")
            self.fig.canvas.draw_idle()
            return
        self.deformed_vertices = result
        self.affected_mask = affected
        self.status_text.set_text(
            f"Rilievo applicato alla faccia {self.selected_face}\n"
            f"{int(affected.sum()):,} vertici modificati – "
            f"spostamento {displacement.min():.2f}/{displacement.max():.2f} mm".replace(",", ".")
        )
        self._draw_model()

    def _clear_texture(self, _event):
        self.deformed_vertices = None
        self.affected_mask = None
        if self.model.faces:
            self.status_text.set_text("Rilievo annullato; selezione conservata.")
            self._draw_model()

    def _reset_view(self, _event):
        if not self.model.faces:
            return
        self.ax3d.view_init(elev=25, azim=-55, roll=0)
        low, high = self.model.bounds
        span = np.maximum(high - low, 1e-6)
        margin = span * 0.04
        self.ax3d.set_xlim(low[0] - margin[0], high[0] + margin[0])
        self.ax3d.set_ylim(low[1] - margin[1], high[1] + margin[1])
        self.ax3d.set_zlim(low[2] - margin[2], high[2] + margin[2])
        self.fig.canvas.draw_idle()

    def _export(self, _event):
        if self.deformed_vertices is None:
            # Evita il vicolo cieco: il pulsante di esportazione genera prima
            # l'anteprima usando i parametri correnti.
            self._apply_preview(None)
            if self.deformed_vertices is None:
                return
        source = Path(self.model.filepath or "modello")
        filepath = source.with_name(f"{source.stem}_lavorato.stl")
        if filepath.exists():
            from datetime import datetime
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = source.with_name(f"{source.stem}_lavorato_{suffix}.stl")
        try:
            # L'export riparte dal B-Rep originale con una triangolazione molto
            # più fine: non riutilizza la mesh grossolana della preview.
            export_size = max(0.18, min(0.45, float(self.sl_mesh.val) / 4.0))
            self.status_text.set_text(
                f"Rigenerazione dal CAD a {export_size:.2f} mm in corso…\n"
                "L'operazione può richiedere alcuni secondi."
            )
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            export_model = CadSurfaceModel()
            export_model.load(source, export_size)

            reference = self.model.faces[self.selected_face]
            diagonal = max(float(np.linalg.norm(np.ptp(self.model.bounds, axis=0))), 1.0)
            def face_score(candidate):
                center_score = np.linalg.norm(candidate.center - reference.center) / diagonal
                area_score = abs(candidate.area - reference.area) / max(reference.area, 1e-9)
                return center_score + area_score
            export_face_tag = min(export_model.faces,
                                  key=lambda tag: face_score(export_model.faces[tag]))
            export_vertices, _, _ = export_model.apply_texture(
                export_face_tag, self.selection_center, float(self.sl_radius.val),
                float(self.sl_frequency.val), float(self.sl_complexity.val),
                float(self.sl_height.val),
                whole_face=self.mode.value_selected == "Intera faccia",
            )
            export_model.export_stl(filepath, export_vertices)
            size = os.path.getsize(filepath) / (1024 * 1024)
            self.status_text.set_text(
                f"STL salvato: {Path(filepath).name} – {size:.1f} MB\n"
                f"{len(export_model.triangles):,} triangoli a {export_size:.2f} mm\n"
                f"Cartella: {Path(filepath).parent}".replace(",", ".")
            )
        except Exception as error:
            self.status_text.set_text(f"Errore esportazione: {error}")
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    try:
        CadTextureApp()
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()
