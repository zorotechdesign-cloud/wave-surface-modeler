# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run the application:**
```powershell
.\.venv\Scripts\Activate.ps1
python onde_generator.py
```

**Install dependencies:**
```powershell
pip install -r requirements.txt
```

There are no tests or lint scripts defined.

## Architecture

The entire application lives in a single file: [onde_generator.py](onde_generator.py) (~302 lines).

**Key constants (lines 20–26):** `FIELD_SIZE_X/Y` (160 mm), `RES_PREVIEW` (100×100 for real-time), `RES_EXPORT` (300×300 for STL), `SEED` (42, fixed for reproducibility).

**Presets (lines 31–37):** Four named configurations stored in `PRESETS` dict — Morbido, Medio, Dettagliato, Estremo.

### Main components

| Component | Description |
|---|---|
| `compute_surface(res, frequency, wrinkles, wave_size, seed)` | Generates X/Y/Z meshgrids using OpenSimplex noise with sinusoidal modulation. |
| `write_stl_binary(filepath, X, Y, Z)` | Converts the surface grid to binary STL via vectorized NumPy — 2 triangles per cell, normals from cross products. |
| `OndeApp` class | Matplotlib GUI with a dark theme, 3D preview (`TkAgg` backend), sliders (Rugosità, Frequenza, Ampiezza onda), RadioButtons for presets, and a save button. |

### Data flow

```
Sliders/Presets → _read_sliders() → compute_surface(RES_PREVIEW) → Matplotlib 3D plot
                                                                       (real-time)

"Salva STL" click → compute_surface(RES_EXPORT=300) → write_stl_binary()
                    → onde_YYYYMMDD_HHMMSS.stl
```

### STL output

Binary STL: 80-byte header + triangle count + 50 bytes/triangle (normal + 3 vertices + attribute). At `RES_EXPORT=300` the output is ~20K triangles and 9–16 MB.

## Key design decisions

- Preview and export use different resolutions (`RES_PREVIEW` vs `RES_EXPORT`) to keep the GUI responsive while producing high-quality files.
- Button feedback ("Salvato!") runs in a daemon thread to avoid blocking the UI event loop.
- All comments and UI labels are in Italian (the project's working language).
