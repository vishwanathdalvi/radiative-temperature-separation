# Geometry-induced steady-state temperature separation in radiative systems with directional emissivity

This repository contains the Python scripts used to generate the Figure 2 cases reported in the manuscript:

**"Geometry-induced steady-state temperature separation in radiative systems with directional emissivity"**

## Contents

- `scripts/figure2_row01.py` to `scripts/figure2_row10.py`  
  Independent Python scripts corresponding to Rows 1–10 of Figure 2.
- `figures/`  
  Example output profile plots available for selected rows.

## Software requirements

The scripts were written in Python 3 and use:

- numpy
- matplotlib
- scipy
- randomgen
- numba (imported in some scripts)

A minimal installation is:

```bash
pip install numpy matplotlib scipy randomgen numba
```

## Notes on execution

Each script is self-contained and can be run independently, for example:

```bash
python scripts/figure2_row01.py
```

The scripts generate transient/convergence plots for the corresponding Figure 2 case.

## Mapping to manuscript Figure 2

- Row 1: `figure2_row01.py`
- Row 2: `figure2_row02.py`
- Row 3: `figure2_row03.py`
- Row 4: `figure2_row04.py`
- Row 5: `figure2_row05.py`
- Row 6: `figure2_row06.py`
- Row 7: `figure2_row07.py`
- Row 8: `figure2_row08.py`
- Row 9: `figure2_row09.py`
- Row 10: `figure2_row10.py`

## Reproducibility note

These scripts reproduce the Monte Carlo ray-tracing cases used for Figure 2. Because the simulations are stochastic, exact trajectory histories vary with random number generation, but the reported steady-state behaviour should be reproducible within statistical uncertainty.

## Contact

Corresponding author: Vishwanath H. Dalvi
