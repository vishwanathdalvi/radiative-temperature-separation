# Breakdown of global thermal equilibration in radiative systems with constrained angular transport

This repository provides the Python scripts used to generate all Figure 2 results reported in the manuscript “Angular phase-space constraints produce temperature separation in passive radiative enclosures”. Each script corresponds directly to one row of Figure 2 and can be run independently.

**"Angular phase-space constraints produce temperature separation in passive radiative enclosures"**

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

## Quick verification (Figure 2)

To reproduce a case (e.g., Figure 2, Row 3):
```bash
python scripts/figure2_row03.py
```
This particular case corresponds to a strongly asymmetric configuration and produces a clear steady-state temperature difference between source and receiver.

The scripts generate transient/convergence plots for the corresponding Figure 2 case.

## Output

Each script opens a matplotlib figure with 2 panels:
- upper panel: temperature difference between bins (statistics are annotated)
- lower panel: steady-state receiver temperature (statistics are annotated)

It also prints out these statistics after every epoch of 1000 rays.

Because the simulations are stochastic, exact trajectory histories and finite-sample fluctuations depend on the random seed; however, the reported steady-state behaviour is reproducible within statistical uncertainty, so the scripts do not fix a single seed.

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

## Validation

Control cases with symmetric geometry and emissivity reproduce equilibrium (no temperature difference), confirming that the implementation does not introduce systematic bias.

## Code Availability
The full simulation scripts for all Figure 2 configurations are available in this repository.
## Contact

Corresponding author: Vishwanath H. Dalvi
