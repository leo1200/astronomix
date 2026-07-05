# Self-gravity methods-paper figures

Clean, reproducible scripts for the four self-gravity figures. Each script runs
its simulations once, caches the raw results under `data/` (`jnp.savez`), and
regenerates the figure under `figures/` from that cache — so plots can be
re-styled without re-running any simulation.

All four figures compare the three finite-difference self-gravity source-term
schemes with the **same** consistent names, colours and markers (defined in
`_common.py`):

| scheme (`self_gravity_version`) | label |
| --- | --- |
| `SIMPLE_SOURCE_TERM`  | `FD, simple source` |
| `FD_FLUX_GRAVITY`     | `FD, flux-based source` |
| `WENO_FLUX_GRAVITY`   | `FD, corrected flux-based source` |

## Scripts → figures

| script | figure | data cache |
| --- | --- | --- |
| `jeans_convergence.py` | `figures/jeans_waves_error_convergence.svg` | `data/jeans_convergence.npz` |
| `slab_convergence.py` | `figures/slab_error_convergence.svg` | `data/slab_convergence.npz` |
| `energy_conservation_comparison.py` | `figures/energy_conservation_comparison.svg` | `data/energy_conservation_comparison.npz` |
| `radial_profiles_comparison.py` | `figures/collapse_radial_profiles_comparison.svg` | `data/radial_profiles_comparison.npz` |

The two convergence figures label the x-axis directly with the cell counts and
report the convergence order of the corrected flux-based source scheme from a
direct power-law fit (fit restricted to the pre-saturation range). Measured
orders: jeans ≈ 4.88, slab ≈ 4.87 (≈ 5th order, as expected for WENO5).

## Running

`astronomix` is installed non-editably in site-packages, so put this worktree
on `PYTHONPATH` to pick up the local copy:

```bash
cd <repo-root>
export PYTHONPATH=$(pwd)

# Re-plot only (uses cached data/*.npz):
python paper_plots/gravity/jeans_convergence.py

# Re-run the simulations and re-cache:
python paper_plots/gravity/jeans_convergence.py --rerun
```

A figure is regenerated from cache automatically; pass `--rerun` to recompute
the simulations.

## Notes

- All forward runs use the **Pallas/Triton FD backend** (`_common.pallas_config_kwargs`),
  ~10x faster than native JAX. The WENO Pallas kernel only engages when every
  spatial dimension is divisible by the `pallas_block_shape` (default `(4,4,8)`),
  which is why the convergence resolutions are multiples of 16.
- The Evrard-collapse setup is shared by the two collapse scripts via
  `_collapse.py` (FD, `N = 128³`, periodic box with manual open Poisson
  boundaries).
- The radial-profiles scatter layers are rasterized in the SVG (axes/labels stay
  vector) to keep the file small despite ~2M points per scheme.
