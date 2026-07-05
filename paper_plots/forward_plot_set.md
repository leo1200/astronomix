# List of forward tests for the paper

## Hydrodynamics

- pytests/hydrodynamics/figures/sound_wave3D_convergence.svg as is
- the simple shock tube, already done in pytests/hydrodynamics/figures/shock_tube1D_test.svg
- the double blast density (tests/hydro_tests/figures/double_blast.pdf) but with different setups:
    - FV (HLL, minmod) at 400 cells
    - FV (HLLC, minmod) at 400 cells
    - FD at 400 cells
    - FV (HLL, minmod) at 10000 cells
- the Sedov blast wave (see e.g. tests/hydro_tests/figures/sedovAM_HLLC.png) in a 4x3 layout with the following rows, all at 128^3 cells:
    - FV (HLL, minmod)
    - FV (HLLC, minmod)
    - FV (AM-HLLC, minmod)
    - FD

## MHD

- pytests/mhd/figures/alfven_wave3D_dp_convergence.svg as is
- Orszag-Tang vortex with a layout like tests/mhd_tests/figures/orszag_tang.png but with the result of the FD solver in the density plot on the left (as an example) and on the right the profiles (same cuts as in the figure) for
    - FV (HLL, minmod)
    - FD
- MHD blast test in 3D with arena/fv_oscillations_comparison.png as is and then arena/results/fd_mhd/figures/mhd_blast_test1_256cells.png with the finite difference result in the first two columns but in the profile row also the finite volume result at the same resolution for comparison as present in the righ column of arena/results/fv_mhd_hll_mid/figures/mhd_blast_test1_256cells.png - you may use the data in the arena folder to generate these paper plots without running the simulations
- the MHD Jet test, now only for the finite difference solver