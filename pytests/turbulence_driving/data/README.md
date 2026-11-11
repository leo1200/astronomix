`turbgen_reference.npz` is the output of the C++ TurbGen (github.com/chfeder/turbulence_generator, commit
d9a7d742139ad873567ea4e53d47c88f9fa856ae, `TurbGen.h`) for nine parameter sets. It is written by the reference
driver in `../reference/`:

    bash ../reference/build.sh <path to TurbGen.h> <build directory>
    python ../reference/make_reference.py <build directory>/turbgen_ref <work directory> turbgen_reference.npz

The driver includes a copy of `TurbGen.h` with a two-line patch (`TurbGen_instrumented.patch`) that records every
Gaussian deviate drawn. The file here was made with g++ 14.3.0 on aarch64 and `-ffp-contract=off` (no fused
multiply-add, so plain IEEE double arithmetic).

Keys `<case>/<name>`:

- `parameters`: JSON of the TurbGen parameters, the grid, the times in update intervals, the velocity dispersion
  handed to the automatic amplitude adjustment
- `scalars`: nmodes, ncmp, t_decay, dt, OUvar, sol_weight_norm, kmin, kmax
- `modes` (nmodes, ndim), `amplitudes` (nmodes)
- `times`: time, pattern updated, step after the call
- `deviates` (draw, mode, component, real/imaginary): the first draw is the initialisation, every further one an
  update of the Ornstein-Uhlenbeck sequence
- `ou_phases`, `coefficients` (mode, component, real/imaginary): after the initialisation and after every call
- `ampl_factor`: internal factor per component (user value^1.5 times the automatic adjustment)
- `field` (call, component, nx[, ny[, nz]]): the acceleration at the cell centres (i + 1/2) L / n
