# Draft: request for a mapped 3D explosion model from the Garching archive

**Status: DRAFT, NOT SENT.** Sending it is the author's call, not the pipeline's.
Fill in the bracketed fields, check the recipient below, and send from your own
account.

## Why this request, in one paragraph (for your own reference, not the email)

`OVERVIEW.md` §5.2 ranks explosion-era structure as the dominant *morphological*
gap, and the phase-randomised null quantifies it: our structure is genuinely
non-Gaussian (Δcoherence +0.56 against a Gaussian null's 0) but **1.6× too
ordered** compared with Chandra's +0.36. The cause is structural, not a tuning
problem — our seed is a Gaussian random field, which is **statistically isotropic
by construction**, while the real ejecta arrive as radially coherent plumes.
Orlando et al. (2025) show the filamentary network forms *during* the explosion
and is merely reprocessed later. We tested the two things we could build
ourselves — a radially coherent plume field (`--plume-sigma`) and Ni bubbles —
and both came back a morphological null (`CALIBRATION.md`, and the negatives
table in `OVERVIEW.md` §5). Building an in-house explosion is blocked on a
degenerate-electron EOS (§7). So the mapped model is the remaining route, and it
is one email.

## Who

The Sukhbold et al. (2016) **progenitor** models in the archive are openly
distributed (`SEWBJ_2015/`, already downloaded to
`/export/data/lstorcks/progenitors/`). The 3D **explosion** models are not — that
directory returns 401. The model Orlando et al. map is
**`W15-2-cw-IIb`** (Wongwathanarat, Janka & Müller). Address the request to the
MPA group that produced it — check the current corresponding author on
arXiv:2503.00130 (Orlando et al. 2025) and on the Wongwathanarat/Janka/Müller
2013/2015/2017 papers before sending, since group contacts change.

---

## The email

**Subject:** Request for access to the mapped 3D neutrino-driven SN model W15-2-cw-IIb

Dear Professor [NAME],

I am [NAME], [POSITION] at [INSTITUTION], working on 3D hydrodynamic modelling of
Cassiopeia A with a differentiable GPU hydrodynamics code we develop in JAX
(`astronomix`). I am writing to ask whether it would be possible to obtain the
mapped 3D neutrino-driven supernova model **W15-2-cw-IIb** — the model used as
the initial condition in Orlando et al. (2025, A&A, arXiv:2503.00130) — or a
comparable Type IIb model from the Garching archive. I appreciate that the 3D
explosion models are not part of the openly distributed set, which is why I am
asking directly.

Some context on why this specific model, and what I would do with it.

We have built a calibrated Route-B pipeline for Cas A: a 1D spherical stage fitted
to the measured shock radii and post-shock density, mapped into 3D at 150 yr and
evolved to 350 yr, then a composition-aware forward model (per-cell µ, µ_e,
non-equilibrium ionization from a carried ionization age, interstellar dust
scattering) that produces a synthetic ACIS event list binned identically to the
real `evt2` data. It reproduces the observed forward and reverse shock radii and
the Chandra count rate to ~15 %.

Where it fails is the *morphology*, and we can now say why quantitatively rather
than impressionistically. Scoring the synthetic and real images with matched
noise and a phase-randomised null, our structure is genuinely non-Gaussian but
comes out about 1.6 times too *ordered* — too few, too coherent features against
Cas A's filamentary web. That is a direct consequence of our initial condition:
we seed the ejecta with a Gaussian random field, which is statistically isotropic
by construction, so its correlation length along a ray equals the one across it.
We have tested the substitutes we could construct ourselves, including a radially
coherent plume field built from von Mises–Fisher lobes and an imposed Ni-bubble
field, and both are morphological nulls on our metrics. This is consistent with
your group's conclusion that the filamentary network is imprinted during the
explosion and subsequently disrupted by the reverse shock, rather than generated
during the remnant phase we simulate.

We also cannot produce the explosion ourselves: our solver is ideal-gas, and a
presupernova core is supported by electron degeneracy — ideal gas plus radiation
accounts for only ~3 % of the central pressure in the KEPLER models, so feeding
those profiles to a γ = 5/3 solver describes a different star.

What I would like to do is straightforward and complementary rather than
competitive: use a mapped 3D model as the initial condition in place of our
statistical seed, and report how much of the residual morphological discrepancy
it closes on our metrics. Our interest is specifically in what a *differentiable*
solver adds downstream — we can compute exact gradients of observables with
respect to physical parameters through the full hydrodynamic evolution, which
makes the calibration a gradient-based fit rather than a parameter scan.

Concretely, I would be grateful for either:

1. the mapped state of `W15-2-cw-IIb` at shock breakout (or whatever epoch is
   convenient), on whatever grid and in whatever format is easiest — we can
   handle Yin-Yang or spherical layouts and interpolate onto our Cartesian grid
   ourselves; or
2. a comparable Type IIb explosion model, if `W15-2-cw-IIb` is not the natural
   one to share.

Density, velocity, and per-species mass fractions would be sufficient; a
temperature or pressure field would be useful but we can reconstruct one under
our own EOS if it is not available.

On attribution and terms: I would of course cite the original model papers and
would be glad to acknowledge the model's provenance explicitly in any resulting
work. If you would prefer a collaborative arrangement, co-authorship on the
resulting paper, an embargo until a particular date, or a restriction to
non-redistribution, I am happy to agree to any of those — please just say which.
If sharing is not possible, a pointer to any comparable publicly available IIb
model would also be very helpful, and I would not press further.

Thank you for considering this, and for the archive — the Sukhbold et al. (2016)
progenitor set has already been directly useful to us. Stripping the s16.0 model
to a Type IIb and excising the iron core gives 3.56 M☉ of ejecta against the
3.0 M☉ we had independently fitted to the X-ray data, which was a reassuring
consistency check we had not expected to get for free.

With best regards,

[NAME]
[POSITION], [INSTITUTION]
[EMAIL]
[ORCID or group page, optional]

---

## Notes on the draft

* **Verify the recipient before sending.** Group contacts change; check the
  corresponding author on arXiv:2503.00130.
* The claim "reproduces ... to ~15 %" is deliberately conservative: it refers to
  r_FS 2.494 against 2.52 ± 0.20 and the count rate 0.86–0.88×. Do not upgrade it
  — the *spectral* agreement is much less settled (`CALIBRATION.md` Result 18).
* The 3.56 vs 3.0 M☉ closing paragraph is `CALIBRATION.md` Result 11. It is
  there because it is true, is directly attributable to their archive, and gives
  a concrete reason to believe the request is not speculative.
* Deliberately **not** in the email: our sub-grid clumping and synchrotron work.
  Both are interpretation layers with fitted parameters, and neither strengthens
  a request about explosion-era structure.
* Offering embargo/co-authorship/non-redistribution up front costs nothing and
  removes the most common reason such requests are declined.
