# License notice — `astronomix/_modules/_chemistry/` (and `setup_helpers/chemistry_setup.py`)

**These chemistry files are licensed under the GNU General Public License v3.0
(GPL-3.0), NOT the MIT license that covers the rest of astronomix.**

Reason: the thermochemistry (`_thermochemistry.py`) is a derivative work of
**KROME** (Grassi et al. 2014, GPL-3.0) — the Glover & Abel (2008) multi-collider
H2 cooling, the `wCool`/`sigmoid` smoothing, the Neufeld & Kaufman (1993) CO
trilinear lookup, and the `heatingChem` critical-density partition were
transcribed from KROME source. The driver (`_chemistry.py`) and setup
(`chemistry_setup.py`) also depend at runtime on **carbox** (GPL-3.0-or-later).

Implications a maintainer must weigh before merging:
- GPL is copyleft: distributing astronomix with these files combined in generally
  subjects the combined/distributed work to GPL-3.0 for the parts that include or
  link this code. Keeping astronomix MIT while carrying this subtree is a
  deliberate mixed-license choice, and its reach should be confirmed.
- No KROME data is bundled. CO cooling requires the caller to supply their own
  `coolCO.dat` (KROME/Omukai, GPL) via `co_cooling_table_path`.
- carbox must be installed for the feature to run (it is imported lazily, so
  `import astronomix` itself is unaffected).

Alternatives that avoid the GPL entanglement (see the project discussion):
1. keep only the MIT-clean species/source-term *mechanism* in astronomix and ship
   this engine as a separate GPL companion package; or
2. reimplement the cooling from the primary papers (not KROME source) so it can be
   MIT, leaving carbox as an optional dependency.
