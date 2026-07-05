# astronomix — contributor & agent guide

## Code style

This project has a house code style: readable, well-narrated code in the spirit
of `astronomix/time_stepping/time_integration.py`. **Read `STYLE_GUIDE.md` and
apply it.**

You may draft in whatever style you find natural, but **before finishing a change
convert the touched files back to the house style** using the conversion
checklist at the end of `STYLE_GUIDE.md`. In short:

- Every `.py` file opens with a `"""..."""` module docstring.
- Imports are grouped and labelled (general / third-party / astronomix).
- Substantial functions have an `Args:`/`Returns:` docstring.
- Comments explain *why*, in full sentences.
- Long functions are split with the box-divider sections.
- Multi-line calls are one-argument-per-line.
- Names are descriptive and spelled out (`density_index`, not `_di`) — readability
  wins over brevity, even at ~20% more code.
- No debug probes, stray `print`s, or commented-out code in committed work.
- Index state arrays through `registered_variables`, never hard-coded indices.

## Verification

Run style/correctness checks on CPU to avoid contending for the shared GPUs:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. python -m py_compile <file>
JAX_PLATFORMS=cpu PYTHONPATH=. python -c "import astronomix"
```
