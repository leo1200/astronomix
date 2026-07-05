# astronomix code style guide

This guide captures the house style of the codebase — the readable, well-narrated
style of files like `astronomix/time_stepping/time_integration.py`. Agents and
contributors may draft in their own style, but code is **converted back to this
style before it is committed / merged**. Treat this document as the reference for
that conversion.

> Rule of thumb: optimize for a physicist reading the code top-to-bottom for the
> first time. Prefer clarity and narration over terseness. The compiler does not
> care about line count; the next reader does.

---

## 1. Module-level docstring (every `.py` file)

Every Python file opens with a short `"""..."""` docstring describing what the
module provides, **before** any imports:

```python
"""
Time integration of the fluid equations.

Sets up the snapshot machinery, the per-step update, and the loop driver
(fixed-step / adaptive while / checkpointed) used by all backends.
"""
```

One sentence is fine for small helper modules; a short paragraph for the
substantial ones. No author/date/changelog lines.

---

## 2. Imports

Imports are split into labelled groups, each introduced by a `# comment` header,
in the order below. A `from X import (...)` that lists several names is written
one-name-per-line in parentheses.

**Group order and contents:**

1. `# general` — Python standard library that is *not* type-related
   (`contextlib`, `functools`, `math`, `itertools`, `timeit`, `os`, ...).
2. `# typing` — type-hint machinery: stdlib `typing` and `types`, **plus** the
   runtime type-checkers `jaxtyping` and `beartype`.
3. `# jax` — everything under `jax` (`import jax`, `import jax.numpy as jnp`,
   `from jax.sharding import ...`, `from jax.experimental import checkify`, ...).
4. **Semantic third-party groups** — one labelled group per *role*, not per
   package name: `# numerics` (numpy, scipy), `# plotting` (matplotlib,
   mpl_toolkits), `# checkpointing` (orbax), `# units and constants` (astropy),
   `# neural networks` (`import equinox as eqx`). Pick the label that describes
   what the import is for.
5. astronomix imports, split by *what* is imported:
   - `# astronomix constants` — `UPPER_CASE` constants and the type aliases
     (`STATE_TYPE`, `FIELD_TYPE`, any `*_TYPE`).
   - `# astronomix containers` — the configuration / data container classes
     (`SimulationConfig`, `SimulationParams`, `HelperData`, `RegisteredVariables`,
     `StateStruct`, `SnapshotData`, the `*Config` / `*Settings` option classes, ...).
   - `# astronomix functions` — function imports (everything else: `lower_case`
     or `_lower_case` callables).
   A more specific descriptive label may be used for a self-contained group when
   it aids reading (e.g. `# progress bar`, `# generic time-integration loop driver`).

Example (from `time_stepping/time_integration.py`):

```python
# general
from contextlib import nullcontext

# typing
from typing import Any, NamedTuple, Union
from types import NoneType

# jax
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jax.experimental import checkify

# astronomix constants
from astronomix.option_classes.simulation_config import (
    BACKWARDS,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    STATE_TYPE,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.data_classes.simulation_state_struct import StateStruct

# astronomix functions
from astronomix._finite_volume._state_evolution.evolve_state import _evolve_state_fv
```

**Classification rule of thumb** for an imported astronomix name: `ALL_CAPS` or a
`*_TYPE` alias → constant; a CapWords config/data class → container; a
`lower_case` / leading-underscore callable → function.

- Keep each `from astronomix... import` on the module it comes from; do not merge
  imports from two different modules into one statement.
- Conditional imports (`try: ... except ImportError:`) and imports local to a
  function are left where they are.

---

## 3. Function docstrings

Public and substantial private functions get a full docstring: one-line/paragraph
summary, blank line, `Args:` (one entry per argument, indented continuation), and
`Returns:`. Wrap prose at a comfortable width.

```python
def time_integration(primitive_state, config, params, registered_variables, ...):
    """
    Integrate the fluid equations in time. For the options of the time
    integration see the simulation configuration and the simulation parameters.

    Args:
        primitive_state: The primitive state array.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        Depending on the configuration (return_snapshots, num_snapshots) either
        the final state of the fluid or snapshots of the time evolution.
    """
```

Tiny one-line helpers may use a single-line docstring (`"""Return ... ."""`).

---

## 4. Comments narrate *why*

Comments are full sentences explaining intent, trade-offs and caveats — not a
restatement of the code. Multi-line rationale is written as prose paragraphs.
Use `NOTE:`, `WARNING:`, `TODO:` (with a reason) for emphasis.

```python
# When the user supplies a multi-device sharding, pjit dispatch needs every JIT
# input leaf to carry a sharding compatible with the target mesh. Promote every
# leaf of ``params`` onto a fully-replicated NamedSharding so pjit always sees a
# concrete sharding.
```

---

## 5. Section dividers for long functions

Long functions are broken into labelled sections with the box-divider style, and
matching `↓ ... ↑` markers around the enclosed block:

```python
# -------------------------------------------------------------
# =============== ↓ Setup of the snapshot array ↓ =============
# -------------------------------------------------------------
...
# -------------------------------------------------------------
# =============== ↑ Setup of the snapshot array ↑ =============
# -------------------------------------------------------------
```

Shorter inline blocks use the lighter dashed marker:

```python
# --------------- ↓ Carry unpacking ↓ ----------------
...
# --------------- ↑ Carry unpacking ↑ ----------------
```

---

## 6. Call formatting

Multi-line calls put **one argument per line** when the call spans lines; do not
pack several arguments onto a line to save space:

```python
primitive_state = _evolve_state_fd(
    primitive_state,
    dt,
    params.gamma,
    config,
    params,
    helper_data_pad,
    registered_variables,
)
```

Short calls that fit comfortably on one line stay on one line.

---

## 7. Naming — readability first

Names are the primary documentation. Spell things out so the code reads almost
like prose, **even if it makes the line longer or adds a temporary variable**.
A reader should never have to decode an abbreviation or guess what a symbol holds.

- `snake_case` for functions and variables; descriptive, no abbreviations
  (`unpad_primitive_state`, not `s`; `velocity_threshold`, not `_thr`;
  `density_index`, not `_di`).
- Prefer a slightly longer, self-explaining name over a comment that explains a
  short one. `maximum_signal_speed` beats `c # max signal speed`.
- Avoid leading-underscore *local* temporaries (`_di`, `_vx`) — the leading
  underscore is reserved for module-private module-level names, not locals.
- It is fine to introduce an intermediate, well-named variable purely to make a
  dense expression legible (favour clarity over saving a line). Roughly +20%
  more code for materially better readability is a good trade.
- Never hard-code state-array indices; index through `registered_variables`
  (e.g. `primitive_state[registered_variables.density_index]`).

---

## 8. No development cruft in committed code

The following do not belong in committed/merged code and are removed during
conversion:

- Environment-gated debug probes (`if os.environ.get("DEEPVOID_PROBE"): ...`).
- `print(...)` / `jax.debug.print(...)` / `jax.debug.callback(_probe, ...)` left
  over from debugging.
- Commented-out code blocks kept "just in case".
- Scratch variables, dead branches, `# TEMPORARY` hacks without a tracked reason.

Genuine, configurable features are **not** cruft (e.g. `deepvoid_blend` /
`preserving_flux` are real `PositivityConfig` options and stay).

---

## 9. Conversion checklist (agent style → house style)

When converting a freshly-written file/function back to house style:

1. [ ] Module docstring present at top of file.
2. [ ] Imports grouped + labelled (general / third-party / astronomix).
3. [ ] Every substantial function has an `Args:`/`Returns:` docstring.
4. [ ] Comments explain *why*, in full sentences.
5. [ ] Long functions split with section dividers.
6. [ ] Multi-line calls are one-argument-per-line.
7. [ ] Descriptive names; no `_di`-style local temporaries.
8. [ ] No debug probes, stray prints, or commented-out code.
9. [ ] `registered_variables` used for all state indexing.
