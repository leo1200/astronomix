"""Map the WENO_VARIANT environment variable onto FDSolverConfig options.

    baseline  the scheme before this branch (weno_admissible_face_state off)
    face      weno_admissible_face_state (now the default)
    pp        weno_positivity_preserving (implies the face state)
"""

import os


def weno_variant_kwargs():
    variant = os.environ.get("WENO_VARIANT", "baseline")
    if variant == "baseline":
        return dict(weno_admissible_face_state=False)
    if variant == "face":
        return {}
    if variant == "pp":
        return dict(weno_positivity_preserving=True)
    raise ValueError(f"unknown WENO_VARIANT {variant!r}")


def weno_variant_name():
    return os.environ.get("WENO_VARIANT", "baseline")
