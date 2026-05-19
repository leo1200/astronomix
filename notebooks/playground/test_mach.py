"""
Standalone unit test for _calculate_mach_at_surface.

Tests the function directly with synthetic primitive states,
no simulation required.

Run with:
    python test_mach_at_surface.py
"""

import jax.numpy as jnp
import jax
from functools import partial

# ============================================================================
# COPY OF FUNCTION UNDER TEST
# (so this file is self-contained, no astronomix import needed)
# ============================================================================

gamma_gas = 5 / 3


def _calculate_mach_at_surface_simple(pressure, shock_surface):
    """
    Stripped-down version of _calculate_mach_at_surface for testing.
    Takes pressure array and shock_surface bool array directly.
    """
    mach_array = jnp.zeros_like(pressure)

    p2 = pressure[:-2]   # post-shock (left neighbor)
    p1 = pressure[2:]    # pre-shock (right neighbor)

    p_ratio = jnp.maximum(p2 / jnp.maximum(p1, 1e-30), 1.0)
    M = jnp.sqrt((p_ratio * (gamma_gas + 1) + (gamma_gas - 1)) / (2 * gamma_gas))

    mach_array = mach_array.at[1:-1].set(
        jnp.where(shock_surface[1:-1], M, 0.0)
    )

    return mach_array


def mach_from_p_ratio(p2_p1):
    """Ground truth: Rankine-Hugoniot M from pressure ratio."""
    return jnp.sqrt((p2_p1 * (gamma_gas + 1) + (gamma_gas - 1)) / (2 * gamma_gas))


def p_ratio_from_mach(M):
    """Inverse: pressure ratio from Mach number."""
    return (2 * gamma_gas * M**2 - (gamma_gas - 1)) / (gamma_gas + 1)


# ============================================================================
# HELPERS
# ============================================================================

def make_pressure_array(n, p_left, p_right, shock_idx):
    """Flat pressure array with a single sharp jump at shock_idx."""
    p = jnp.ones(n) * p_right
    p = p.at[:shock_idx].set(p_left)
    return p


def make_surface_array(n, shock_idx):
    """Boolean array with single True at shock_idx."""
    s = jnp.zeros(n, dtype=jnp.bool_)
    s = s.at[shock_idx].set(True)
    return s


def check(name, got, expected, atol=0.05):
    ok = jnp.isclose(got, expected, atol=atol)
    symbol = "✓" if ok else "✗ FAIL"
    print(f"  {symbol}  {name}: got {got:.4f}, expected {expected:.4f}")
    return bool(ok)


# ============================================================================
# TESTS
# ============================================================================

def test_sod_tube():
    """Sod tube pressure values from actual simulation output."""
    print("\nTest 1: Sod tube (p_left=0.2441, p_right=0.1191)")
    n, idx = 501, 250
    p = make_pressure_array(n, 0.2441, 0.1191, idx)
    s = make_surface_array(n, idx)

    mach = _calculate_mach_at_surface_simple(p, s)
    M_got = mach[idx]
    M_expected = mach_from_p_ratio(0.2441 / 0.1191)

    return check("Mach at shock", M_got, M_expected)


def test_known_mach_1_3():
    """Construct pressure jump that exactly gives M=1.3."""
    print("\nTest 2: Known M=1.3")
    M_in = 1.3
    p1 = 0.1
    p2 = p_ratio_from_mach(M_in) * p1

    n, idx = 501, 250
    p = make_pressure_array(n, p2, p1, idx)
    s = make_surface_array(n, idx)

    mach = _calculate_mach_at_surface_simple(p, s)
    M_got = mach[idx]

    return check("Mach at shock", M_got, M_in)


def test_known_mach_5_0():
    """Construct pressure jump that exactly gives M=5.0."""
    print("\nTest 3: Known M=5.0")
    M_in = 5.0
    p1 = 0.1
    p2 = p_ratio_from_mach(M_in) * p1

    n, idx = 501, 250
    p = make_pressure_array(n, p2, p1, idx)
    s = make_surface_array(n, idx)

    mach = _calculate_mach_at_surface_simple(p, s)
    M_got = mach[idx]

    return check("Mach at shock", M_got, M_in)


def test_no_shock_flat_pressure():
    """Flat pressure everywhere — surface cell should return 0 (masked out)."""
    print("\nTest 4: No shock (flat pressure, surface cell returns 0)")
    n, idx = 501, 250
    p = jnp.ones(n) * 0.1
    s = make_surface_array(n, idx)

    mach = _calculate_mach_at_surface_simple(p, s)
    M_got = mach[idx]

    # p_ratio = 1.0 → M = 1.0, but surface is True so it should return M=1.0
    M_expected = mach_from_p_ratio(1.0)
    return check("Mach at flat surface", M_got, M_expected)


def test_non_surface_cells_are_zero():
    """All non-surface cells must be exactly 0."""
    print("\nTest 5: Non-surface cells are zero")
    n, idx = 501, 250
    p = make_pressure_array(n, 0.2441, 0.1191, idx)
    s = make_surface_array(n, idx)

    mach = _calculate_mach_at_surface_simple(p, s)

    # zero out the surface cell, rest must all be 0
    non_surface = mach.at[idx].set(0.0)
    all_zero = jnp.all(non_surface == 0.0)
    ok = bool(all_zero)
    symbol = "✓" if ok else "✗ FAIL"
    print(f"  {symbol}  All non-surface cells zero: {ok}")
    return ok


def test_surface_at_boundary():
    """Surface cell near boundary (idx=1) should still work."""
    print("\nTest 6: Surface cell near left boundary (idx=1)")
    n = 501
    idx = 1
    p = make_pressure_array(n, 0.2441, 0.1191, idx)
    s = make_surface_array(n, idx)

    mach = _calculate_mach_at_surface_simple(p, s)
    M_got = mach[idx]
    M_expected = mach_from_p_ratio(0.2441 / 0.1191)

    return check("Mach at boundary surface", M_got, M_expected)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TEST: _calculate_mach_at_surface")
    print("=" * 60)

    results = [
        test_sod_tube(),
        test_known_mach_1_3(),
        test_known_mach_5_0(),
        test_no_shock_flat_pressure(),
        test_non_surface_cells_are_zero(),
        test_surface_at_boundary(),
    ]

    passed = sum(results)
    failed = len(results) - passed

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(results)} tests")
    print("=" * 60)