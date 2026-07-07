Option 1 — smooth the fields before differencing
python# apply a simple 3-point average before computing gradient
def _smooth(field):
    smoothed = jnp.zeros_like(field)
    smoothed = smoothed.at[1:-1].set(
        (field[:-2] + field[1:-1] + field[2:]) / 3.0
    )
    return smoothed

# then in _calculate_gradient:
grad_field = _calculate_gradient(_smooth(field), config, r)
Option 2 — widen the stencil
Instead of neighbors at i±1, use i±2:
pythongrad_field = grad_field.at[2:-2].set(
    (field[4:] - field[:-4]) / (4 * config.grid_spacing)
)
This averages over a wider window, suppressing cell-scale noise at the cost of slightly reduced spatial resolution of the gradient.

Option 2 is closer to what Pfrommer et al. implicitly assume — their grid cells in a cosmological simulation are much smoother than a shock tube with a Riemann solver, so a wider stencil is appropriate.