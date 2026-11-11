"""
The variable registry: where in the state array each field is stored.

Keeping the layout of the state array in one place is what makes the code
modular and readable; every consumer indexes the state through the registry
rather than through hard-coded positions. New variables, e.g. densities of
chemical species, are registered here.

NOTE: For MHD the cell-centred magnetic field occupies three consecutive slots
behind the gas variables. The finite-volume VL2 scheme appends the cleaning
scalar psi or, with constrained transport, the interface (face-centred) field;
the finite-difference solver appends the interface field and may append
further rows behind the interface field: first the dual-energy internal-energy
density ``g``, then the passive-scalar block (the user's scalars followed by the
shock-history scalars). ``_evolve_state_fd`` splits these trailing rows off
before the hydro / MHD update, so on the state that update sees the interface
field is again in the last three slots.
"""

# typing
from typing import (
    NamedTuple,
    Union,
)

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    IDEAL_GAS,
    ISOTHERMAL,
    VL2,
)

# astronomix containers
from astronomix.option_classes.simulation_config import (
    SimulationConfig,
    StaticIntVector,
)

# astronomix functions
from astronomix.option_classes.simulation_config import solver_mode_to_string


#: Number of library-managed scalars appended to the passive-scalar block when
#: ``config.track_shock_history`` is set. Defined here rather than in
#: ``_passive_scalars`` because that module imports :class:`RegisteredVariables`
#: from this one.
NUM_SHOCK_HISTORY_SCALARS = 4

#: Positions of the shock-history scalars within their block, which starts at
#: ``RegisteredVariables.shock_history_index`` (their meaning is documented in
#: ``_passive_scalars``). The two accumulators are the last two slots.
ENTROPY_INITIAL_SLOT = 0
SHOCKED_FRACTION_SLOT = 1
TIME_SINCE_SHOCK_SLOT = 2
DENSITY_TIME_SLOT = 3


# =============================================================

# Each spatial dimension (e.g. the x-axis) corresponds both to an axis in the
# state array (along which that coordinate varies) and to the fields tied to
# that axis (e.g. the x-velocity or the x-component of the magnetic field). A
# common pattern is a loop over the spatial dimensions that needs exactly this
# mapping, which ``AxisInfo`` bundles together.


class AxisInfo(NamedTuple):
    """The array axis and field indices associated with one spatial dimension.

    Attributes:
        axis_in_array: The axis in the state array along which this coordinate
            varies.
        velocity_index: The index of the velocity component along this axis.
        magnetic_index: The index of the magnetic field component along this axis.
    """

    axis_in_array: int
    velocity_index: int
    magnetic_index: int

# =============================================================


class RegisteredVariables(NamedTuple):
    """
    The registered variables are the variables that are
    stored in the state array. The order of the variables
    in the state array is important and should be consistent
    throughout the code.
    """

    #: Number of variables
    num_vars: int = 3

    # Baseline variables

    #: Density index
    density_index: int = 0

    #: Velocity index
    velocity_index: Union[int, StaticIntVector] = 1
    # in e.g. 3D, we have three velocity components, each with its own index

    #: Momentum density index, same as velocity index
    #: introduced for readability when dealing with
    #: the conserved state
    momentum_index: Union[int, StaticIntVector] = 1

    #: Magnetic field index
    magnetic_index: Union[int, StaticIntVector] = -1

    #: Magnetic field at interfaces index
    #: used in finite difference MHD constrained transport
    interface_magnetic_field_index: Union[int, StaticIntVector] = -1

    #: Index of the GLM divergence-cleaning scalar psi (Dedner et al. 2002),
    #: carried by the VL2 finite-volume MHD scheme after the magnetic field
    #: (AthenaPK's ``IPS`` slot); -1 when inactive. ``magnetic_psi_active``
    #: marks this cell-centred VL2 GLM-MHD layout, which stores all three
    #: velocity and field components in any dimensionality.
    magnetic_psi_index: int = -1
    magnetic_psi_active: bool = False

    #: Pressure index
    pressure_index: int = 2

    #: Energy index, same as pressure index
    #: introduced for readability when dealing with
    #: the conserved state.
    energy_index: int = 2

    # Additional variables, these
    # have to be registered

    #: stellar wind density index
    wind_density_index: int = -1
    wind_density_active: bool = False

    #: Dual-energy internal-energy density ``g = rho e`` (Bryan et al. 1995).
    #: Stored behind all other variables except the passive scalars (for MHD
    #: behind the interface magnetic field). ``_evolve_state_fd`` splits it off
    #: (after the passive scalars), so the ``[:-3]`` interface-field convention
    #: holds on the state the MHD update sees, advects it and uses it in the
    #: coupled pressure recovery. Active only for finite-difference ideal-gas
    #: (hydro or MHD) runs with ``config.dual_energy``.
    internal_energy_index: int = -1
    internal_energy_active: bool = False

    #: Passive scalars: per-parcel labels advected with the flow without acting
    #: back on it (composition mass fractions, an ejecta / circumstellar
    #: discriminator, the shock-history bookkeeping). Stored as one contiguous
    #: block at the very end of the state array, behind the dual-energy ``g``
    #: and the MHD interface magnetic field, so stripping the block off leaves
    #: a state the hydro / MHD machinery already understands.
    #: ``_evolve_state_fd`` strips the block before the update, advects it
    #: operator-split and reattaches it. ``num_passive_scalars`` counts the
    #: user's scalars plus, when ``config.track_shock_history`` is set, the
    #: ``NUM_SHOCK_HISTORY_SCALARS`` (four) library-managed shock-history
    #: scalars, which form the end of the block starting at
    #: ``shock_history_index`` (see the ``*_SLOT`` constants for their order).
    passive_scalar_index: int = -1
    num_passive_scalars: int = 0
    passive_scalars_active: bool = False
    shock_history_index: int = -1
    shock_history_active: bool = False

    # here you can add more variables


def get_registered_variables(config: SimulationConfig) -> RegisteredVariables:
    """Build the variable registry for a given simulation configuration.

    Starts from the baseline (density, velocity, pressure) registry and grows /
    re-indexes it for the active solver mode, dimensionality, equation of state
    and any extra tracked fields (MHD, stellar-wind density, the
    dual-energy internal energy, passive scalars and the shock history), so
    that every field lands at the index the rest of the code expects.

    Args:
        config: The simulation configuration.

    Returns:
        The registered variables.
    """

    registered_variables = RegisteredVariables()

    if config.solver_mode == FINITE_VOLUME and config.mhd and config.time_integrator == VL2:

        # The AthenaPK-equivalent GLM-MHD scheme always carries all three
        # velocity and field components (also in 1D and 2D), followed by the
        # divergence-cleaning scalar psi: (rho, v_x, v_y, v_z, p, B_x, B_y, B_z,
        # psi), AthenaPK's variable order.
        registered_variables = RegisteredVariables(
            density_index=0,
            velocity_index=StaticIntVector(1, 2, 3),
            pressure_index=4,
            magnetic_index=StaticIntVector(5, 6, 7),
            magnetic_psi_index=8,
            magnetic_psi_active=True,
            num_vars=9,
        )
        # NOTE: this layout does not register the stellar-wind density; the
        # VL2 scheme does not evolve it.

        if config.vl2_constrained_transport:
            # VL2 with constrained transport: no cleaning scalar; the
            # face-centred field (the primary magnetic variable, index i
            # holding the value at i + 1/2 as in the finite-difference layout)
            # follows the cell-centred one: (rho, v_x, v_y, v_z, p, B_x, B_y,
            # B_z, b_x, b_y, b_z).
            registered_variables = RegisteredVariables(
                density_index=0,
                velocity_index=StaticIntVector(1, 2, 3),
                pressure_index=4,
                magnetic_index=StaticIntVector(5, 6, 7),
                interface_magnetic_field_index=StaticIntVector(8, 9, 10),
                num_vars=11,
            )

    elif config.solver_mode == FINITE_VOLUME:

        if config.mhd:
            raise ValueError(
                "Finite-volume MHD is available with the VL2 scheme only "
                "(time_integrator=VL2)."
            )

        if config.dimensionality == 2:
            # we have two velocity components
            registered_variables = registered_variables._replace(
                num_vars=registered_variables.num_vars + 1
            )

            # update the velocity index
            registered_variables = registered_variables._replace(
                velocity_index=StaticIntVector(1, 2, -1)
            )

            # update the pressure index
            registered_variables = registered_variables._replace(
                pressure_index=registered_variables.num_vars - 1
            )

        if config.dimensionality == 3:
            # we have three velocity components
            registered_variables = registered_variables._replace(
                num_vars=registered_variables.num_vars + 2
            )

            # update the velocity index to be an array
            registered_variables = registered_variables._replace(
                velocity_index=StaticIntVector(1, 2, 3)
            )

            # update the pressure index
            registered_variables = registered_variables._replace(
                pressure_index=registered_variables.num_vars - 1
            )

        # NOTE: CURRENTLY ONLY IMPLEMENTED FOR FINITE VOLUME MODE
        if config.wind_config.trace_wind_density:
            registered_variables = registered_variables._replace(
                wind_density_index=registered_variables.num_vars
            )
            registered_variables = registered_variables._replace(
                num_vars=registered_variables.num_vars + 1
            )
            registered_variables = registered_variables._replace(wind_density_active=True)

    if config.solver_mode == FINITE_DIFFERENCE:

        if config.mhd:

            # The finite-difference MHD update always carries all three velocity
            # components (even in 1D and 2D) for the magnetic field update, so
            # the registry is set explicitly per equation of state rather than
            # derived from the dimensionality as in the hydrodynamics case.
            # NOTE: the cell-centred magnetic field is stored before the
            # interface magnetic field, which occupies the last three slots of
            # this layout; the dual-energy g and the passive scalars, if any,
            # are appended behind it below.
            if config.equation_of_state == IDEAL_GAS:
                registered_variables = RegisteredVariables(
                    density_index=0,
                    velocity_index=StaticIntVector(1, 2, 3),
                    pressure_index=4,
                    magnetic_index=StaticIntVector(5, 6, 7),
                    interface_magnetic_field_index=StaticIntVector(8, 9, 10),
                    num_vars=11,
                )
            elif config.equation_of_state == ISOTHERMAL:
                registered_variables = RegisteredVariables(
                    density_index=0,
                    velocity_index=StaticIntVector(1, 2, 3),
                    pressure_index=-1,
                    magnetic_index=StaticIntVector(4, 5, 6),
                    interface_magnetic_field_index=StaticIntVector(7, 8, 9),
                    num_vars=10,
                )
        else:
            # redundant with the FINITE_VOLUME case
            # included for readability
            if config.dimensionality == 1:
                registered_variables = RegisteredVariables(
                    density_index=0,
                    velocity_index=1,
                    pressure_index=2,
                    num_vars=3,
                )
            elif config.dimensionality == 2:
                registered_variables = RegisteredVariables(
                    density_index=0,
                    velocity_index=StaticIntVector(1, 2),
                    pressure_index=3,
                    num_vars=4,
                )
            elif config.dimensionality == 3:
                registered_variables = RegisteredVariables(
                    density_index=0,
                    velocity_index=StaticIntVector(1, 2, 3),
                    pressure_index=4,
                    num_vars=5,
                )

            if config.equation_of_state == ISOTHERMAL:
                registered_variables = registered_variables._replace(
                    pressure_index=-1
                )
                registered_variables = registered_variables._replace(
                    num_vars=registered_variables.num_vars - 1
                )

        # Dual-energy formalism: the internal-energy density ``g`` goes behind
        # everything registered so far (for hydro simply behind the pressure,
        # for MHD behind the interface magnetic field).
        if config.equation_of_state == IDEAL_GAS and config.dual_energy:
            registered_variables = registered_variables._replace(
                internal_energy_index=registered_variables.num_vars,
                num_vars=registered_variables.num_vars + 1,
                internal_energy_active=True,
            )

        # Passive scalars: one contiguous block behind everything else, so that
        # stripping it off leaves a state the hydro / MHD machinery already
        # understands. The library-managed shock-history scalars form the end
        # of the block, behind the user's.
        num_scalars = int(config.num_passive_scalars)
        if config.track_shock_history:
            num_scalars += NUM_SHOCK_HISTORY_SCALARS
        if num_scalars > 0:
            passive_scalar_index = registered_variables.num_vars
            if config.track_shock_history:
                shock_history_index = (
                    passive_scalar_index + num_scalars - NUM_SHOCK_HISTORY_SCALARS
                )
            else:
                shock_history_index = -1
            registered_variables = registered_variables._replace(
                passive_scalar_index=passive_scalar_index,
                num_passive_scalars=num_scalars,
                num_vars=registered_variables.num_vars + num_scalars,
                passive_scalars_active=True,
                shock_history_index=shock_history_index,
                shock_history_active=bool(config.track_shock_history),
            )

    elif config.num_passive_scalars > 0 or config.track_shock_history:
        raise NotImplementedError(
            "passive scalars are implemented for the finite-difference solver "
            "only (the finite-volume Riemann solvers would each need a scalar "
            "flux); got solver_mode = "
            f"{solver_mode_to_string(config.solver_mode)}"
        )

    # shorthands
    registered_variables = registered_variables._replace(
        momentum_index=registered_variables.velocity_index
    )
    registered_variables = registered_variables._replace(
        energy_index=registered_variables.pressure_index
    )

    # here you can register more variables

    return registered_variables
