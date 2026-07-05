"""
Conversion between simulation code units and physical units.

The simulation works in dimensionless code units defined by a chosen unit
length, unit mass and unit velocity. :class:`CodeUnits` builds the derived
code units (time, density, pressure, energy, magnetic field) from those three
base units and offers convenience conversions back to physical quantities.
When post-processing with Paicos these helpers are not needed.
"""

# units and constants
from astropy import constants as c
from astropy import units as u


class CodeUnits:
    """Derived code-unit system built from a base length, mass and velocity.

    From the three base units every other code unit (time, density, pressure,
    energy, magnetic field) follows by dimensional analysis. The magnetic field
    is stored as ``B / sqrt(mu_0)``, which has units of ``sqrt(pressure)``.
    """

    def __init__(self, unit_length, unit_mass, unit_velocity):
        """Define the code-unit system from three astropy-valued base units.

        Args:
            unit_length: The code length, e.g. ``3 * u.parsec``.
            unit_mass: The code mass, e.g. ``1e5 * u.M_sun``.
            unit_velocity: The code velocity, e.g. ``1 * u.km / u.s``.
        """
        self.code_length = u.def_unit('code_length', unit_length)
        self.code_mass = u.def_unit('code_mass', unit_mass)
        self.code_velocity = u.def_unit('code_velocity', unit_velocity)

        # The remaining code units are fixed by the three base units above.
        self.code_time = self.code_length / self.code_velocity
        self.code_density = self.code_mass / self.code_length**3
        self.code_pressure = self.code_mass / self.code_length / self.code_time**2
        self.code_energy = self.code_mass * self.code_velocity**2

        # The code stores the magnetic field as B / sqrt(mu_0), which carries
        # units of sqrt(pressure).
        self.code_magnetic_field = self.code_pressure ** 0.5

    def init_from_unit_params(UnitLength_in_cm, UnitMass_in_g, UnitVelocity_in_cm_per_s):
        """Build a :class:`CodeUnits` from raw CGS unit magnitudes.

        Args:
            UnitLength_in_cm: The code length in centimetres.
            UnitMass_in_g: The code mass in grams.
            UnitVelocity_in_cm_per_s: The code velocity in cm/s.

        Returns:
            The corresponding :class:`CodeUnits` instance.
        """
        return CodeUnits(
            UnitLength_in_cm * u.cm,
            UnitMass_in_g * u.g,
            UnitVelocity_in_cm_per_s * u.cm / u.s,
        )

    def get_temperature_from_internal_energy(self, internal_energy, gamma = 5 / 3, hydrogen_abundance = 0.76):
        """Convert specific internal energy (code units) to a temperature.

        Uses the ideal-gas relation ``T = (gamma - 1) u mu m_H / k_B`` with the
        mean molecular weight derived from the hydrogen abundance.

        Args:
            internal_energy: The specific internal energy in code units.
            gamma: The adiabatic index of the gas.
            hydrogen_abundance: The hydrogen mass fraction, used to set the mean
                molecular weight.

        Returns:
            The temperature as an astropy quantity in kelvin.
        """
        mhydrogen = c.m_e + c.m_p
        gm1 = gamma - 1
        mean_molecular_weight = 4 / (5 * hydrogen_abundance + 3)
        return (gm1 * internal_energy * mean_molecular_weight * mhydrogen * self.code_velocity**2 / c.k_B).to(u.K)

    def print_simulation_parameters(self, final_time_wanted):
        """Print the base and derived code units in physical units.

        Args:
            final_time_wanted: A target physical time, reported in code-time
                units for convenience when choosing a run length.
        """
        print(f"Code length in cm: {self.code_length.to(u.cm)}")
        print(f"Code mass in g: {self.code_mass.to(u.g)}")
        print(f"Code velocity in cm/s: {self.code_velocity.to(u.cm / u.s)}")
        print(f"Code time in s: {self.code_time.to(u.s)}")
        print(f"Final time in code units: {final_time_wanted.to(self.code_time)}")
        print(f"Code density in g/cm^3: {self.code_density.to(u.g / u.cm**3)}")
        print(f"Code pressure in g/cm/s^2: {self.code_pressure.to(u.g / u.cm / u.s**2)}")
