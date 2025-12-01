## Unit helpers

# The following functions are used to convert from
# units of the simulation code to physical units.
# When using Paicos this is not necessary.

# ===== imports =======
from astropy import units as u
from astropy import constants as c
# =====================


# ============ CODE UNITS CLASS ============

class CodeUnits:
    
    def __init__(self, unit_length, unit_mass, unit_velocity):
        # expects input in astropy units
        # e.g. unit_length = 3 * u.parsec
        #      unit_mass = 1e5 * u.M_sun
        #      unit_velocity = 1 * u.km / u.s

        self.code_length = u.def_unit('code_length', unit_length)
        self.code_mass = u.def_unit('code_mass', unit_mass)
        self.code_velocity = u.def_unit('code_velocity', unit_velocity)

        self.code_time = self.code_length / self.code_velocity
        self.code_density = self.code_mass / self.code_length**3
        self.code_pressure = self.code_mass / self.code_length / self.code_time**2
        self.code_energy = self.code_mass * self.code_velocity**2

        # B / sqrt(\mu_0) has units of sqrt(pressure)
        self.code_magnetic_field = self.code_pressure ** 0.5

    def init_from_unit_params(UnitLength_in_cm, UnitMass_in_g, UnitVelocity_in_cm_per_s):
        return CodeUnits(UnitLength_in_cm * u.cm, UnitMass_in_g * u.g, UnitVelocity_in_cm_per_s * u.cm / u.s)
    
    def get_temperature_from_internal_energy(self, internal_energy, gamma = 5 / 3, hydrogen_abundance = 0.76):
        mhydrogen = c.m_e + c.m_p
        gm1 = gamma - 1
        mean_molecular_weight = 4 / (5 * hydrogen_abundance + 3)
        return (gm1 * internal_energy * mean_molecular_weight * mhydrogen * self.code_velocity**2 / c.k_B).to(u.K)
    
    def print_simulation_parameters(self, final_time_wanted):
        print(f"Code length in cm: {self.code_length.to(u.cm)}")
        print(f"Code mass in g: {self.code_mass.to(u.g)}")
        print(f"Code velocity in cm/s: {self.code_velocity.to(u.cm / u.s)}")
        print(f"Code time in s: {self.code_time.to(u.s)}")
        print(f"Final time in code units: {final_time_wanted.to(self.code_time)}")
        print(f"Code density in g/cm^3: {self.code_density.to(u.g / u.cm**3)}")
        print(f"Code pressure in g/cm/s^2: {self.code_pressure.to(u.g / u.cm / u.s**2)}")
