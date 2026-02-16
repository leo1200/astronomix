import numpy as np
from astropy.constants import M_sun
from astropy import units as u
from scipy.integrate import solve_ivp


# class for the Weaver et al. (1977) stellar wind solution
# initiated with the
#          terminal velocity v_inf
#          mass loss rate M_dot
#          background density rho_0
# which can be used to calculate the density and velocity at time t.
class Weaver:
    def __init__(self, v_inf, M_dot, rho_0, p_0, num_xi=100, gamma=5 / 3):
        # input wind parameters
        ## terminal velocity of the wind
        self.v_inf = v_inf
        ## mass loss rate of the wind
        self.M_dot = M_dot
        ## background density of the ISM
        self.rho_0 = rho_0
        ## background pressure of the ISM
        self.p_0 = p_0

        # derived wind parameters
        ## mechanical luminosity of the wind
        self.L_w = 0.5 * M_dot * v_inf**2

        # constants
        self.xi_crit = 0.86
        self.alpha = 0.88
        self.gamma = gamma

        # number of points to calculate the shell profiles
        self.num_xi = num_xi

        # calculate the shell profiles
        self.calculate_shell_profiles()

    def calculate_shell_profiles(self):
        # integrate equations 6 to 7 in weaver II, with
        # 3 * (U - xi) * U' - 2U + 3 P' / G = 0
        # (U - xi) * G' / G + U' + 2U / xi = 0
        # 3 * (U - xi) * P' - 3 * gamma * P * (U - xi) * G' / G - 4P = 0
        # where ' denotes \partial_\xi
        # from xi = 1 to xi = xi_crit with G(1) = 4, U(1) = 3/4, P(1) = 3/4
        # r = R_2 * xi; rho = rho_0 * G(xi)
        # v = V_2 * U(xi); p = rho_0 * V_2^2 * P(xi)

        gamma = self.gamma

        # the equations are given by
        def shell_equation(xi, y):
            G, U, P = y
            G_prime = (
                2
                * (-2 * xi**2 * G * U + 5 * xi * G * U**2 + 2 * xi * P - 3 * G * U**3)
                * G
                / (
                    3
                    * xi
                    * (
                        gamma * xi * P
                        - gamma * P * U
                        - xi**3 * G
                        + 3 * xi**2 * G * U
                        - 3 * xi * G * U**2
                        + G * U**3
                    )
                )
            )
            U_prime = (
                2
                * (-3 * gamma * P * U + xi**2 * G * U - xi * G * U**2 + 2 * xi * P)
                / (3 * xi * (gamma * P - xi**2 * G + 2 * xi * G * U - G * U**2))
            )
            P_prime = (
                2
                * (-2 * gamma * xi * U + 3 * gamma * U**2 + 2 * xi**2 - 2 * xi * U)
                * G
                * P
                / (3 * xi * (gamma * P - xi**2 * G + 2 * xi * G * U - G * U**2))
            )
            return [G_prime, U_prime, P_prime]

        # initial conditions
        G1 = 4
        U1 = 3 / 4
        P1 = 3 / 4

        sol = solve_ivp(
            shell_equation,
            [1, self.xi_crit],
            [G1, U1, P1],
            t_eval=np.linspace(1, self.xi_crit, self.num_xi),
        )

        self.xi = np.flip(sol.t)
        self.U = np.flip(sol.y[1])
        self.G = np.flip(sol.y[0])
        self.P = np.flip(sol.y[2])

    # returns the inner shock radius R_1
    def get_inner_shock_radius(self, t):
        return (
            0.9
            * self.alpha**1.5
            * (1 / self.rho_0 * self.M_dot) ** (3 / 10)
            * self.v_inf ** (1 / 10)
            * t ** (2 / 5)
        )

    # returns the outer shock radius R_2
    def get_outer_shock_radius(self, t):
        return self.alpha * (self.L_w * t**3 / self.rho_0) ** (1 / 5)

    # returns the critical radius R_c
    def get_critical_radius(self, t):
        return self.xi_crit * self.get_outer_shock_radius(t)

    def get_radial_range_wind_interior(self, delta_R, t):
        R_1 = self.get_inner_shock_radius(t)
        R_c = self.get_critical_radius(t)
        return (
            np.arange(
                R_1.to(u.parsec).value,
                R_c.to(u.parsec).value,
                delta_R.to(u.parsec).value,
            )
            * u.parsec
        )

    def get_radial_range_free_wind(self, delta_R, t):
        R_1 = self.get_inner_shock_radius(t)
        return (
            np.arange(
                delta_R.to(u.parsec).value,
                R_1.to(u.parsec).value,
                delta_R.to(u.parsec).value,
            )
            * u.parsec
        )

    def get_radial_range_undisturbed_ism(self, delta_R, R_max, t):
        R_2 = self.get_outer_shock_radius(t)
        return (
            np.arange(
                R_2.to(u.parsec).value,
                R_max.to(u.parsec).value,
                delta_R.to(u.parsec).value,
            )
            * u.parsec
        )

    # get inner pressure
    def get_pressure_profile(self, delta_R, R_max, t):
        # free wind profile (?????)
        # Rs_free_wind = self.get_radial_range_free_wind(delta_R, t)
        # pressures_free_wind = 1/20 * self.M_dot * self.v_inf / (4 * np.pi * Rs_free_wind ** 2)

        # wind interior
        Rs_wind_interior = (
            np.array(
                [
                    self.get_inner_shock_radius(t).to(u.parsec).value,
                    self.get_critical_radius(t).to(u.parsec).value,
                ]
            )
            * u.parsec
        )
        pressure_wind_interior = (
            5
            / (22 * np.pi * (self.xi_crit * self.alpha) ** 3)
            * (self.L_w**2 * self.rho_0**3) ** (1 / 5)
            * t ** (-4 / 5)
            * np.array([1, 1])
        )

        # shell
        # as of the RH-jump conditions at a contact discontinuity,
        # the pressure is continuous across the contact discontinuity
        Rs_shell = self.xi * self.get_outer_shock_radius(t)
        V2 = 15 / 25 * self.get_critical_radius(t) / t
        pressure_shell = self.rho_0 * V2**2 * self.P
        pressure_shell = (
            pressure_shell / pressure_shell[0] * pressure_wind_interior[1]
        )  # why is this necessary? sth wrong with the solution??

        # undisturbed ISM
        Rs_undisturbed_ism = self.get_radial_range_undisturbed_ism(delta_R, R_max, t)
        pressure_undisturbed_ism = self.p_0 * np.ones(len(Rs_undisturbed_ism))

        return np.concatenate(
            (Rs_wind_interior, Rs_shell, Rs_undisturbed_ism)
        ), np.concatenate(
            (pressure_wind_interior, pressure_shell, pressure_undisturbed_ism)
        )

    # get velocity profile
    def get_velocity_profile(self, delta_R, R_max, t):
        # free wind profile
        Rs_free_wind = self.get_radial_range_free_wind(delta_R, t)
        velocities_free_wind = self.v_inf * np.ones(len(Rs_free_wind))

        # wind interior
        Rs_wind_interior = self.get_radial_range_wind_interior(delta_R, t)
        R_c = self.get_critical_radius(t)
        velocities_wind_interior = (
            11 / 25 * R_c**3 / (Rs_wind_interior**2 * t) + 4 / 25 * Rs_wind_interior / t
        )

        # shell
        Rs_shell = self.xi * self.get_outer_shock_radius(t)
        # by the RH-jump conditions at a contact discontinuity,
        # the velocity at the beginning of the shell is the
        # velocity at the end of the wind interior, so
        V2 = 15 / 25 * R_c / t
        velocities_shell = V2 * self.U / self.U[0]

        # undisturbed ISM
        # here the velocity is 0
        Rs_undisturbed_ism = self.get_radial_range_undisturbed_ism(delta_R, R_max, t)
        velocities_unisturbed_ism = np.zeros(len(Rs_undisturbed_ism))

        return np.concatenate(
            (Rs_free_wind, Rs_wind_interior, Rs_shell, Rs_undisturbed_ism)
        ), np.concatenate(
            (
                velocities_free_wind,
                velocities_wind_interior,
                velocities_shell,
                velocities_unisturbed_ism,
            )
        )

    # get density profile
    def get_density_profile(self, delta_R, R_max, t):
        # free wind profile
        Rs_free_wind = self.get_radial_range_free_wind(delta_R, t)
        densities_free_wind = self.M_dot / (4 * np.pi * Rs_free_wind**2 * self.v_inf)

        # profile in the wind interior
        Rs_wind_interior = self.get_radial_range_wind_interior(delta_R, t)
        R_c = self.get_critical_radius(t)
        densities_wind_interior = (
            0.628
            * (self.M_dot**2 * self.rho_0**3 * self.v_inf ** (-6)) ** (1 / 5)
            * t ** (-4 / 5)
            * (1 - (Rs_wind_interior / R_c) ** 3) ** (-8 / 33)
        )

        # profile in the shell
        Rs_shell = self.xi * self.get_outer_shock_radius(t)
        densities_shell = self.rho_0 * self.G

        # profile in the undisturbed ISM
        Rs_undisturbed_ism = self.get_radial_range_undisturbed_ism(delta_R, R_max, t)
        densities_unisturbed_ism = self.rho_0 * np.ones(len(Rs_undisturbed_ism))

        return np.concatenate(
            (Rs_free_wind, Rs_wind_interior, Rs_shell, Rs_undisturbed_ism)
        ), np.concatenate(
            (
                densities_free_wind,
                densities_wind_interior,
                densities_shell,
                densities_unisturbed_ism,
            )
        )
