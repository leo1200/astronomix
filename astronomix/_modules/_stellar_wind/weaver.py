"""
Analytic Weaver et al. (1977) stellar-wind bubble solution.

Provides the :class:`Weaver` self-similar solution for a wind-blown bubble:
given the wind terminal velocity, mass-loss rate and the ambient ISM density /
pressure, it integrates the shell structure equations and exposes the radial
density, velocity and pressure profiles (free wind, shocked wind interior,
swept-up shell and undisturbed ISM) at a given time.
"""

# numerics
import numpy as np
from scipy.integrate import solve_ivp

# units and constants
from astropy.constants import M_sun
from astropy import units as u


class Weaver:
    """
    Weaver et al. (1977) self-similar stellar-wind bubble solution.

    Initialised from the wind terminal velocity ``v_inf``, the mass-loss rate
    ``M_dot`` and the ambient ISM density ``rho_0`` / pressure ``p_0``; the
    resulting object yields the density, velocity and pressure profiles of the
    wind bubble at any time ``t``.
    """

    def __init__(self, v_inf, M_dot, rho_0, p_0, num_xi=100, gamma=5 / 3):
        # Input wind parameters: terminal velocity, mass-loss rate, and the
        # ambient ISM density and pressure.
        self.v_inf = v_inf
        self.M_dot = M_dot
        self.rho_0 = rho_0
        self.p_0 = p_0

        # Derived wind parameter: the mechanical luminosity of the wind.
        self.L_w = 0.5 * M_dot * v_inf**2

        # Self-similar constants of the Weaver solution.
        self.xi_crit = 0.86
        self.alpha = 0.88
        self.gamma = gamma

        # Number of points used to sample the shell profiles.
        self.num_xi = num_xi

        # Integrate the shell structure equations once on construction.
        self.calculate_shell_profiles()

    def calculate_shell_profiles(self):
        """
        Integrate the self-similar shell structure equations.

        Integrates equations 6 to 7 of Weaver II inward from the outer shock
        (``xi = 1``) to the critical radius (``xi = xi_crit``):

            3 (U - xi) U' - 2 U + 3 P' / G = 0
            (U - xi) G' / G + U' + 2 U / xi = 0
            3 (U - xi) P' - 3 gamma P (U - xi) G' / G - 4 P = 0

        where a prime denotes a derivative with respect to xi, with boundary
        conditions G(1) = 4, U(1) = 3/4, P(1) = 3/4. The dimensionless profiles
        relate to physical quantities by r = R_2 xi, rho = rho_0 G(xi),
        v = V_2 U(xi) and p = rho_0 V_2^2 P(xi). The results are stored on the
        instance ordered from the critical radius outward.
        """
        gamma = self.gamma

        # Right-hand side of the self-similar shell ODE system.
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

        # Boundary conditions at the outer shock (xi = 1).
        G1 = 4
        U1 = 3 / 4
        P1 = 3 / 4

        sol = solve_ivp(
            shell_equation,
            [1, self.xi_crit],
            [G1, U1, P1],
            t_eval=np.linspace(1, self.xi_crit, self.num_xi),
        )

        # Store the profiles ordered from the critical radius outward.
        self.xi = np.flip(sol.t)
        self.U = np.flip(sol.y[1])
        self.G = np.flip(sol.y[0])
        self.P = np.flip(sol.y[2])

    def get_inner_shock_radius(self, t):
        """Return the inner (wind) shock radius R_1 at time ``t``."""
        return (
            0.9
            * self.alpha**1.5
            * (1 / self.rho_0 * self.M_dot) ** (3 / 10)
            * self.v_inf ** (1 / 10)
            * t ** (2 / 5)
        )

    def get_outer_shock_radius(self, t):
        """Return the outer shock radius R_2 at time ``t``."""
        return self.alpha * (self.L_w * t**3 / self.rho_0) ** (1 / 5)

    def get_critical_radius(self, t):
        """Return the critical radius R_c at time ``t``."""
        return self.xi_crit * self.get_outer_shock_radius(t)

    def get_radial_range_wind_interior(self, delta_R, t):
        """Return the radial sampling points across the shocked wind interior."""
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
        """Return the radial sampling points across the freely expanding wind."""
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
        """Return the radial sampling points across the undisturbed ISM."""
        R_2 = self.get_outer_shock_radius(t)
        return (
            np.arange(
                R_2.to(u.parsec).value,
                R_max.to(u.parsec).value,
                delta_R.to(u.parsec).value,
            )
            * u.parsec
        )

    def get_pressure_profile(self, delta_R, R_max, t):
        """
        Return the radial pressure profile of the wind bubble at time ``t``.

        Concatenates the (uniform) shocked-wind-interior pressure, the shell
        pressure and the undisturbed-ISM pressure, returning the matching
        radii alongside the pressures.
        """
        # Shocked wind interior: bracketed by the inner shock and the critical
        # radius, with a uniform interior pressure.
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

        # Shell. By the Rankine-Hugoniot jump conditions the pressure is
        # continuous across the contact discontinuity, so the shell pressure is
        # renormalised to match the wind-interior pressure at its inner edge.
        Rs_shell = self.xi * self.get_outer_shock_radius(t)
        V2 = 15 / 25 * self.get_critical_radius(t) / t
        pressure_shell = self.rho_0 * V2**2 * self.P
        # NOTE: this explicit renormalisation should not be needed if the
        # self-similar solution were exact; it compensates a small inconsistency
        # at the contact discontinuity.
        pressure_shell = pressure_shell / pressure_shell[0] * pressure_wind_interior[1]

        # Undisturbed ISM: the ambient pressure beyond the outer shock.
        Rs_undisturbed_ism = self.get_radial_range_undisturbed_ism(delta_R, R_max, t)
        pressure_undisturbed_ism = self.p_0 * np.ones(len(Rs_undisturbed_ism))

        return np.concatenate(
            (Rs_wind_interior, Rs_shell, Rs_undisturbed_ism)
        ), np.concatenate(
            (pressure_wind_interior, pressure_shell, pressure_undisturbed_ism)
        )

    def get_velocity_profile(self, delta_R, R_max, t):
        """
        Return the radial velocity profile of the wind bubble at time ``t``.

        Concatenates the free wind, the shocked wind interior, the shell and the
        (zero-velocity) undisturbed ISM, returning the matching radii alongside
        the velocities.
        """
        # Free wind: constant terminal velocity inside the inner shock.
        Rs_free_wind = self.get_radial_range_free_wind(delta_R, t)
        velocities_free_wind = self.v_inf * np.ones(len(Rs_free_wind))

        # Shocked wind interior.
        Rs_wind_interior = self.get_radial_range_wind_interior(delta_R, t)
        R_c = self.get_critical_radius(t)
        velocities_wind_interior = (
            11 / 25 * R_c**3 / (Rs_wind_interior**2 * t) + 4 / 25 * Rs_wind_interior / t
        )

        # Shell. By the Rankine-Hugoniot jump conditions the velocity at the
        # inner edge of the shell equals the velocity at the end of the wind
        # interior, which fixes the normalisation V2.
        Rs_shell = self.xi * self.get_outer_shock_radius(t)
        V2 = 15 / 25 * R_c / t
        velocities_shell = V2 * self.U / self.U[0]

        # Undisturbed ISM: at rest beyond the outer shock.
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

    def get_density_profile(self, delta_R, R_max, t):
        """
        Return the radial density profile of the wind bubble at time ``t``.

        Concatenates the free wind, the shocked wind interior, the shell and the
        undisturbed ISM, returning the matching radii alongside the densities.
        """
        # Free wind: the r^-2 density profile of a steady spherical wind.
        Rs_free_wind = self.get_radial_range_free_wind(delta_R, t)
        densities_free_wind = self.M_dot / (4 * np.pi * Rs_free_wind**2 * self.v_inf)

        # Shocked wind interior.
        Rs_wind_interior = self.get_radial_range_wind_interior(delta_R, t)
        R_c = self.get_critical_radius(t)
        densities_wind_interior = (
            0.628
            * (self.M_dot**2 * self.rho_0**3 * self.v_inf ** (-6)) ** (1 / 5)
            * t ** (-4 / 5)
            * (1 - (Rs_wind_interior / R_c) ** 3) ** (-8 / 33)
        )

        # Swept-up shell.
        Rs_shell = self.xi * self.get_outer_shock_radius(t)
        densities_shell = self.rho_0 * self.G

        # Undisturbed ISM: the ambient density beyond the outer shock.
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
