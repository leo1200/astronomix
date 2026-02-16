# TOWNSEND SCHEME DOES NOT WORK CURRENTLY

from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

from astronomix._physics_modules._cooling.cooling_options import PiecewisePowerLawParams
from astronomix.units.unit_helpers import CodeUnits

from astropy import units as u
import astropy.constants as c
from astropy.constants import m_p

def schure_cooling(
    code_units: CodeUnits,
):

    # high temperature cooling table from
    # https://arxiv.org/pdf/0909.5204
    # the paper also includes the low temperature
    # cooling curve from
    # Dalgarno & McCray (1972)

    # T in K
    log10_T = np.array([
        3.80, 3.84, 3.88, 3.92, 3.96, 4.00, 4.04, 4.08, 4.12, 4.16,
        4.20, 4.24, 4.28, 4.32, 4.36, 4.40, 4.44, 4.48, 4.52, 4.56,
        4.60, 4.64, 4.68, 4.72, 4.76, 4.80, 4.84, 4.88, 4.92, 4.96,
        5.00, 5.04, 5.08, 5.12, 5.16, 5.20, 5.24, 5.28, 5.32, 5.36,
        5.40, 5.44, 5.48, 5.52, 5.56, 5.60, 5.64, 5.68, 5.72, 5.76,
        5.80, 5.84, 5.88, 5.92, 5.96, 6.00, 6.04, 6.08, 6.12, 6.16,
        6.20, 6.24, 6.28, 6.32, 6.36, 6.40, 6.44, 6.48, 6.52, 6.56,
        6.60, 6.64, 6.68, 6.72, 6.76, 6.80, 6.84, 6.88, 6.92, 6.96,
        7.00, 7.04, 7.08, 7.12, 7.16, 7.20, 7.24, 7.28, 7.32, 7.36,
        7.40, 7.44, 7.48, 7.52, 7.56, 7.60, 7.64, 7.68, 7.72, 7.76,
        7.80, 7.84, 7.88, 7.92, 7.96, 8.00, 8.04, 8.08, 8.12, 8.16
    ])

    T = 10**log10_T

    # convert T to code units
    T = jnp.array((T * u.K * c.k_B / c.m_p).to(code_units.code_energy / code_units.code_mass).value)

    log10_T = jnp.log10(T)

    reference_temperature = T[-1]

    # Lambda in erg cm^3 / s
    log10_Lambda = np.array([
        -25.7331, -25.0383, -24.4059, -23.8288, -23.3027, -22.8242, -22.3917, -22.0067, -21.6818, -21.4529,
        -21.3246, -21.3459, -21.4305, -21.5293, -21.6138, -21.6615, -21.6551, -21.5919, -21.5092, -21.4124,
        -21.3085, -21.2047, -21.1067, -21.0194, -20.9413, -20.8735, -20.8205, -20.7805, -20.7547, -20.7455,
        -20.7565, -20.7820, -20.8008, -20.7994, -20.7847, -20.7687, -20.7590, -20.7544, -20.7505, -20.7545,
        -20.7888, -20.8832, -21.0450, -21.2286, -21.3737, -21.4573, -21.4935, -21.5098, -21.5345, -21.5863,
        -21.6548, -21.7108, -21.7424, -21.7576, -21.7696, -21.7883, -21.8115, -21.8303, -21.8419, -21.8514,
        -21.8690, -21.9057, -21.9690, -22.0554, -22.1488, -22.2355, -22.3084, -22.3641, -22.4033, -22.4282,
        -22.4408, -22.4443, -22.4411, -22.4334, -22.4242, -22.4164, -22.4134, -22.4168, -22.4267, -22.4418,
        -22.4603, -22.4830, -22.5112, -22.5449, -22.5819, -22.6177, -22.6483, -22.6719, -22.6883, -22.6985,
        -22.7032, -22.7037, -22.7008, -22.6950, -22.6869, -22.6769, -22.6655, -22.6531, -22.6397, -22.6258,
        -22.6111, -22.5964, -22.5816, -22.5668, -22.5519, -22.5367, -22.5216, -22.5062, -22.4912, -22.4753
    ])

    Lambda = 10**log10_Lambda

    # convert Lambda to code units
    Lambda = jnp.array(
        (
            Lambda * u.erg * u.cm ** 3 / u.s / c.m_p ** 2
        ).to(
            code_units.code_energy * code_units.code_length ** 3 / (code_units.code_time * code_units.code_mass ** 2)
    ).value)

    log10_Lambda = jnp.log10(Lambda)

    # piecewise power law fits in the form
    # Lambda(T) = Lambda_k * (T / T_k)^alpha_k for T_k <= T < T_{k+1}
    # with T in K and Lambda in erg cm^3 / s
    alpha = (log10_Lambda[1:] - log10_Lambda[:-1]) / (log10_T[1:] - log10_T[:-1])

    # coefficients Y_k, following Eq. A6 in Townsend 2009
    Y_table = jnp.zeros_like(T)
    Y_table = Y_table.at[-1].set(0.0) # Y_n = 0
    Lambda_N = Lambda[-1]
    T_N = T[-1]
    for k in range(len(alpha) - 1, -1, -1):
        T_k = T[k]
        T_k1 = T[k + 1]
        Lambda_k = Lambda[k]
        alpha_k = alpha[k]

        Y_table = Y_table.at[k].set(
            Y_table[k + 1] - jnp.where(
                alpha_k != 1.0,
                1 / (1 - alpha_k) * Lambda_N / Lambda_k * T_k / T_N * (1 - (T_k / T_k1) ** (alpha_k - 1)),
                Lambda_N / Lambda_k * T_k / T_N * jnp.log(T_k / T_k1),
            )
        )

    return PiecewisePowerLawParams(
        log10_T_table = log10_T,
        log10_Lambda_table = log10_Lambda,
        alpha_table = alpha,
        Y_table = Y_table,
        reference_temperature = reference_temperature
    )