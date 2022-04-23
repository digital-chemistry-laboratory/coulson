"""Functions related to Slater orbitals."""

from __future__ import annotations

import math

import numpy as np
import scipy.interpolate
import scipy.misc

from coulson.parameters import (
    CHARGES,
    ELECTRON_CONFIGURATIONS,
    N,
    N_ELECTRONS,
    N_STAR,
    Z_EFF,
)
from coulson.typing import Array1DFloat


def z_eff_slater(
    electrons: dict[int, int], charge: float = 0, population: float = 1
) -> float:
    """Returns effective charge for p orbital according to Slater's rules.

    Args:
        electrons: Electron configuration as numbers per principal shell
        charge: Atom base charge
        population: Number of pi electrons in the orbital

    Returns:
        z_eff: Effective charge
    """
    # Get highest principal quantum number
    n_max = max(electrons.keys())

    # Get total charge
    z = sum(electrons.values()) + charge + 1

    # Calcualte shielding
    s = 0.0
    for n, n_electrons in electrons.items():
        if n < n_max - 1:
            s += n_electrons * 1
        if n == n_max - 1:
            s += n_electrons * 0.85
        if n == n_max:
            s += (n_electrons + population - 1) * 0.35

    # Calculate effective charge
    z_eff = z - s

    return z_eff


def _overlap_from_interpolation(
    n_1: int, n_2: int, p: float, t: float, dp: int = 0
) -> float:
    """Calculates STO overlap interals from iterpolation.

    Args:
        n_1: Principal quantum number of orbital 1.
        n_2: Principal quantum number of orbital 2.
        p: p parameter
        t: t parameter
        dp: Order of derivative for p parameter

    Returns:
        overlap: Overlap integral (dp == 0) or overlap integral derivative (dp > 0)
    """
    # Load spline interpolator
    spl = spl = spls.setdefault(
        (n_1, n_2), scipy.interpolate.SmoothBivariateSpline._from_tck(TCK[(n_1, n_2)])
    )

    overlap: float = spl.ev([p], [t], dx=dp)[0]

    return overlap


def _overlap_from_formula(  # noqa: C901
    n_1: int, n_2: int, p: float, t: float
) -> float:
    """Calculates STO overlap interals from formulas.

    Formulas taken from 10.1063/1.1747150.

    Args:
        n_1: Principal quantum number of orbital 1.
        n_2: Principal quantum number of orbital 2.
        p: p parameter
        t: t parameter

    Returns:
        overlap: Overlap integral
    """
    # Use special formulas for p = 0
    overlap: float
    if p == 0:
        if n_1 == n_2 == 2:
            overlap = (1 - t**2) ** (5 / 2)
        elif n_1 == n_2 == 3:
            overlap = (1 - t**2) ** (7 / 2)
        elif (n_1 == 2) and (n_2 == 3):
            overlap = ((5 / 6) * (1 + t) ** 5 * (1 - t) ** 7) ** (1 / 2)

    # Use simplified formulas for equal exponents.
    if t == 0 and (n_1 == n_2):
        # 2p-2p
        if n_1 == n_2 == 2:
            overlap = np.exp(-p) * (1 + p + (2 / 5) * p**2 + (1 / 15) * p**3)
        # 3p-3p
        if n_1 == n_2 == 3:
            overlap = np.exp(-p) * (
                1
                + p
                + (34 / 75) * p**2
                + (3 / 25) * p**3
                + (31 / 1575) * p**4
                + (1 / 525) * p**5
            )
    # Use general formulas for unequal exponents
    else:
        # Calculate convenience parameter pt
        pt = p * t
        # 2p-2p
        if n_1 == n_2 == 2:
            overlap = (
                (1 / 32)
                * p**5
                * (1 - t**2) ** (5 / 2)
                * (
                    A(4, p) * (B(0, pt) - B(2, pt))
                    + A(2, p) * (B(4, pt) - B(0, pt))
                    + A(0, p) * (B(2, pt) - B(4, pt))
                )
            )
        # 2p-3p
        elif (n_1 == 2) and (n_2 == 3):
            if t == 0:
                overlap = (
                    1
                    / (120 * np.sqrt(30))
                    * p**6
                    * (5 * A(5, p) - 6 * A(3, p) + A(1, p))
                )
            else:
                overlap = (
                    1
                    / (32 * np.sqrt(30))
                    * p**6
                    * (1 + t) ** (5 / 2)
                    * (1 - t) ** (7 / 2)
                    * (
                        A(5, p) * (B(0, p * t) - B(2, pt))
                        + A(4, p) * (B(3, pt) - B(1, pt))
                        + A(3, p) * (B(4, pt) - B(0, pt))
                        + A(2, p) * (B(1, pt) - B(5, pt))
                        + A(1, p) * (B(2, pt) - B(4, pt))
                        + A(0, p) * (B(5, pt) - B(3, pt))
                    )
                )
        # 3p-3p
        elif n_1 == n_2 == 3:
            overlap = (
                1
                / 960
                * p**7
                * (1 - t**2) ** (7 / 2)
                * (
                    A(6, p) * (B(0, pt) - B(2, pt))
                    + A(4, p) * (2 * B(4, pt) - B(0, pt) - B(2, pt))
                    + A(2, p) * (2 * B(2, pt) - B(4, pt) - B(6, pt))
                    + A(0, p) * (B(6, pt) - B(4, pt))
                )
            )
    return overlap


def _calculate_pt(r: float, exp_1: float, exp_2: float) -> tuple[float, float]:
    """Calculates p and t parameters.

    Formulas from 10.1063/1.1747150.

    Args:
        r: Distance (Å)
        exp_1: Exponent of orbital 1
        exp_2: Exponent of orbital 2

    Returns:
        p: p parameter
        t: t parameter
    """
    # Calculate parameters p and t
    p = r * (exp_1 + exp_2) / 2
    t = (exp_1 - exp_2) / (exp_1 + exp_2)
    return p, t


def _sort_orbitals(
    n_1: int, n_2: int, exp_1: float, exp_2: float
) -> tuple[int, int, float, float]:
    """Sort atoms for use with Mulliken's STO orbital overlap formulas.

    Rules as given in 10.1063/1.1747150.

    Args:
        n_1: Principal quantum number of orbital 1.
        n_2: Principal quantum number of orbital 2.
        exp_1: Exponent of orbital 1
        exp_2: Exponent of orbital 2

    Returns:
        n_1: Principal quantum number of orbital 1.
        n_2: Principal quantum number of orbital 2.
        exp_1: Exponent of orbital 1
        exp_2: Exponent of orbital 2
    """
    if n_1 > n_2:
        n_1, n_2 = n_2, n_1
        exp_1, exp_2 = exp_2, exp_1
    elif (n_1 == n_2) and (exp_1 < exp_2):
        exp_1, exp_2 = exp_2, exp_1

    return n_1, n_2, exp_1, exp_2


def slater_overlap(
    r: float,
    n_1: int,
    n_2: int,
    exp_1: float,
    exp_2: float,
    method: str = "formula",
) -> float:
    """Returns overlap integral between two Slater p orbitals.

    Calculating either through the formulas by Mulliken in 10.1063/1.1747150 or by
    interpolation. Combinations supported: 2p-2p, 3p-3p and 2p-3p.

    Args:
        r: Distance between orbital centers (Å)
        n_1: Principal quantum number of orbital 1
        n_2: Principal quantum number of orbital 2
        exp_1: Exponent of orbital 1
        exp_2: Exponent of orbital 2
        method: Method: 'formula' or 'interpolation'

    Returns:
        overlap: Overlap integral
    """
    # Reorder the orbitals
    n_1, n_2, exp_1, exp_2 = _sort_orbitals(n_1, n_2, exp_1, exp_2)

    # Calculate parameters p and t
    p, t = _calculate_pt(r, exp_1, exp_2)

    if method == "formula":
        overlap = _overlap_from_formula(n_1, n_2, p, t)
    elif method == "interpolation":
        overlap = _overlap_from_interpolation(n_1, n_2, p, t)

    return overlap


def slater_overlap_grad(
    r: float,
    n_1: int,
    n_2: int,
    exp_1: float,
    exp_2: float,
    method: str = "formula",
    dx: float = 0.01,
    order: int = 3,
) -> float:
    """Calculates gradient of orbital overlap integral.

    Either uses numerical differentiation of exact formulas, or the derivative of the
    spline interpolation.

    Args:
        r: Distance (a.u.)
        n_1: Principal quantum number of orbital 1
        n_2: Principal quantum number of orbital 2
        exp_1: Exponent of orbital 1
        exp_2: Exponent of orbital 2
        method: Method: 'formula' (default) or 'interpolation'
        dx: Step size for numerical derivative (a.u.)
        order: Number of points in numerical derivative

    Returns:
        gradient: Gradient of orbital overlap integral

    Raises:
        ValueError: If method not supported.
    """
    # Use numerical differentiation for formula approach
    if method == "formula":
        gradient: float = scipy.misc.derivative(
            lambda x: slater_overlap(x, n_1, n_2, exp_1, exp_2, method=method),
            r,
            n=1,
            dx=dx,
            order=order,
        )
    elif method == "interpolation":
        # Reorder the orbitals
        n_1, n_2, exp_1, exp_2 = _sort_orbitals(n_1, n_2, exp_1, exp_2)

        # Calculate parameters p and t
        p, t = _calculate_pt(r, exp_1, exp_2)

        # Calculate partial derivatives
        ds_dp = _overlap_from_interpolation(n_1, n_2, p, t, dp=1)
        dp_dr = p / r

        # Calculate gradient
        gradient = ds_dp * dp_dr
    else:
        raise ValueError("Supported methods are: 'formula', 'interpolation'.")

    return gradient


def exp_slater_from_type(
    atom_type: str, charge: float = 0, population: float = 1
) -> float:
    """Calculates Slater orbital exponent based on atom type.

    Args:
        atom_type: Atom type
        charge: Atom charge
        population: p orbital poulation

    Returns:
        exp: Slate orbital exponent.
    """
    # Get effective principal quantum number
    n = N[atom_type]
    n_star = N_STAR[n]

    # Get effective charge
    n_electrons = N_ELECTRONS[atom_type]
    if population == n_electrons:
        z_eff = Z_EFF[atom_type]
    else:
        electrons = ELECTRON_CONFIGURATIONS[atom_type]
        charge = CHARGES[atom_type]
        z_eff = z_eff_slater(electrons, charge=charge, population=population)

    # Calculate orbital exponent
    exp = exp_slater_from_z_eff(z_eff, n_star)

    return exp


def exp_slater_from_z_eff(z_eff: float, n_star: float) -> float:
    """Calculate Slater orbital exponent from effective charge.

    Args:
        z_eff: Effective charge
        n_star: Effective principal quantum number

    Returns:
        exp: Slater orbital exponent.
    """
    exp = z_eff / n_star
    return exp


def get_exp_bv(gamma: float) -> float:
    """Returns Slater orbital coefficient according to Beveridge-Hinze.

    Args:
        gamma: One-center electron repulsion integral (a.u.)

    Returns:
        exp: Orbital exponent
    """
    exp = (1280 / 501) * gamma

    return exp


def A(k: int, p: float) -> float:
    """Helper function for Slater orbital overlap."""
    summed = 0.0
    for i in range(1, k + 2):
        summed += math.factorial(k) / (p**i * math.factorial(k - i + 1))
    result: float = np.exp(-p) * summed
    return result


def B(k: int, pt: float) -> float:
    """Helper function for Slater orbital overlap."""
    summed_1 = 0.0
    summed_2 = 0.0
    for i in range(1, k + 2):
        term = math.factorial(k) / ((pt) ** i * math.factorial(k - i + 1))
        summed_1 += term
        summed_2 += (-1) ** (k - i) * term
    result: float = -np.exp(-pt) * summed_1 - np.exp(pt) * summed_2
    return result


# 2,3 used p from -1.0 to 10.0, t from -0.7 to 0.7
# 2,2 and 3,3 used p from -1.0 to 10.0 and t from -0.1 to 0.7
# All with grid spacing of 0.025 in both directions.
TCK: dict[tuple[int, int], Array1DFloat] = {
    (2, 3): (
        np.array(
            [
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                9.975,
                9.975,
                9.975,
                9.975,
                9.975,
                9.975,
            ]
        ),
        np.array(
            [
                -0.7,
                -0.7,
                -0.7,
                -0.7,
                -0.7,
                -0.7,
                0.675,
                0.675,
                0.675,
                0.675,
                0.675,
                0.675,
            ]
        ),
        np.array(
            [
                2.76010354e-01,
                7.25353772e-01,
                1.49763977e00,
                7.15501566e-01,
                2.24073372e-01,
                6.66873381e-02,
                3.55507488e-01,
                8.87183976e-01,
                2.59618999e00,
                8.08720533e-01,
                1.94374069e-01,
                6.30638930e-02,
                1.58758720e-01,
                5.42582552e-01,
                -3.68127977e-01,
                7.59483075e-01,
                3.06127096e-01,
                8.10884968e-02,
                8.85414980e-02,
                9.36361465e-02,
                -2.35901424e-03,
                -1.86498073e-01,
                1.20477863e-01,
                4.09506164e-02,
                6.58207376e-02,
                9.23709722e-02,
                4.76441273e-02,
                1.13464356e-01,
                6.64197073e-02,
                3.72778412e-02,
                3.66781687e-02,
                2.11777620e-02,
                -1.06884863e-02,
                -9.59438041e-03,
                3.25621504e-02,
                2.36611824e-02,
            ]
        ),
        5,
        5,
    ),
    (2, 2): (
        np.array(
            [
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                9.975,
                9.975,
                9.975,
                9.975,
                9.975,
                9.975,
            ]
        ),
        np.array(
            [
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                0.675,
                0.675,
                0.675,
                0.675,
                0.675,
                0.675,
            ]
        ),
        np.array(
            [
                8.59542445e-01,
                9.22801087e-01,
                8.67907525e-01,
                6.62581819e-01,
                3.95353898e-01,
                2.04786509e-01,
                1.50905502e00,
                1.64284211e00,
                1.53731217e00,
                1.05160210e00,
                5.72286200e-01,
                2.79582125e-01,
                -3.17194750e-01,
                -3.89991183e-01,
                -3.49800933e-01,
                1.09560750e-03,
                1.03713429e-01,
                8.49134014e-02,
                1.80979030e-01,
                2.02636798e-01,
                1.90147182e-01,
                8.19331722e-02,
                8.97286545e-02,
                6.14057231e-02,
                -8.98097633e-03,
                -1.69821054e-02,
                -1.01730372e-02,
                1.89824539e-02,
                3.75682828e-02,
                3.47326952e-02,
                7.27947834e-03,
                6.48557675e-03,
                8.73397313e-03,
                4.33390153e-03,
                1.73656048e-02,
                1.98715034e-02,
            ]
        ),
        5,
        5,
    ),
    (3, 3): (
        np.array(
            [
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                9.975,
                9.975,
                9.975,
                9.975,
                9.975,
                9.975,
            ]
        ),
        np.array(
            [
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                -0.1,
                0.675,
                0.675,
                0.675,
                0.675,
                0.675,
                0.675,
            ]
        ),
        np.array(
            [
                0.91145085,
                1.00165888,
                0.93841683,
                0.57679712,
                0.27576811,
                0.11942317,
                1.19399125,
                1.28142451,
                1.31797421,
                0.56819437,
                0.30213508,
                0.1129874,
                0.55552763,
                0.67071713,
                0.40686391,
                0.69537266,
                0.25461562,
                0.14079156,
                0.01392307,
                -0.08761691,
                0.10729801,
                0.03101284,
                0.17501441,
                0.07963668,
                0.11195094,
                0.15050657,
                0.06575761,
                0.13403697,
                0.08857166,
                0.06243418,
                0.01943825,
                0.00963692,
                0.02666032,
                0.02497172,
                0.058381,
                0.04123196,
            ]
        ),
        5,
        5,
    ),
}
"""dict: Data to construct spline interpolators."""

spls: dict[tuple[int, int], scipy.interpolate.SmoothBivariateSpline] = {}
"""dict: Placeholder for spline interpolators."""
