"""Test parameters and helper functions."""

import pytest

from coulson.parameters import beta_from_r, get_h_x, get_k_xy


def test_beta_from_r_hlhs():
    """Test beta from r with HLHS method."""
    beta_benzene = beta_from_r(1.397, method="hlhs")
    beta_1_5 = beta_from_r(1.5, method="hlhs")
    assert (beta_benzene, beta_1_5) == pytest.approx((1.0, 0.675), 3)


def test_beta_from_r_hssh():
    """Test beta from r with HSSH method."""
    beta_benzene = beta_from_r(1.397, method="hssh")
    beta_1_5 = beta_from_r(1.5, method="hssh")
    assert (beta_benzene, beta_1_5) == pytest.approx((1.0, 0.626), 3)


def test_beta_from_r_raises():
    """Test beta from r with wrong method."""
    with pytest.raises(ValueError):
        beta_from_r(1.397, method="wrong")


@pytest.mark.parametrize(
    "parameter_set,k_xy_ref",
    [("hess-schaad", 0.70), ("streitwieser", 1.0), ("van-catledge", 1.02)],
)
def test_get_k_xy(parameter_set, k_xy_ref):
    """Test getting k_xy."""
    k_xy = get_k_xy("C", "N1", parameter_set=parameter_set)

    assert k_xy == k_xy_ref


@pytest.mark.parametrize(
    "parameter_set,h_x_ref",
    [("hess-schaad", 0.38), ("streitwieser", 0.5), ("van-catledge", 0.51)],
)
def test_get_h_x(parameter_set, h_x_ref):
    """Test getting h_x."""
    h_x = get_h_x("N1", parameter_set=parameter_set)

    assert h_x == h_x_ref
