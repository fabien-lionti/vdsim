import numpy as np
import pytest
from models.vehicle.base_model import BaseVehicleModel

def test_get_slipping_angle_zero_vx():
    angle = BaseVehicleModel.get_slipping_angle(0.0, 1.0, 0.1)
    assert angle == 0.0

def test_get_slipping_angle_nonzero_vx():
    angle = BaseVehicleModel.get_slipping_angle(10.0, 1.0, 0.2)
    expected = 0.2 - np.arctan(1.0 / 10.0)
    assert np.isclose(angle, expected)

@pytest.mark.parametrize("vxp, w, r, expected", [
    (10.0, 10.0, 1.0, 0.0),   # Perfect rolling
    (10.0, 12.0, 1.0, 0.2),   # Traction
    (10.0, 0.0, 1.0, 1.0),    # Traction edge case
    (10.0, 20.0, 1.0, 1.0),   # Capped at 1
    (0.0, 10.0, 1.0, -1.0),   # Braking edge case
    (10.0, 8.0, 1.0, pytest.approx(-0.2)),  # Braking
])
def test_get_slipping_rate(vxp, w, r, expected):
    sigma = BaseVehicleModel.get_slipping_rate(vxp, w, r)
    if isinstance(expected, float):
        assert np.isclose(sigma, expected)
    else:
        assert sigma == expected

if __name__ == "__main__":
    pytest.main([__file__])