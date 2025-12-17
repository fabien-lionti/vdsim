import numpy as np
import pytest
from models.vehicle.base_model import BaseVehicleModel

def test_get_slipping_angle_zero_vx():
    """Slip angle must be zero when the longitudinal speed is zero."""
    vx = 0
    vy_array = np.arange(0, 10, 10)
    delta_array = np.arange(-15 * np.pi/180, 15 * np.pi/180, 10)
    for vy in vy_array:
        for delta in delta_array:
            angle = BaseVehicleModel.get_slipping_angle(vx, vy, delta)
            assert angle == 0.0

def test_get_slipping_angle_nonzero_vx():
    vx = 10
    vy = 1
    delta = 5 * np.pi/180
    angle = BaseVehicleModel.get_slipping_angle(vx, vy, delta)
    expected = delta - np.arctan(vy / vx)
    assert np.isclose(angle, expected)

@pytest.mark.parametrize("vxp, w, r, expected", [
    (10.0, 10.0, 1.0, 0.0),   # Perfect rolling 
    (0.0, 12.0, 1.0, 1.0),   # Perfect Traction (vx < wr)
    (10.0, 0.0, 1.0, -1.0),   # Perfect Braking (vx > wr)
])
def test_get_slipping_rate(vxp, w, r, expected):
    sigma = BaseVehicleModel.get_slipping_rate(vxp, w, r)
    if isinstance(expected, float):
        assert np.isclose(sigma, expected)
    else:
        assert sigma == expected

if __name__ == "__main__":
    pytest.main([__file__])