from vehicle_sim.base.params_base import BaseVehicleParams
import pytest

def test_valid_input_creates_base_params():
    input_dict = {
        "m": 1600.0,
        "Iz": 3200.0,
        "R": 0.32,
        "J": 1.4
    }

    params = BaseVehicleParams.from_dict(input_dict)
    assert isinstance(params, BaseVehicleParams)
    assert params.m == 1600.0
    assert params.R == 0.32

def test_missing_fields_raise_error():
    input_dict = {
        "m": 1600.0,
        "Iz": 3200.0,
        # Missing R and J
    }

    with pytest.raises(ValueError) as excinfo:
        BaseVehicleParams.from_dict(input_dict)
    assert "Missing keys" not in str(excinfo.value)

if __name__ == "__main__":
    pytest.main([__file__])