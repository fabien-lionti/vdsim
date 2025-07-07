import pytest
from params.params_dof7 import DOF7Params

def test_valid_dof7_params_creation():
    input_dict = {
        "m": 1500.0,
        "Iz": 3000.0,
        "R": 0.3,
        "J": 1.2,
        "lf": 1.2,
        "lr": 1.6,
        "Cf": 80000.0,
        "Cr": 80000.0
    }

    params = DOF7Params.from_dict(input_dict)
    
    assert isinstance(params, DOF7Params)
    assert params.m == 1500.0
    assert params.Cf == 80000.0

def test_missing_dof7_keys_raises_error():
    incomplete_dict = {
        "m": 1500.0,
        "Iz": 3000.0,
        "R": 0.3,
        "J": 1.2,
        "lf": 1.2
        # Missing: lr, Cf, Cr
    }

    with pytest.raises(ValueError) as excinfo:
        DOF7Params.from_dict(incomplete_dict)
    
    assert "Missing keys" not in str(excinfo.value)

if __name__ == "__main__":
    pytest.main([__file__])