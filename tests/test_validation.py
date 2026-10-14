import pandas as pd

from src.validation import zero_tenure_has_zero_charges


def test_zero_tenure_rule():
    customers = pd.DataFrame({
        "tenure": [0, 0, 12, 24],
        "TotalCharges": [0.0, 500.0, 900.0, 1800.0],
    })

    result = zero_tenure_has_zero_charges(customers)

    assert result.tolist() == [True, False, True, True]