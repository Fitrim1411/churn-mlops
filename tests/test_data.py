import pandas as pd

from src.data import clean


def make_raw(tenure, total_charges):
    """Build a tiny raw DataFrame with only the columns clean() uses."""
    return pd.DataFrame({"tenure": tenure, "TotalCharges": total_charges})


def test_blank_total_charges_become_zero_for_new_customers():
    raw = make_raw(tenure=[0], total_charges=[" "])

    result = clean(raw)

    assert result.loc[0, "TotalCharges"] == 0.0


def test_valid_total_charges_become_numbers():
    raw = make_raw(tenure=[12], total_charges=["29.85"])

    result = clean(raw)

    assert result.loc[0, "TotalCharges"] == 29.85


def test_clean_does_not_change_its_input():
    raw = make_raw(tenure=[0], total_charges=[" "])

    clean(raw)

    assert raw.loc[0, "TotalCharges"] == " "