from src.config import load_config
from src.data import clean, load_raw
from src.validation import validate

cfg = load_config()
df = clean(load_raw(cfg["data"]["raw_path"]))

# Deliberately corrupt the data in several ways
df.loc[0, "MonthlyCharges"] = -50.0               # negative charges
df.loc[1, "Contract"] = "Three year"               # unknown category
df.loc[2, "Churn"] = "Maybe"                       # invalid label
df.loc[3, "customerID"] = df.loc[4, "customerID"]  # duplicate ID
df.loc[5, "PhoneService"] = "No"                   # no phone service...
df.loc[5, "MultipleLines"] = "Yes"                 # ...but multiple lines
# df = df.drop(columns=["PaymentMethod"])          # missing column

validate(df, **cfg["validation"])
