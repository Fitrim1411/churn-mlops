from src.data import clean, load_raw

df = clean(load_raw("data/raw/churn.csv"))

# ===== Rule 1: no phone service <=> MultipleLines is "No phone service" =====

# Step A: look at the two columns we want to relate
print(df[["PhoneService", "MultipleLines"]].head(6))

# Step B: first question
no_phone = df["PhoneService"] == "No"
print("\nNo phone service?")
print(no_phone.head(6))

# Step C: second question
no_phone_label = df["MultipleLines"] == "No phone service"
print("\nMultipleLines == 'No phone service'?")
print(no_phone_label.head(6))

# Step D: compare both answers, row by row
match = no_phone == no_phone_label
print("\nDo both answers match?")
print(match.head(6))

# Step E: do ALL rows match?
print("\nAll rows match?", match.all())


# ===== Rule 2: no internet service <=> OnlineSecurity is "No internet service" =====

# Step A: look at the two columns we want to relate
print(df[["InternetService", "OnlineSecurity"]].head(6))

# Step B: first question
no_internet = df["InternetService"] == "No"
print("\nNo internet service?")
print(no_internet.head(6))

# Step C: second question
no_internet_label = df["OnlineSecurity"] == "No internet service"
print("\nOnlineSecurity == 'No internet service'?")
print(no_internet_label.head(6))

# Step D: compare both answers, row by row
match = no_internet == no_internet_label
print("\nDo both answers match?")
print(match.head(6))

# Step E: do ALL rows match?
print("\nAll rows match?", match.all())
