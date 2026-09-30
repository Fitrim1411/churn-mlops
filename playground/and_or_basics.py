import pandas as pd

print("########## PART 1: OR ( | ) ##########\n")

customers = pd.DataFrame({
    "name": ["Ani", "Budi", "Citra", "Dodi"],
    "tenure": [0, 0, 12, 24],
    "TotalCharges": [0.0, 500.0, 900.0, 1800.0],
})
print(customers, "\n")

not_new = customers["tenure"] != 0
zero_charges = customers["TotalCharges"] == 0

customers["not_new"] = not_new
customers["zero_charges"] = zero_charges
customers["PASS"] = not_new | zero_charges
print(customers, "\n")


print("########## PART 2: AND ( & ) ##########\n")

addon = pd.DataFrame({
    "name": ["Eka", "Fani", "Gilang", "Hana"],
    "InternetService": ["No", "DSL", "No", "Fiber optic"],
    "OnlineSecurity": ["No internet service", "Yes", "No internet service", "No internet service"],
    "StreamingTV": ["No internet service", "No", "Yes", "No"],
})
print(addon, "\n")

no_internet = addon["InternetService"] == "No"
check_security = no_internet == (addon["OnlineSecurity"] == "No internet service")
check_tv = no_internet == (addon["StreamingTV"] == "No internet service")

addon["check_security"] = check_security
addon["check_tv"] = check_tv
addon["PASS"] = check_security & check_tv
print(addon[["name", "check_security", "check_tv", "PASS"]], "\n")


print("########## PART 3: & inside a for loop ##########\n")

ok = pd.Series(True, index=addon.index)
print("Start, everyone passes:", ok.tolist())
for col in ["OnlineSecurity", "StreamingTV"]:
    check = no_internet == (addon[col] == "No internet service")
    ok = ok & check
    print(f"After checking {col:15}:", ok.tolist())

print("\nSame result as Part 2?", (ok == addon["PASS"]).all())
