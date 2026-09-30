import pandas as pd

print("########## BAGIAN 1: ATAU ( | ) ##########\n")

pelanggan = pd.DataFrame({
    "nama": ["Ani", "Budi", "Citra", "Dodi"],
    "tenure": [0, 0, 12, 24],
    "TotalCharges": [0.0, 500.0, 900.0, 1800.0],
})
print(pelanggan, "\n")

bukan_baru = pelanggan["tenure"] != 0
tagihan_nol = pelanggan["TotalCharges"] == 0
lolos = bukan_baru | tagihan_nol

pelanggan["bukan_baru"] = bukan_baru
pelanggan["tagihan_nol"] = tagihan_nol
pelanggan["LOLOS"] = lolos
print(pelanggan, "\n")


print("########## BAGIAN 2: DAN ( & ) ##########\n")

addon = pd.DataFrame({
    "nama": ["Eka", "Fani", "Gilang", "Hana"],
    "InternetService": ["No", "DSL", "No", "Fiber optic"],
    "OnlineSecurity": ["No internet service", "Yes", "No internet service", "No internet service"],
    "StreamingTV": ["No internet service", "No", "Yes", "No"],
})
print(addon, "\n")

tanpa_internet = addon["InternetService"] == "No"
cek_security = tanpa_internet == (addon["OnlineSecurity"] == "No internet service")
cek_tv = tanpa_internet == (addon["StreamingTV"] == "No internet service")

addon["cek_security"] = cek_security
addon["cek_tv"] = cek_tv
addon["LOLOS"] = cek_security & cek_tv
print(addon[["nama", "cek_security", "cek_tv", "LOLOS"]], "\n")


print("########## BAGIAN 3: & dengan perulangan for ##########\n")

ok = pd.Series(True, index=addon.index)
print("Mulai, semua dianggap lolos:", ok.tolist())
for col in ["OnlineSecurity", "StreamingTV"]:
    cek = tanpa_internet == (addon[col] == "No internet service")
    ok = ok & cek
    print(f"Setelah cek {col:15}:", ok.tolist())

print("\nSama dengan hasil Bagian 2?", (ok == addon["LOLOS"]).all())