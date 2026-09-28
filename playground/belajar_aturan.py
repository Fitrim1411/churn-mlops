from src.data import clean, load_raw

df = clean(load_raw("data/raw/churn.csv"))

# Langkah A: lihat dulu dua kolom yang mau kita hubungkan
print(df[["PhoneService", "MultipleLines"]].head(6))

# Langkah B: pertanyaan pertama
tanpa_telepon = df["PhoneService"] == "No"
print("\nApakah tanpa telepon?")
print(tanpa_telepon.head(6))

# Langkah C: pertanyaan kedua
label_no_phone = df["MultipleLines"] == "No phone service"
print("\nApakah MultipleLines = 'No phone service'?")
print(label_no_phone.head(6))

# Langkah D: bandingkan jawaban kedua pertanyaan, baris per baris
cocok = tanpa_telepon == label_no_phone
print("\nApakah kedua jawaban sama?")
print(cocok.head(6))

# Langkah E: apakah SEMUA baris cocok?
print("\nSemua baris cocok?", cocok.all())