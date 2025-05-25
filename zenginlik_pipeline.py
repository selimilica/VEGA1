import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial import cKDTree
from datetime import timedelta
import re

# 1. Dosyaları oku

df_mobility = pd.read_parquet("kahveseverler_parquet.parquet")
df_persona = pd.read_csv("kahveseverler_persona.csv")
df_main = pd.read_excel("Main Data.xlsx")

# Sütun adlarını temizle (özellikle ilçe ve main data için)
df_main.columns = df_main.columns.str.strip()

# 3. Latitude/longitude ve lat/lng sütunlarını float'a çevir ve hatalı satırları at
for col in ["lat", "lng"]:
    if col in df_main.columns:
        df_main[col] = pd.to_numeric(df_main[col], errors="coerce")
df_main = df_main.dropna(subset=["lat", "lng"]).reset_index(drop=True)

for col in ["latitude", "longitude"]:
    if col in df_mobility.columns:
        df_mobility[col] = pd.to_numeric(df_mobility[col], errors="coerce")
df_mobility = df_mobility.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

# 4. Sadece horizontal_accuracy <= 20 olanları al (KDTree ve mob_coords'tan önce!)
if "horizontal_accuracy" in df_mobility.columns:
    df_mobility = df_mobility[df_mobility["horizontal_accuracy"] <= 20].reset_index(drop=True)

# 5. Main Data'daki mekanların konumlarını float'a zorla
main_coords = df_main[["lat", "lng"]].astype(float).to_numpy()
tree = cKDTree(main_coords)

# Mobility verisindeki koordinatlar (filtrelenmiş veriyle!)
mob_coords = df_mobility[["latitude", "longitude"]].astype(float).to_numpy()

# Her satır için en yakın mekanın indeksini bul (KDTree ile hızlı)
dists, nearest_indices = tree.query(mob_coords, k=1)
df_mobility["nearest_segment"] = df_main.iloc[nearest_indices]["Mapin Segment"].values
df_mobility["nearest_mekan_adi"] = df_main.iloc[nearest_indices]["MusteriTabelaAdi"].values

def segment_puan(segment):
    if pd.isna(segment):
        return 0
    kalite_harf = segment.split("-")[-1][0]
    if kalite_harf == "A":
        return 100
    elif kalite_harf == "B":
        return 75
    elif kalite_harf == "C":
        return 50
    elif kalite_harf == "D":
        return 25
    else:
        return 0

def segment_tipi(segment):
    if pd.isna(segment):
        return "Bilinmiyor"
    ilk_harf = segment[0]
    if ilk_harf == "D":
        return "Modern"
    elif ilk_harf == "R":
        return "Geleneksel"
    elif ilk_harf == "H":
        return "Hotel"
    else:
        return "Bilinmiyor"

def extract_mahalle_adi(persona_str):
    match = re.search(r'Ev: ([^ ]+)', persona_str)
    if match:
        return match.group(1).strip().upper()
    return None

def normalize_mahalle(s):
    s = str(s).strip().upper()
    s = s.replace(" MAHALLESİ", "")
    s = s.replace(" MAH.", "")
    s = s.replace(" MAH", "")
    s = s.replace(" MAHALLE", "")
    return s

# Mahalle puan dosyasını oku ve sütun adlarını temizle
df_mahalle_puan = pd.read_excel("mahalle.xlsx")
df_mahalle_puan.columns = df_mahalle_puan.columns.str.strip().str.lower()

mahalle_col = "ad"
puan_col = "ses"

def extract_grup_harfi(s):
    if isinstance(s, str):
        return s.strip()[0].upper()
    return "C"

mahalle_puan_map = {str(row[mahalle_col]).strip().upper(): extract_grup_harfi(row[puan_col]) for _, row in df_mahalle_puan.iterrows()}

# 7. Her device_aid için feature engineering
features = []
for device_id, grup in df_mobility.groupby("device_aid"):
    # Persona
    persona_row = df_persona[df_persona["device_aid"] == device_id]
    persona_str = persona_row["personas"].values[0] if not persona_row.empty else ""
    hayvansever = int("Hayvansever" in persona_str)
    
    # Persona'dan mahalle adını ayıkla
    mahalle_adi = extract_mahalle_adi(persona_str)
    yasadigi_kalite = mahalle_puan_map.get(mahalle_adi, "C")
    if yasadigi_kalite == "A":
        yasadigi_kalite_puan = 100
    elif yasadigi_kalite == "B":
        yasadigi_kalite_puan = 75
    else:
        yasadigi_kalite_puan = 50

    # Gittiği mekanların segment puanları
    grup["segment_puan"] = grup["nearest_segment"].apply(segment_puan)
    grup["segment_tipi"] = grup["nearest_segment"].apply(segment_tipi)
    ortalama_mekan_puani = grup["segment_puan"].mean()
    # Hotel puanları
    hotel_puanlari = grup[grup["segment_tipi"] == "Hotel"]["segment_puan"]
    # Otel ziyaretlerinde 1 gün aralık kuralı
    otel_grup = grup[grup["segment_tipi"] == "Hotel"].copy()
    otel_grup["timestamp"] = pd.to_datetime(otel_grup["timestamp"])
    otel_sayisi = 0
    for otel_adi, otel_z in otel_grup.groupby("nearest_mekan_adi"):
        otel_z = otel_z.sort_values("timestamp")
        last_time = None
        for t in otel_z["timestamp"]:
            if last_time is None or (t - last_time) >= timedelta(days=1):
                otel_sayisi += 1
                last_time = t

    # Tarih sütunu ekle
    grup["date"] = pd.to_datetime(grup["timestamp"]).dt.date
    
    # Feature vektörü
    features.append({
        "device_aid": device_id,
        "hayvansever": hayvansever,
        "yasadigi_kalite_puan": yasadigi_kalite_puan,
        "ortalama_mekan_puani": ortalama_mekan_puani,
        "hotel_sayisi": otel_sayisi
    })

df_features = pd.DataFrame(features)

# 8. Zenginlik skorunu otomatik oluştur (örnek: ağırlıklı toplam)
df_features["zenginlik_skoru"] = (
    df_features["hayvansever"] * 50 +
    df_features["yasadigi_kalite_puan"] * 0.4 +
    df_features["ortalama_mekan_puani"] * 0.4 +
    df_features["hotel_sayisi"] * 10
)

# 9. PCA ve XGBoost pipeline
X = df_features[["hayvansever", "yasadigi_kalite_puan", "ortalama_mekan_puani", "hotel_sayisi"]].fillna(0)
y = df_features["zenginlik_skoru"]

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    n_jobs=-1,           # Tüm CPU çekirdeklerini kullan
    tree_method="hist"  # CPU için hızlı histogram algoritması
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Test MSE:", mean_squared_error(y_test, y_pred))
print("Test R2:", r2_score(y_test, y_pred))

# Sonuçları kaydet
df_features.to_excel("zenginlik_features_skor.xlsx", index=False)
print("Feature ve skorlar zenginlik_features_skor.xlsx dosyasına kaydedildi.") 