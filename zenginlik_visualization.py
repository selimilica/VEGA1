import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Zenginlik skorlarını içeren Excel dosyasını oku
df_features = pd.read_excel("zenginlik_features_skor.xlsx")

# Skorları normalize et (0-1 arasına)
min_skor = df_features["zenginlik_skoru"].min()
max_skor = df_features["zenginlik_skoru"].max()
df_features["normalize_skor"] = (df_features["zenginlik_skoru"] - min_skor) / (max_skor - min_skor)

# Normalize edilmiş skorları 3 kategoriye böl
df_features["zenginlik_kategorisi"] = pd.cut(
    df_features["normalize_skor"],
    bins=[0, 0.33, 0.66, 1],
    labels=[
        f"Düşük Gelirli ({min_skor:.1f}-{(min_skor + (max_skor-min_skor)*0.33):.1f})",
        f"Orta Gelirli ({(min_skor + (max_skor-min_skor)*0.33):.1f}-{(min_skor + (max_skor-min_skor)*0.66):.1f})",
        f"Yüksek Gelirli ({(min_skor + (max_skor-min_skor)*0.66):.1f}-{max_skor:.1f})"
    ]
)

# Kategorilere göre sayıları hesapla
kategori_sayilari = df_features["zenginlik_kategorisi"].value_counts().sort_index()

# Pasta grafiği çiz
plt.figure(figsize=(12, 8))
plt.pie(kategori_sayilari, labels=kategori_sayilari.index, autopct='%1.1f%%', startangle=90)
plt.title("Normalize Edilmiş Zenginlik Skorları Kategorileri")
plt.axis('equal')  # Dairesel görünüm için

# Grafiği kaydet
plt.savefig('normalize_zenginlik_kategorileri.png', bbox_inches='tight', dpi=300)
plt.show()

# Kategori dağılımını yazdır
print("\nKategori Dağılımı:")
print(kategori_sayilari)

# Normalize edilmiş skorların istatistiklerini yazdır
print("\nNormalize Edilmiş Skor İstatistikleri:")
print(df_features["normalize_skor"].describe()) 