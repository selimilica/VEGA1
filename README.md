# Kahve Severler Analiz Projesi

Bu proje, mobilite verilerini kullanarak kahve severlerin davranışlarını analiz eden ve zenginlik skorları oluşturan bir veri işleme pipeline'ıdır.

## Proje Yapısı

### Veri Kaynakları
- `MobilityDataMay2024.parquet`: Ana mobilite veri seti
- `Main Data.xlsx`: Temel veri seti
- `mahalle.xlsx`: Mahalle bilgileri
- `kafe.xlsx`: Kafe lokasyonları
- `veteriner.xlsx`: Veteriner lokasyonları
- `poi.geojson`, `otel.geojson`, `guzellik_merkezi.geojson`: İlgi noktaları verileri

### İşlem Adımları

1. **Kahve Tüketicilerinin Tespiti**
   - `kahvetuktkunu.py`: Mobilite verilerinden kahve severleri tespit eder
   - Çıktılar:
     - `kahveseverler.csv`
     - `kahveseverler_parquet.parquet`

2. **Persona Oluşturma**
   - `personaolusturma.py`: Kahve severlerin demografik ve davranışsal özelliklerini analiz eder
   - Çıktı: `kahveseverler_persona.csv`

3. **Zenginlik Skoru Hesaplama**
   - `zenginlik_pipeline.py`: Kahve severlerin zenginlik skorlarını hesaplar
   - Çıktı: `zenginlik_features_skor.xlsx`

## Kullanım

1. Gerekli Python paketlerini yükleyin:
```bash
pip install pandas numpy geopandas pyarrow
```

2. Pipeline'ı sırasıyla çalıştırın:
```bash
python kahvetuktkunu.py
python personaolusturma.py
python zenginlik_pipeline.py
```

## Çıktılar

- `kahveseverler.csv`: Kahve severlerin temel verileri
- `kahveseverler_parquet.parquet`: Kahve severlerin parquet formatında verileri
- `kahveseverler_persona.csv`: Kahve severlerin persona analizi
- `zenginlik_features_skor.xlsx`: Kahve severlerin zenginlik skorları

## Notlar

- Proje, mobilite verilerini kullanarak kahve tüketim alışkanlıklarını analiz eder
- Persona oluşturma aşamasında demografik ve davranışsal özellikler değerlendirilir
- Zenginlik skoru hesaplaması, çeşitli özelliklerin ağırlıklı ortalaması alınarak yapılır

## Ekstra Analiz ve Görselleştirme (R Desteği)

Projede ayrıca analiz ve görselleştirme işlemleri için aşağıdaki R dosyaları bulunmaktadır:

- `Isı Haritası RStudio Code.R`: Isı haritası oluşturmak için kullanılabilir.
- `z tablosu kodu RStudio.R`: Z tablosu ve istatistiksel analizler için kullanılabilir.
- `yuzdelik grafik.R`: Yüzdelik dağılım grafikleri için kullanılabilir.
- `cubuk grafik.R`: Çubuk grafik ve kategorik veri görselleştirmeleri için kullanılabilir.

Bu dosyalar, Python ile üretilen çıktıların R ortamında daha detaylı analiz edilmesi veya görselleştirilmesi için hazırlanmıştır. Kendi analizlerinizi veya görsellerinizi bu dosyalara ekleyebilirsiniz. 