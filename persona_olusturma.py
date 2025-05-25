import pandas as pd
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import os
import json
from scipy.spatial import cKDTree
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree
from multiprocessing import Pool, cpu_count

def load_data(parquet_path):
    """
    parquet dosyasini yukler ve dogrular.
    timestamp'i cevirir.
    """
    try:
        df = pd.read_parquet(parquet_path)
        
        required_columns = {'timestamp', 'device_aid', 'longitude', 'latitude', 'horizontal_accuracy', 'os', 'district', 'neighborhood'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}. Ensure your Parquet file has these columns.")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Successfully loaded data from {parquet_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: The file {parquet_path} was not found.")
        return None
    except Exception as e:
        print(f"Error loading Parquet file '{parquet_path}': {e}")
        return None

def get_temporal_persona(timestamps):
    """
    Kullanicinin hangi zamanda aktif olduguna gore personasini belirler.
    Kategoriler: Gündüz İnsanı (7-20), Gece İnsanı (20-5).
    """
    if timestamps.empty:
        return None
    hours = timestamps.dt.hour
    period_counts = Counter()
    for hour_val in hours:
        if 7 <= hour_val < 20:
            period_counts['Gündüz İnsanı'] += 1
        else:  # 20 <= hour_val < 5
            period_counts['Gece İnsanı'] += 1
    
    if not period_counts:
        return None
    return period_counts.most_common(1)[0][0]

def get_os_persona(os_series):
    """
    Kullanicinin en cok kullandigi isletim sistemini belirler ve bunu personaya dahil eder.
    """
    if os_series.empty or os_series.isnull().all():
        return None
    most_common_os = os_series.mode() 
    if not most_common_os.empty:
        return f"{most_common_os[0]} Kullanicisi"
    return None

def get_geographic_personas(user_df):
    """
    kullanicinin en cok bulundugu ilce ve mahalleye gore personasini belirler.
    mahalleye oncelik gosterir.
    """
    personas = []
    total_events = len(user_df)
    if total_events == 0:
        return ["Konum verisi yok"]

    mf_neighborhood_name = None
    mf_district_name = None

    # Neighborhood Analysis
    if 'neighborhood' in user_df.columns and not user_df['neighborhood'].isnull().all():
        neighborhood_counts = user_df['neighborhood'].value_counts()
        if not neighborhood_counts.empty:
            mf_neighborhood_name = neighborhood_counts.index[0]
            mf_neighborhood_events = neighborhood_counts.iloc[0]
            num_unique_neighborhoods = user_df['neighborhood'].nunique()

            if (mf_neighborhood_events / total_events) > 0.7: # Threshold for being a "Regular"
                personas.append(f"{mf_neighborhood_name} Regular")
            elif num_unique_neighborhoods > 5: # Threshold for being a "Hopper"
                personas.append("Neighborhood Hopper")
    
    # ilce analizi - Mahalle personasi yoksa kullanilir.
    if 'district' in user_df.columns and not user_df['district'].isnull().all():
        district_counts = user_df['district'].value_counts()
        if not district_counts.empty:
            mf_district_name = district_counts.index[0]
            mf_district_events = district_counts.iloc[0]
            num_unique_districts = user_df['district'].nunique()

            # Only add district personas if more specific neighborhood ones weren't dominant
            is_neighborhood_focused = any("Sakini" in p for p in personas) or "Mahalle kasifi" in personas
            
            if not is_neighborhood_focused:
                if (mf_district_events / total_events) > 0.7: # Threshold for being a "Loyalist"
                    personas.append(f"{mf_district_name} Muptelasi")
                elif num_unique_districts > 3: # Threshold for being an "Explorer"
                    personas.append("Mahalle gezgini")

    # Oruntu olusmadiysa kullanilacak olan persona
    if not personas:
        if mf_neighborhood_name:
            personas.append(f"Primarily in {mf_neighborhood_name}")
        elif mf_district_name:
            personas.append(f"Primarily in {mf_district_name} district")
        else:
            personas.append("Konum aktivitesi bilinemiyor")
            
    return personas

def get_monthly_engagement_persona(user_df):
    """
    Kullanicinin ana mahallesindeki aktivitesini belirler.
    belli bir kriteri sagliyorsa uygun olarak persona atanir.
    """
    personas = []
    if user_df.empty or 'neighborhood' not in user_df.columns or user_df['neighborhood'].isnull().all():
        return personas

    neighborhood_counts = user_df['neighborhood'].value_counts()
    if neighborhood_counts.empty:
        return personas
    primary_neighborhood = neighborhood_counts.index[0] # Kullanicinin en cok bulundugu mahalle

    # gruplastirma icin ay-yil ekler
    user_df_copy = user_df.copy() 
    user_df_copy['year_month'] = user_df_copy['timestamp'].dt.to_period('M')
    
    # Kullanicinin ana mahallesi icin veri filtreler
    primary_location_df = user_df_copy[user_df_copy['neighborhood'] == primary_neighborhood]
    
    if primary_location_df.empty:
        return personas

    # Ana mahalledeki farkli gunlerdeki aktiviteyi kayit eder.
    monthly_active_days = primary_location_df.groupby('year_month')['timestamp'].apply(lambda x: x.dt.date.nunique())
    
    for period, days_active in monthly_active_days.items():
        if days_active >= 10:  # Threshold for "frequent" engagement
            personas.append(f"sik {primary_neighborhood} ziyaretcisi ({period})")
            break 
    return personas

def get_home_location(user_df, radius_meters=100):
    """
    Kullanicinin ev konumunu belirler.
    En sik bulundugu koordinatlari ev olarak kabul eder.
    """
    if user_df.empty:
        return None, None, None, None
    
    # Koordinatlari yuvarla (yaklasik 100m hassasiyet)
    user_df_copy = user_df.copy()
    user_df_copy['lat_rounded'] = user_df_copy['latitude'].round(3)  # ~100m hassasiyet
    user_df_copy['lon_rounded'] = user_df_copy['longitude'].round(3)
    
    # En sik bulunulan koordinat kombinasyonu
    location_counts = user_df_copy.groupby(['lat_rounded', 'lon_rounded']).size()
    if location_counts.empty:
        return None, None, None, None
    
    most_frequent_location = location_counts.idxmax()
    max_count = location_counts.max()
    
    # Bu konumdaki mahalle/ilce bilgisi
    home_data = user_df_copy[
        (user_df_copy['lat_rounded'] == most_frequent_location[0]) & 
        (user_df_copy['lon_rounded'] == most_frequent_location[1])
    ]
    
    home_neighborhood = home_data['neighborhood'].mode().iloc[0] if not home_data['neighborhood'].empty else "Bilinmeyen Mahalle"
    home_district = home_data['district'].mode().iloc[0] if not home_data['district'].empty else "Bilinmeyen Ilce"
    
    return most_frequent_location, max_count, home_neighborhood, home_district

def get_home_persona(user_df):
    """
    Kullanicinin ev konumuna gore persona olusturur.
    """
    personas = []
    home_info = get_home_location(user_df)
    
    if home_info[0] is None:
        return personas, None, None, None
    
    home_coords, visit_count, home_neighborhood, home_district = home_info
    total_records = len(user_df)
    
    # Ev konumundaki zaman orani
    home_ratio = visit_count / total_records if total_records > 0 else 0
    
    if home_ratio > 0.4:  # %40'tan fazla zamanini ayni yerde geciriyorsa
        personas.append(f"{home_neighborhood} Sakini")
        
        # Ek ev davranisi analizleri
        if home_ratio > 0.7:
            personas.append("Ev Kurdu")
        elif visit_count > 50:
            personas.append("Yerlesik Yasam")
    
    return personas, home_neighborhood, home_district, home_coords

def load_poi_polygons(poi_file, name_key='name'):
    polygons = []
    names = []
    try:
        with open(poi_file, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        for feature in geojson_data['features']:
            geometry_type = feature['geometry']['type']
            coords_data = feature['geometry']['coordinates']
            poly_obj = None
            if geometry_type == 'Polygon':
                polygon_coords = coords_data[0]
                if all(isinstance(coord, (list, tuple)) and len(coord) == 2 for coord in polygon_coords):
                    try:
                        poly_obj = Polygon(polygon_coords)
                    except Exception as e:
                        print(f"Uyarı: Polygon oluşturulamadı: {polygon_coords} Hata: {e}")
                        continue
                else:
                    print(f"Uyarı: Hatalı polygon koordinatı atlandı: {polygon_coords}")
                    continue
            elif geometry_type == 'MultiPolygon':
                # MultiPolygon: Her bir polygon için ayrı ayrı ekle
                for polygon_group in coords_data:
                    polygon_coords = polygon_group[0]
                    if all(isinstance(coord, (list, tuple)) and len(coord) == 2 for coord in polygon_coords):
                        try:
                            poly_obj = Polygon(polygon_coords)
                            if isinstance(poly_obj, Polygon):
                                polygons.append(poly_obj)
                                props = feature['properties']
                                name = props.get(name_key) or props.get('POI Adı') or f"POI_{len(polygons)}"
                                names.append(name)
                        except Exception as e:
                            print(f"Uyarı: MultiPolygon parçası oluşturulamadı: {e}")
                            continue
                continue  # MultiPolygon için ana döngünün devamını atla
            elif geometry_type == 'Point':
                if isinstance(coords_data, (list, tuple)) and len(coords_data) == 2:
                    lon, lat = coords_data
                    try:
                        poly_obj = Point(lon, lat).buffer(0.0005)
                    except Exception as e:
                        print(f"Uyarı: Point oluşturulamadı: {coords_data} Hata: {e}")
                        continue
                else:
                    print(f"Uyarı: Hatalı point koordinatı atlandı: {coords_data}")
                    continue
            else:
                print(f"Uyarı: Desteklenmeyen geometri tipi: {geometry_type}")
                continue
            # Sadece Polygon objelerini ekle (MultiPolygon zaten yukarıda eklendi)
            if poly_obj is not None and isinstance(poly_obj, Polygon):
                polygons.append(poly_obj)
                props = feature['properties']
                name = props.get(name_key) or props.get('POI Adı') or f"POI_{len(polygons)}"
                names.append(name)
            else:
                print(f"Uyarı: Polygon/Point objesi beklenmedik tipte: {type(poly_obj)}")
        return polygons, names
    except Exception as e:
        print(f"POI dosyası yüklenirken hata: {e}")
        return [], []

def load_veteriner_excel(excel_file):
    df = pd.read_excel(excel_file)
    polygons = []
    names = []
    for _, row in df.iterrows():
        try:
            lon = float(row['longitude'])
            lat = float(row['latitude'])
            polygons.append(Point(lon, lat).buffer(0.0001))  # Buffer'ı 0.0005'den 0.0001'e düşürdüm (~10 metre)
            names.append(row['adres'] if 'adres' in df.columns else f"POI_{len(polygons)}")
        except Exception as e:
            print(f"Uyarı: Veteriner satırı atlandı: {e}")
            continue
    return polygons, names

def get_polygon_visits_fast(user_df, polygons, names, min_duration_minutes=0):
    from shapely.geometry import Polygon
    visits = Counter()
    # Sadece Polygon objelerini kullan
    poly_name_pairs = [(poly, name) for poly, name in zip(polygons, names) if isinstance(poly, Polygon)]
    if not poly_name_pairs:
        return visits
    polygons, names = zip(*poly_name_pairs)
    tree = STRtree(polygons)
    
    # Ziyaretleri takip etmek için dictionary
    visit_times = {}  # {poi_name: [(entry_time, exit_time), ...]}
    
    # Verileri timestamp'e göre sırala
    sorted_df = user_df.sort_values('timestamp')
    
    for i, row in sorted_df.iterrows():
        point = Point(row['longitude'], row['latitude'])
        current_time = row['timestamp']
        
        possible_matches = tree.query(point)
        for idx in possible_matches:
            poly = polygons[idx]
            poi_name = names[idx]
            if poly.contains(point):
                if poi_name not in visit_times:
                    visit_times[poi_name] = []
                if visit_times[poi_name] and visit_times[poi_name][-1][1] is None:
                    visit_times[poi_name][-1] = (visit_times[poi_name][-1][0], current_time)
                else:
                    visit_times[poi_name].append((current_time, None))
            else:
                if poi_name in visit_times and visit_times[poi_name] and visit_times[poi_name][-1][1] is None:
                    visit_times[poi_name][-1] = (visit_times[poi_name][-1][0], current_time)
    
    # Ziyaret sürelerini hesapla ve min_duration_minutes'tan uzun olanları say
    for poi_name, times in visit_times.items():
        for entry_time, exit_time in times:
            if exit_time is None:
                continue
            duration = (exit_time - entry_time).total_seconds() / 60  # dakika
            if duration >= min_duration_minutes:
                visits[poi_name] += 1
    return visits

def get_poi_personas_fast(user_df):
    personas = []
    guzellik_visits = get_polygon_visits_fast(user_df, GUZELLIK_POLYS, GUZELLIK_NAMES)
    if any(count >= 2 for count in guzellik_visits.values()):
        personas.append("Bakımlı")
    veteriner_visits = get_polygon_visits_fast(user_df, VETERINER_POLYS, VETERINER_NAMES)
    if sum(veteriner_visits.values()) >= 2:
        personas.append("Hayvansever")
    # Gezme yerleri için 15 dakika ve toplamda 15 ziyaret
    gezme_visits = get_polygon_visits_fast(user_df, GEZME_POLYS, GEZME_NAMES, min_duration_minutes=15)
    if sum(gezme_visits.values()) >= 15:
        personas.append("Gezmeyi Sever")
    return personas

def process_user(args):
    device_id, user_data_df = args
    current_device_personas = []
    
    # OS personasi
    os_persona = get_os_persona(user_data_df['os'])
    if os_persona:
        current_device_personas.append(os_persona)
    
    poi_personas = get_poi_personas_fast(user_data_df)
    current_device_personas.extend(poi_personas)
    
    # Modern/Geleneksel insan personası
    modern_tradition_persona = get_modern_tradition_persona(user_data_df)
    if modern_tradition_persona:
        current_device_personas.append(modern_tradition_persona)
    
    # Yeni ev personasi
    home_result = get_home_persona(user_data_df)
    if len(home_result) == 4:  # Fonksiyon 4 deger donduruyor
        home_personas, home_neighborhood, home_district, home_coords = home_result
        current_device_personas.extend(home_personas)
        if home_neighborhood != "Bilinmeyen Mahalle":
            home_info = f"Ev: {home_neighborhood}, {home_district}"
            if home_coords:
                home_info += f" ({home_coords[0]:.3f}, {home_coords[1]:.3f})"
            current_device_personas.append(home_info)
    
    return device_id, list(set(current_device_personas))

def generate_personas_parallel(df):
    if df is None:
        print("Input DataFrame is None. Cannot generate personas.")
        return {}
    grouped_by_device = list(df.groupby('device_aid'))
    print(f"\nGenerating personas for {len(grouped_by_device)} unique device_aids (paralel)...")
    with Pool(cpu_count()) as pool:
        results = pool.map(process_user, grouped_by_device)
    return dict(results)

def create_dummy_parquet(file_path='dummy_location_data.parquet'):
    """Test amacli parquet dosyasi kurar."""
    data = {
        'timestamp': pd.to_datetime(
            ['2023-01-01 08:00:00', '2023-01-01 09:00:00', '2023-01-02 14:00:00', '2023-01-15 22:00:00'] * 50 + 
            ['2023-01-01 06:00:00', '2023-01-02 07:00:00', '2023-01-03 08:00:00'] * 5 + 
            ['2023-02-05 10:00:00', '2023-02-06 11:00:00', '2023-02-07 12:00:00', 
             '2023-02-08 10:00:00', '2023-02-09 11:00:00', '2023-02-10 12:00:00',
             '2023-02-11 10:00:00', '2023-02-12 11:00:00', '2023-02-13 12:00:00',
             '2023-02-14 10:00:00', '2023-02-15 12:00:00'] + 
            ['2023-03-01 23:00:00', '2023-03-02 01:00:00', '2023-03-03 00:30:00'] * 70 
        ),
        'device_aid': ['user1'] * 200 + ['user2'] * 15 + ['user3'] * 11 + ['user4'] * 210,
        'longitude': [10.0] * 200 + [20.0] * 15 + [30.0, 30.1, 30.2, 30.0, 30.1, 30.2, 30.0, 30.1, 30.2, 30.0, 30.1] + [40.0] * 210,
        'latitude': [10.0] * 200 + [20.0] * 15 + [30.0] * 11 + [40.0] * 210,
        'horizontal_accuracy': [10.0, 15.0] * 100 + [50.0] * 15 + [8.0] * 11 + [150.0] * 210,
        'os': ['Android'] * 150 + ['iOS'] * 50 + ['Android'] * 15 + ['iOS'] * 11 + ['Android'] * 210,
    }
    
    # ilce ve mahalle bazli personalar
    user1_neighborhoods = []
    base_neighborhoods_u1 = ['OldTown', 'NewCity', 'MarketSquare', 'Parkside', 'Riverside', 'HilltopView']
    for i in range(200): user1_neighborhoods.append(base_neighborhoods_u1[i % len(base_neighborhoods_u1)]) # Makes user1 a Neighborhood Hopper

    user1_districts = []
    base_districts_u1 = ['Downtown', 'Uptown', 'Midtown', 'WestEnd']
    for i in range(200): user1_districts.append(base_districts_u1[i % len(base_districts_u1)]) # Makes user1 a District Explorer if not Hopper

    data['neighborhood'] = user1_neighborhoods + \
                           ['GreenValley'] * 15 + \
                           ['CafeStreet'] * 11 + \
                           ['WarehouseDistrict'] * 210
    
    data['district'] = user1_districts + \
                       ['SuburbA'] * 15 + \
                       ['CenterTown'] * 11 + \
                       ['IndustrialZone'] * 210

    dummy_df = pd.DataFrame(data)
    dummy_df.to_parquet(file_path)
    print(f"Generated dummy Parquet file at: {file_path}")

def main():
    parquet_path = "kahveseverler_parquet.parquet"
    df = pd.read_parquet(parquet_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Horizontal accuracy filtrelemesi - düşük doğruluklu GPS verilerini eleme
    initial_count = len(df)
    df = df[df['horizontal_accuracy'] <= 100]  # 100 metre altındaki veriler
    filtered_count = len(df)
    print(f"Horizontal accuracy filtrelemesi: {initial_count} -> {filtered_count} kayıt ({initial_count - filtered_count} veri elendi)")
    
    personas = generate_personas_parallel(df)
    personas_df = pd.DataFrame([
        {'device_aid': device_id, 'personas': ', '.join(persona_list)}
        for device_id, persona_list in personas.items()
    ])
    personas_df.to_csv('kahveseverler_persona.csv', index=False, encoding='utf-8-sig')
    print("Persona'lar 'kahveseverler_persona.csv' dosyasına kaydedildi.")

# POI'leri global olarak yükle (fonksiyon tanımlarından sonra!)
GUZELLIK_POLYS, GUZELLIK_NAMES = load_poi_polygons('guzellik_merkezi.geojson', name_key='POI Adı')
VETERINER_POLYS, VETERINER_NAMES = load_veteriner_excel('veteriner.xlsx')
GEZME_POLYS, GEZME_NAMES = load_poi_polygons('poi.geojson')

# --- YENİ EK: Mekanları yükle ---
def load_mechan_places(excel_path):
    df = pd.read_excel(excel_path)
    # Sadece gerekli sütunları al
    df = df[['MusteriTabelaAdi', 'SatisKanali', 'lat', 'lng']]
    df = df.dropna(subset=['lat', 'lng', 'SatisKanali'])
    return df

# Global olarak mekanları yükle
MEKAN_DF = load_mechan_places('Main Data.xlsx')

# --- YENİ EK: Kullanıcı ziyaretlerini analiz eden fonksiyon iskeleti ---
def get_modern_tradition_persona(user_df, mekan_df=MEKAN_DF, distance_threshold=0.0008):
    # Latitude/longitude'ları float'a çevir
    user_df = user_df.copy()
    user_df['latitude'] = pd.to_numeric(user_df['latitude'], errors='coerce')
    user_df['longitude'] = pd.to_numeric(user_df['longitude'], errors='coerce')
    mekan_df = mekan_df.copy()
    mekan_df['lat'] = pd.to_numeric(mekan_df['lat'], errors='coerce')
    mekan_df['lng'] = pd.to_numeric(mekan_df['lng'], errors='coerce')
    user_points = user_df[['latitude', 'longitude']].dropna().values
    if len(user_points) == 0:
        return None
    visited_modern = set()
    visited_tradition = set()
    for idx, mekan in mekan_df.iterrows():
        mekan_point = np.array([mekan['lat'], mekan['lng']])
        dists = np.linalg.norm(user_points - mekan_point, axis=1)
        if (dists < distance_threshold).any():
            if mekan['SatisKanali'].upper().startswith('MODERN'):
                visited_modern.add(mekan['MusteriTabelaAdi'])
            elif mekan['SatisKanali'].upper().startswith('TRADIT'):
                visited_tradition.add(mekan['MusteriTabelaAdi'])
    n_modern = len(visited_modern)
    n_tradition = len(visited_tradition)
    if n_modern > 3 and n_modern > n_tradition:
        return 'Modern İnsan'
    elif n_tradition > 3 and n_tradition > n_modern:
        return 'Geleneksel İnsan'
    return None

if __name__ == "__main__":
    main()