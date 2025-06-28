"""
Moduł do przetwarzania danych dla modelu ML przewidywania liczby personelu.
Zawiera funkcje do feature engineering, encoding i podziału na zbiory train/test.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import os

# Ustawienie bazowej ścieżki projektu
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

def load_raw_data(filepath=None):
    """
    Ładuje surowe dane ze sklepu.
    
    Args:
        filepath: ścieżka do pliku CSV z danymi
        
    Returns:
        pd.DataFrame: DataFrame z surowymi danymi
    """
    if filepath is None:
        filepath = RAW_DATA_DIR / 'store_data.csv'
    
    df = pd.read_csv(filepath, encoding='utf-8')
    df['date'] = pd.to_datetime(df['date'])
    print(f"Załadowano {len(df)} rekordów z okresu {df['date'].min()} do {df['date'].max()}")
    return df

def create_cyclical_features(df):
    """
    Tworzy cykliczne features dla godziny (sin/cos) żeby model rozumiał cykliczność.
    
    Args:
        df: DataFrame z kolumną 'hour'
        
    Returns:
        pd.DataFrame: DataFrame z dodanymi kolumnami hour_sin, hour_cos
    """
    df = df.copy()
    
    # Konwersja godziny na radiany (0-23 -> 0-2π)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    print("✅ Dodano cykliczne features: hour_sin, hour_cos")
    return df

def encode_categorical_features(df):
    """
    Enkoduje features kategoryczne na liczby.
    
    Args:
        df: DataFrame z danymi
        
    Returns:
        pd.DataFrame: DataFrame z enkodowanymi features
        dict: słownik z encoderami do późniejszego użycia
    """
    df = df.copy()
    encoders = {}
    
    # Enkodowanie dnia tygodnia (poniedziałek=0, wtorek=1, ...)
    day_mapping = {
        'Poniedziałek': 0, 'Wtorek': 1, 'Środa': 2, 'Czwartek': 3,
        'Piątek': 4, 'Sobota': 5, 'Niedziela': 6
    }
    df['day_of_week_encoded'] = df['day_of_week'].map(day_mapping)
    
    # Enkodowanie pogody
    weather_encoder = LabelEncoder()
    df['weather_encoded'] = weather_encoder.fit_transform(df['weather'])
    encoders['weather'] = weather_encoder
    
    print("✅ Enkodowano features kategoryczne:")
    print(f"   - day_of_week: {day_mapping}")
    # Bezpieczne utworzenie mapowania dla weather
    weather_classes = weather_encoder.classes_
    weather_encoded = weather_encoder.transform(weather_classes)
    weather_mapping = dict(zip(weather_classes, weather_encoded)) if weather_classes is not None else {}
    print(f"   - weather: {weather_mapping}")
    
    return df, encoders

def create_target_variable(df):
    """
    Tworzy zmienną docelową: optimal_staff na podstawie rzeczywistych wzorców.
    Uwzględnia godziny szczytu, weekendy i inne czynniki.
    
    Args:
        df: DataFrame z danymi
        
    Returns:
        pd.DataFrame: DataFrame z dodaną kolumną optimal_staff_count
    """
    df = df.copy()
    
    # Bazowa liczba personelu na podstawie rzeczywistych danych
    base_staff = df['staff_working'].copy()
    
    # Korekty na podstawie analizy wzorców:
    # 1. Godziny szczytu wymagają więcej personelu
    peak_hours = (df['hour'] >= 16) & (df['hour'] <= 19)
    lunch_hours = (df['hour'] >= 12) & (df['hour'] <= 15)
    
    # 2. Weekendy wymagają więcej personelu
    weekend_boost = df['is_weekend'] * 0.3
    
    # 3. Świętowanie zwiększa potrzeby
    holiday_adjustment = df['is_holiday'] * (-0.2)
    
    # 4. Pogoda wpływa na ruch
    weather_adjustment = 0
    if 'weather_encoded' in df.columns:
        # Deszcz = więcej klientów w sklepie = więcej personelu
        weather_adjustment = (df['weather_encoded'] == 0) * 0.2  # deszcz
    
    # Obliczenie optymalnej liczby personelu
    optimal_staff = (base_staff + 
                    peak_hours * 1.0 +  # +1 osoba w godzinach szczytu
                    lunch_hours * 0.5 +  # +0.5 osoby w godzinach lunchu
                    weekend_boost +      # +30% w weekendy
                    holiday_adjustment + # -20% w święta
                    weather_adjustment)  # +20% w deszczu
    
    # Zaokrąglenie i ograniczenie do sensownego zakresu
    df['optimal_staff_count'] = np.round(optimal_staff).astype(int)
    df['optimal_staff_count'] = df['optimal_staff_count'].clip(lower=1, upper=8)
    
    print("✅ Utworzono ulepszoną target variable: optimal_staff_count")
    print(f"   - Zakres: {df['optimal_staff_count'].min()} - {df['optimal_staff_count'].max()}")
    print(f"   - Średnia: {df['optimal_staff_count'].mean():.1f}")
    print(f"   - Rozkład: {df['optimal_staff_count'].value_counts().to_dict()}")
    
    return df

def create_additional_features(df):
    """
    Tworzy dodatkowe features pomocne dla modelu.
    
    Args:
        df: DataFrame z danymi
        
    Returns:
        pd.DataFrame: DataFrame z dodatkowymi features
    """
    df = df.copy()
    
    # Miesiąc jako liczba
    df['month'] = df['date'].dt.month
    
    # Dzień miesiąca
    df['day_of_month'] = df['date'].dt.day
    
    # Czy to godzina szczytu (16-19)
    df['is_peak_hour'] = ((df['hour'] >= 16) & (df['hour'] <= 19)).astype(int)
    
    # Czy to godzina lunchu (12-15)
    df['is_lunch_hour'] = ((df['hour'] >= 12) & (df['hour'] <= 15)).astype(int)
    
    # Konwersja boolean na int
    df['is_weekend'] = df['is_weekend'].astype(int)
    df['is_holiday'] = df['is_holiday'].astype(int)
    
    print("✅ Dodano dodatkowe features: month, day_of_month, is_peak_hour, is_lunch_hour")
    return df

def split_train_test(df, test_days=30):
    """
    Dzieli dane na train/test - ostatnie 30 dni to test.
    
    Args:
        df: DataFrame z danymi
        test_days: liczba ostatnich dni do testu
        
    Returns:
        tuple: (train_df, test_df, split_date)
    """
    # Znajdź datę podziału
    max_date = df['date'].max()
    split_date = max_date - timedelta(days=test_days)
    
    # Podział danych
    train_df = df[df['date'] <= split_date].copy()
    test_df = df[df['date'] > split_date].copy()
    
    print(f"✅ Podział train/test:")
    print(f"   - Train: {len(train_df)} rekordów (do {split_date.strftime('%Y-%m-%d')})")
    print(f"   - Test: {len(test_df)} rekordów (od {(split_date + timedelta(days=1)).strftime('%Y-%m-%d')})")
    
    return train_df, test_df, split_date

def get_feature_columns():
    """
    Zwraca listę kolumn używanych jako features w modelu.
    
    Returns:
        list: lista nazw kolumn
    """
    return [
        'hour', 'hour_sin', 'hour_cos',
        'day_of_week_encoded', 'month', 'day_of_month',
        'is_weekend', 'is_holiday', 'is_peak_hour', 'is_lunch_hour',
        'weather_encoded'
    ]

def preprocess_data(filepath=None, save_processed=True):
    """
    Główna funkcja przetwarzająca dane - wykonuje wszystkie kroki preprocessing.
    
    Args:
        filepath: ścieżka do surowych danych
        save_processed: czy zapisać przetworzone dane
        
    Returns:
        tuple: (train_df, test_df, encoders, feature_columns)
    """
    print("🔄 Rozpoczynam preprocessing danych...")
    
    # 1. Ładowanie danych
    df = load_raw_data(filepath)
    
    # 2. Feature engineering
    df = create_cyclical_features(df)
    df, encoders = encode_categorical_features(df)
    df = create_target_variable(df)
    df = create_additional_features(df)
    
    # 3. Podział train/test
    train_df, test_df, split_date = split_train_test(df)
    
    # 4. Zapisanie przetworzonych danych
    if save_processed:
        # Utwórz katalog processed jeśli nie istnieje
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        train_df.to_csv(PROCESSED_DATA_DIR / 'train_data.csv', index=False, encoding='utf-8')
        test_df.to_csv(PROCESSED_DATA_DIR / 'test_data.csv', index=False, encoding='utf-8')
        print("💾 Zapisano przetworzone dane do data/processed/")
    
    # 5. Features do modelu
    feature_columns = get_feature_columns()
    
    print(f"✅ Preprocessing zakończony! Features: {len(feature_columns)}")
    print(f"Features: {feature_columns}")
    
    return train_df, test_df, encoders, feature_columns

if __name__ == "__main__":
    print("=== PREPROCESSING DANYCH ===")
    
    # Wykonanie preprocessing
    train_df, test_df, encoders, features = preprocess_data()
    
    # Podstawowe statystyki
    print("\n=== STATYSTYKI TARGET VARIABLE ===")
    print("Train set:")
    # Use pandas value_counts() with explicit Series handling to fix linter warnings  
    train_counts = pd.Series(train_df['optimal_staff_count']).value_counts().sort_index()
    print(train_counts)
    print("\nTest set:")
    test_counts = pd.Series(test_df['optimal_staff_count']).value_counts().sort_index()
    print(test_counts)
    
    print("\n✅ Preprocessing zakończony pomyślnie!") 