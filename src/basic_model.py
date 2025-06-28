"""
Podstawowy model ML do przewidywania optymalnej liczby personelu w supermarkecie.
Używa Random Forest Regressor do predykcji na podstawie różnych features.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from typing import Optional, Union, Dict, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def find_project_root():
    """
    Znajduje root katalogu projektu, szukając pliku requirements.txt.
    
    Returns:
        str: ścieżka do root katalogu projektu
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir
    
    # Idź w górę aby znaleźć root projekt (gdzie jest requirements.txt)
    while project_root and not os.path.exists(os.path.join(project_root, 'requirements.txt')):
        parent = os.path.dirname(project_root)
        if parent == project_root:  # Dotarliśmy do root systemu
            break
        project_root = parent
    
    return project_root

class StaffPredictionModel:
    """
    Model do przewidywania optymalnej liczby personelu w supermarkecie.
    """
    
    def __init__(self, n_estimators=50, max_depth=8, min_samples_split=20, min_samples_leaf=10, random_state=42):
        """
        Inicjalizacja modelu z regularyzacją przeciw overfittingowi.
        
        Args:
            n_estimators: liczba drzew w Random Forest (zmniejszona z 100 do 50)
            max_depth: maksymalna głębokość drzewa (ograniczona do 8)
            min_samples_split: minimalna liczba próbek do podziału węzła
            min_samples_leaf: minimalna liczba próbek w liściu
            random_state: seed dla powtarzalności wyników
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1  # Użyj wszystkich dostępnych rdzeni
        )
        self.feature_columns = None
        self.encoders = None
        self.is_trained = False
        
    def train(self, train_df, feature_columns, encoders=None):
        """
        Trenuje model na danych treningowych.
        
        Args:
            train_df: DataFrame z danymi treningowymi
            feature_columns: lista kolumn używanych jako features
            encoders: słownik z encoderami (opcjonalny)
        """
        print("🚀 Rozpoczynam trenowanie modelu Random Forest...")
        
        # Zapisz konfigurację
        self.feature_columns = feature_columns
        self.encoders = encoders
        
        # Przygotowanie danych
        X_train = train_df[feature_columns]
        y_train = train_df['optimal_staff_count']
        
        print(f"📊 Dane treningowe: {len(X_train)} próbek, {len(feature_columns)} features")
        print(f"Features: {feature_columns}")
        
        # Trenowanie modelu
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Obliczenie metryki na zbiorze treningowym
        train_predictions = self.model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_predictions)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
        train_r2 = r2_score(y_train, train_predictions)
        
        print("✅ Model wytrenowany!")
        print(f"📈 Metryki na zbiorze treningowym:")
        print(f"   - MAE: {train_mae:.3f}")
        print(f"   - RMSE: {train_rmse:.3f}")
        print(f"   - R²: {train_r2:.3f}")
        
        return {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2
        }
    
    def evaluate(self, test_df):
        """
        Ewaluuje model na danych testowych.
        
        Args:
            test_df: DataFrame z danymi testowymi
            
        Returns:
            dict: słownik z metrykami
        """
        if not self.is_trained:
            raise ValueError("Model nie jest wytrenowany! Użyj metody train() najpierw.")
        
        print("📊 Ewaluacja modelu na zbiorze testowym...")
        
        # Przygotowanie danych
        X_test = test_df[self.feature_columns]
        y_test = test_df['optimal_staff_count']
        
        # Predykcje
        predictions = self.model.predict(X_test)
        
        # Obliczenie metryk
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        print("📈 Metryki na zbiorze testowym:")
        print(f"   - MAE: {mae:.3f} pracowników")
        print(f"   - RMSE: {rmse:.3f} pracowników")
        print(f"   - R²: {r2:.3f}")
        
        # Sprawdzenie kryteriów sukcesu
        print("\n🎯 Sprawdzenie kryteriów sukcesu:")
        print(f"   - R² > 0.7: {'✅' if r2 > 0.7 else '❌'} ({r2:.3f})")
        print(f"   - MAE < 1.0: {'✅' if mae < 1.0 else '❌'} ({mae:.3f})")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions,
            'actual': y_test
        }
    
    def predict(self, data):
        """
        Przewiduje liczbę personelu dla nowych danych.
        
        Args:
            data: DataFrame lub dict z danymi do predykcji
            
        Returns:
            array: przewidywana liczba personelu
        """
        if not self.is_trained:
            raise ValueError("Model nie jest wytrenowany! Użyj metody train() najpierw.")
        
        # Konwersja dict na DataFrame jeśli potrzebne
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Predykcja
        X = data[self.feature_columns]
        predictions = self.model.predict(X)
        
        # Zaokrąglij do liczb całkowitych i ogranicz do sensownego zakresu
        predictions = np.round(predictions).astype(int)
        predictions = np.clip(predictions, 1, 8)
        
        return predictions
    
    def get_feature_importance(self, plot=True, top_n=10):
        """
        Zwraca ważność features w modelu.
        
        Args:
            plot: czy stworzyć wykres
            top_n: ile top features pokazać na wykresie
            
        Returns:
            pd.DataFrame: DataFrame z ważnością features
        """
        if not self.is_trained:
            raise ValueError("Model nie jest wytrenowany! Użyj metody train() najpierw.")
        
        # Uzyskaj ważność features
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        if plot:
            plt.figure(figsize=(10, 6))
            top_features = feature_importance.head(top_n)
            
            sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
            plt.title(f'Top {top_n} najważniejszych features', fontsize=14, fontweight='bold')
            plt.xlabel('Ważność', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.tight_layout()
            plt.show()
        
        return feature_importance
    
    def predict_next_week(self, base_date=None, weather='słonecznie') -> pd.DataFrame:
        """
        Przewiduje liczbę personelu na następny tydzień.
        
        Args:
            base_date: data początkowa (domyślnie dziś)
            weather: przewidywana pogoda
            
        Returns:
            pd.DataFrame: DataFrame z przewidywaniami
        """
        if not self.is_trained:
            raise ValueError("Model nie jest wytrenowany! Użyj metody train() najpierw.")
        
        if base_date is None:
            base_date = datetime.now().date()
        
        # Enkodowanie pogody
        weather_encoded = 0  # domyślnie 'słonecznie'
        if self.encoders and 'weather' in self.encoders:
            try:
                weather_encoded = self.encoders['weather'].transform([weather])[0]
            except:
                weather_encoded = 0
        
        predictions_data = []
        
        # Dla każdego dnia tygodnia
        for day_offset in range(7):
            current_date = base_date + timedelta(days=day_offset)
            day_of_week = current_date.weekday()  # 0=poniedziałek
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # Mapowanie dnia tygodnia
            day_names = ['Poniedziałek', 'Wtorek', 'Środa', 'Czwartek', 'Piątek', 'Sobota', 'Niedziela']
            day_name = day_names[day_of_week]
            
            # Dla każdej godziny pracy (8-22)
            for hour in range(8, 23):
                # Przygotowanie danych do predykcji
                hour_sin = np.sin(2 * np.pi * hour / 24)
                hour_cos = np.cos(2 * np.pi * hour / 24)
                is_peak_hour = 1 if 16 <= hour <= 19 else 0
                is_lunch_hour = 1 if 12 <= hour <= 15 else 0
                
                data = {
                    'hour': hour,
                    'hour_sin': hour_sin,
                    'hour_cos': hour_cos,
                    'day_of_week_encoded': day_of_week,
                    'month': current_date.month,
                    'day_of_month': current_date.day,
                    'is_weekend': is_weekend,
                    'is_holiday': 0,  # Zakładamy brak świąt
                    'is_peak_hour': is_peak_hour,
                    'is_lunch_hour': is_lunch_hour,
                    'weather_encoded': weather_encoded
                }
                
                # Predykcja
                predicted_staff = self.predict(data)[0]
                
                predictions_data.append({
                    'date': current_date,
                    'day_name': day_name,
                    'hour': hour,
                    'predicted_staff': predicted_staff
                })
        
        return pd.DataFrame(predictions_data)
    
    def save_model(self, filepath='models/staff_prediction_model.joblib'):
        """
        Zapisuje wytrenowany model do pliku.
        
        Args:
            filepath: ścieżka do pliku
        """
        if not self.is_trained:
            raise ValueError("Model nie jest wytrenowany! Użyj metody train() najpierw.")
        
        # Jeśli używamy względnej ścieżki, upewnij się, że jest względem root projektu
        if not os.path.isabs(filepath):
            project_root = find_project_root()
            filepath = os.path.join(project_root, filepath)
        
        # Upewnij się, że katalog istnieje
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Zapisz cały obiekt modelu
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'encoders': self.encoders,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model zapisany do: {filepath}")
    
    def load_model(self, filepath='models/staff_prediction_model.joblib'):
        """
        Ładuje model z pliku.
        
        Args:
            filepath: ścieżka do pliku
        """
        # Jeśli używamy względnej ścieżki, upewnij się, że jest względem root projektu
        if not os.path.isabs(filepath):
            project_root = find_project_root()
            filepath = os.path.join(project_root, filepath)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Plik modelu nie został znaleziony: {filepath}")
            
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.encoders = model_data['encoders']
        self.is_trained = model_data['is_trained']
        
        print(f"📂 Model załadowany z: {filepath}")

def train_and_evaluate_model(train_df, test_df, feature_columns, encoders=None):
    """
    Funkcja pomocnicza do szybkiego trenowania i ewaluacji modelu.
    
    Args:
        train_df: dane treningowe
        test_df: dane testowe
        feature_columns: lista features
        encoders: encodery
        
    Returns:
        tuple: (model, metrics)
    """
    print("=== TRENOWANIE I EWALUACJA MODELU ===")
    
    # Stworzenie i trenowanie modelu
    model = StaffPredictionModel()
    train_metrics = model.train(train_df, feature_columns, encoders)
    
    # Ewaluacja
    test_metrics = model.evaluate(test_df)
    
    # Zapisanie modelu
    model.save_model()
    
    # Zwrócenie wyników
    all_metrics = {**train_metrics, **test_metrics}
    
    return model, all_metrics

if __name__ == "__main__":
    print("=== BASIC MODEL DEMO ===")
    
    # Import preprocessing
    from preprocessing import preprocess_data
    
    # Preprocessing danych
    train_df, test_df, encoders, features = preprocess_data()
    
    # Trenowanie i ewaluacja
    model, metrics = train_and_evaluate_model(train_df, test_df, features, encoders)
    
    # Feature importance
    print("\n=== WAŻNOŚĆ FEATURES ===")
    importance_df = model.get_feature_importance(plot=False)
    print(importance_df.head(10))
    
    # Przykładowe przewidywania
    print("\n=== PRZYKŁADOWE PRZEWIDYWANIA ===")
    predictions_df = model.predict_next_week()
    
    # Pokaż przewidywania na poniedziałek
    monday_predictions = predictions_df.loc[predictions_df['day_name'] == 'Poniedziałek'].copy()
    key_hours = monday_predictions.loc[monday_predictions['hour'].isin([8, 12, 17])].copy()
    
    print("Przewidywania na poniedziałek:")
    for _, row in key_hours.iterrows():
        print(f"   {row['hour']:02d}:00 - {row['predicted_staff']} pracowników")
    
    print("\n✅ Model gotowy do użycia!") 