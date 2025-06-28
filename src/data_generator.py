"""
Generator danych syntetycznych dla systemu planowania personelu supermarketu.
Generuje dane za cały rok 2024 z realistycznymi wzorcami dużego polskiego sklepu.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker

# Ustawienie seed dla powtarzalności wyników
np.random.seed(42)
random.seed(42)
fake = Faker('pl_PL')

def generate_store_data():
    """
    Generuje syntetyczne dane dużego sklepu za cały rok 2024
    """
    print("Rozpoczynam generowanie danych dla dużego supermarketu...")
    
    # Definicja okresu danych - cały rok 2024
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # Polskie święta w całym roku 2024
    polish_holidays = [
        datetime(2024, 1, 1),   # Nowy Rok
        datetime(2024, 1, 6),   # Trzech Króli
        datetime(2024, 3, 31),  # Wielkanoc
        datetime(2024, 4, 1),   # Poniedziałek Wielkanocny
        datetime(2024, 5, 1),   # Święto Pracy
        datetime(2024, 5, 3),   # Święto Konstytucji 3 Maja
        datetime(2024, 5, 30),  # Boże Ciało
        datetime(2024, 8, 15),  # Wniebowzięcie NMP
        datetime(2024, 11, 1),  # Wszystkich Świętych
        datetime(2024, 11, 11), # Święto Niepodległości
        datetime(2024, 12, 25), # Boże Narodzenie
        datetime(2024, 12, 26), # Drugi dzień Świąt
    ]
    
    # Słownik polskich dni tygodnia
    polish_days = {
        0: 'Poniedziałek',
        1: 'Wtorek', 
        2: 'Środa',
        3: 'Czwartek',
        4: 'Piątek',
        5: 'Sobota',
        6: 'Niedziela'
    }
    
    data = []
    current_date = start_date
    
    while current_date <= end_date:
        # Sprawdzenie czy to weekend i święto
        is_weekend = current_date.weekday() >= 5
        is_holiday = current_date in polish_holidays
        day_name = polish_days[current_date.weekday()]
        
        # Sezonowość - zwięksi ruch przed świętami i w okresie letnim
        month = current_date.month
        seasonal_multiplier = 1.0
        if month in [12, 1]:  # Grudzień/Styczeń - większy ruch
            seasonal_multiplier = 1.3
        elif month in [7, 8]:  # Wakacje - mniejszy ruch w dni robocze
            seasonal_multiplier = 0.9 if not is_weekend else 1.2
        elif month in [11]:   # Listopad - większy ruch przed świętami
            seasonal_multiplier = 1.15
        
        # Generowanie danych dla każdej godziny (7-23) - dłuższe godziny otwarcia
        for hour in range(7, 24):
            # Bazowa liczba klientów zależna od godziny - WIĘKSZY SKLEP
            if 16 <= hour <= 19:  # Peak hours
                base_customers = np.random.normal(120, 20)  # Zwiększone z 45
            elif 12 <= hour <= 15:  # Lunch time
                base_customers = np.random.normal(95, 15)   # Zwiększone z 35
            elif 9 <= hour <= 11:   # Morning
                base_customers = np.random.normal(70, 12)   # Zwiększone z 25
            elif 20 <= hour <= 22:  # Evening
                base_customers = np.random.normal(60, 10)   # Zwiększone z 20
            else:  # Early morning/late evening
                base_customers = np.random.normal(35, 8)    # Nowe przedziały
            
            # Modyfikatory
            customers = max(10, base_customers)  # Minimum 10 klientów
            
            # Sezonowość
            customers *= seasonal_multiplier
            
            # Weekend boost
            if is_weekend:
                customers *= 1.5  # Zwiększone z 1.4
            
            # Święta - mniej klientów
            if is_holiday:
                customers *= 0.5  # Zmniejszone z 0.6
            
            # Pogoda
            weather_options = ['słonecznie', 'pochmurnie', 'deszcz', 'śnieg']
            # Sezonowa pogoda
            if month in [12, 1, 2]:  # Zima
                weather_probs = [0.2, 0.4, 0.2, 0.2]
            elif month in [6, 7, 8]:  # Lato
                weather_probs = [0.5, 0.3, 0.2, 0.0]
            else:  # Wiosna/jesień
                weather_probs = [0.4, 0.4, 0.2, 0.0]
            
            weather = np.random.choice(weather_options, p=weather_probs)
            
            # Wpływ pogody na ruch
            if weather == 'deszcz':
                customers *= 1.25
            elif weather == 'śnieg':
                customers *= 0.8
            elif weather == 'słonecznie' and is_weekend:
                customers *= 0.9  # Ludzie wychodzą na zewnątrz
            
            customers = int(round(customers))
            
            # Liczba personelu - dla WIĘKSZEGO SKLEPU
            if customers <= 30:
                staff_working = 4   # Minimum 4 osoby
            elif customers <= 60:
                staff_working = 6
            elif customers <= 90:
                staff_working = 8
            elif customers <= 120:
                staff_working = 10
            elif customers <= 150:
                staff_working = 12
            else:
                staff_working = 15  # Maksimum 15 osób na zmianie
            
            # Dodanie losowości do personelu
            if random.random() < 0.08:  # 8% szans na niedostatek personelu
                staff_working = max(3, staff_working - 2)
            elif random.random() < 0.05:  # 5% szans na nadmiar personelu
                staff_working += 2
            
            # Sprzedaż - koreluje z liczbą klientów
            avg_basket = np.random.normal(95, 18)  # Zwiększony średni koszyk z 85
            sales = customers * avg_basket
            
            # Weekend - wyższe koszyki
            if is_weekend:
                sales *= 1.2  # Zwiększone z 1.15
            
            # Święta - niższe koszyki ale więcej produktów premium
            if is_holiday:
                sales *= 0.95  # Zmniejszone z 0.9
            
            # Sezonowość sprzedaży
            if month == 12:  # Grudzień - wyższe koszyki
                sales *= 1.4
            elif month in [7, 8]:  # Wakacje
                sales *= 1.1
            
            sales = round(sales, 2)
            
            # Dodanie rekordu
            data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'day_of_week': day_name,
                'hour': hour,
                'customers_per_hour': customers,
                'staff_working': staff_working,
                'sales_pln': sales,
                'is_weekend': is_weekend,
                'is_holiday': is_holiday,
                'weather': weather,
                'month': month,
                'seasonal_multiplier': round(seasonal_multiplier, 2)
            })
        
        current_date += timedelta(days=1)
        
        # Progress indicator co 30 dni
        if (current_date - start_date).days % 30 == 0:
            print(f"Postęp: {current_date.strftime('%Y-%m-%d')} ({((current_date - start_date).days / 365 * 100):.1f}%)")
    
    # Tworzenie DataFrame
    df = pd.DataFrame(data)
    
    print(f"Wygenerowano {len(df)} rekordów")
    print(f"Okres: {start_date.strftime('%Y-%m-%d')} do {end_date.strftime('%Y-%m-%d')}")
    
    return df

def save_data(df, filename='data/raw/store_data.csv'):
    """
    Zapisuje DataFrame do pliku CSV
    """
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"Dane zapisane do pliku: {filename}")
    
    # Podstawowe statystyki
    print("\n=== PODSTAWOWE STATYSTYKI ===")
    print(f"Łączna liczba rekordów: {len(df)}")
    print(f"Średnia liczba klientów na godzinę: {df['customers_per_hour'].mean():.1f}")
    print(f"Łączna sprzedaż: {df['sales_pln'].sum():,.2f} PLN")
    print(f"Średni personel na zmianie: {df['staff_working'].mean():.1f}")
    print(f"Maksymalna liczba klientów w godzinie: {df['customers_per_hour'].max()}")
    print(f"Maksymalna liczba personelu: {df['staff_working'].max()}")
    
    print("\n=== ROZKŁAD POGODY ===")
    print(df['weather'].value_counts())
    
    print("\n=== WEEKENDY VS DNI ROBOCZE ===")
    weekend_stats = df.groupby('is_weekend')['customers_per_hour'].mean()
    print(f"Dni robocze: {weekend_stats[False]:.1f} klientów/h")
    print(f"Weekendy: {weekend_stats[True]:.1f} klientów/h")
    print(f"Wzrost w weekendy: {((weekend_stats[True]/weekend_stats[False])-1)*100:.1f}%")
    
    print("\n=== SEZONOWOŚĆ (średnia klientów/h) ===")
    monthly_stats = df.groupby('month')['customers_per_hour'].mean()
    month_names = {1: 'Styczeń', 2: 'Luty', 3: 'Marzec', 4: 'Kwiecień', 
                   5: 'Maj', 6: 'Czerwiec', 7: 'Lipiec', 8: 'Sierpień',
                   9: 'Wrzesień', 10: 'Październik', 11: 'Listopad', 12: 'Grudzień'}
    for month, avg in monthly_stats.items():
        print(f"{month_names[month]}: {avg:.1f} klientów/h")
    
    print("\n=== SPRZEDAŻ MIESIĘCZNA ===")
    monthly_sales = df.groupby('month')['sales_pln'].sum()
    for month, sales in monthly_sales.items():
        print(f"{month_names[month]}: {sales:,.2f} PLN")

if __name__ == "__main__":
    print("=== GENERATOR DANYCH DUŻEGO SUPERMARKETU ===")
    
    # Generowanie danych
    df = generate_store_data()
    
    # Zapisanie do pliku
    save_data(df)
    
    print("\n✅ Generator zakończył pracę pomyślnie!")
    print("Dane obejmują cały rok 2024 dla dużego supermarketu.")
    print("Uruchom notebook 01_basic_analysis.ipynb aby zobaczyć analizę danych.") 