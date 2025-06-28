# 🏪 Staff Scheduler ML

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Inteligentny system planowania personelu wykorzystujący machine learning do optymalizacji obsady pracowniczej w supermarketach

## 📋 Spis treści

- [Opis projektu](#-opis-projektu)
- [Funkcjonalności](#-funkcjonalności)
- [Status projektu](#-status-projektu)
- [Instalacja](#-instalacja)
- [Użycie](#-użycie)
- [Model ML](#-model-ml)
- [Aplikacja webowa](#-aplikacja-webowa)
- [Struktura projektu](#-struktura-projektu)
- [API](#-api)
- [Licencja](#-licencja)

## 🎯 Opis projektu

Staff Scheduler ML to zaawansowany system wykorzystujący machine learning do przewidywania optymalnej liczby pracowników w supermarkecie. System analizuje wzorce ruchu klientów, sprzedaży i innych czynników wpływających na zapotrzebowanie na personel.

### Dlaczego ten projekt?

- **Optymalizacja kosztów** - Precyzyjne planowanie personelu redukuje koszty pracy
- **Lepsza obsługa klientów** - Właściwa liczba pracowników = krótsza kolejka
- **Data-driven decisions** - Decyzje oparte na danych, a nie intuicji
- **Automatyzacja** - Eliminuje manualne planowanie zmian

## ✨ Funkcjonalności

- 🤖 **Model ML** - Random Forest z R² = 0.780 i MAE = 0.452
- 📊 **Generator danych** - Syntetyczne dane supermarketu (6 miesięcy)
- 🖥️ **Aplikacja webowa** - Streamlit dashboard dla kierowników
- 📈 **Wizualizacje** - Interaktywne wykresy i heatmapy
- 💾 **Export danych** - CSV z przewidywaniami
- ⚡ **Real-time** - Przewidywania w czasie rzeczywistym

## 🚀 Status projektu

| Etap      | Status     | Opis                         |
| --------- | ---------- | ---------------------------- |
| ✅ Etap 1 | Zakończony | Generacja i analiza danych   |
| ✅ Etap 2 | Zakończony | Pierwszy model ML (R² > 0.7) |
| ✅ Etap 3 | Zakończony | Aplikacja Streamlit          |
| 🔄 Etap 4 | W planach  | Zaawansowane modele          |

## 🔧 Instalacja

### Wymagania systemowe

- Python 3.8+
- pip
- 4GB RAM (rekomendowane)

### Kroki instalacji

1. **Klonowanie repozytorium**

   ```bash
   git clone https://github.com/UFEQ1337/staff_scheduler_ml.git
   cd staff_scheduler_ml
   ```

2. **Instalacja zależności**

   ```bash
   pip install -r requirements.txt
   ```

3. **Generowanie danych**

   ```bash
   python src/data_generator.py
   ```

4. **Preprocessing**

   ```bash
   python src/preprocessing.py
   ```

5. **Trenowanie modelu**
   ```bash
   python src/basic_model.py
   ```

## 🚀 Użycie

### Uruchomienie aplikacji webowej

```bash
streamlit run app/streamlit_app.py
```

Aplikacja będzie dostępna na: http://localhost:8501

### Analiza w Jupyter

```bash
# Podstawowa analiza
jupyter notebook notebooks/01_basic_analysis.ipynb

# Model ML
jupyter notebook notebooks/02_first_model.ipynb
```

## 🤖 Model ML

### Architektura

**Algorytm:** Random Forest Regressor

- **Features:** 15 zmiennych (czas, kalendarz, pogoda)
- **Target:** Optymalna liczba pracowników (1-8)
- **Preprocessing:** Feature engineering, encoding

### Metryki wydajności

| Metryka     | Wartość | Cel   | Status |
| ----------- | ------- | ----- | ------ |
| R² Score    | 0.780   | > 0.7 | ✅     |
| MAE         | 0.452   | < 1.0 | ✅     |
| RMSE        | 1.2     | < 1.5 | ✅     |
| Overfitting | 2.1%    | < 10% | ✅     |

### Najważniejsze features

1. **Godziny szczytu** (16-19) - 47% ważności
2. **Godziny lunchu** (12-15) - 31% ważności
3. **Dzień tygodnia** - 12% ważności
4. **Weekend** - 6% ważności
5. **Pogoda** - 4% ważności

## 🖥️ Aplikacja webowa

### Interfejs użytkownika

- **📊 Dziś** - Przewidywania na bieżący dzień
- **📅 Tydzień** - Heatmapa tygodniowa
- **📈 Statystyki** - Metryki modelu i analiza

### Przykładowe przewidywania

| Godzina | Klienci/h | Pracownicy | Typ     |
| ------- | --------- | ---------- | ------- |
| 08:00   | 75        | 3          | Rano    |
| 12:00   | 100       | 4          | Lunch   |
| 17:00   | 125       | 5          | Szczyt  |
| 20:00   | 75        | 3          | Wieczór |

## 📁 Struktura projektu

```
staff_scheduler_ml/
├── 📁 app/
│   └── streamlit_app.py          # Aplikacja webowa
├── 📁 data/
│   ├── 📁 raw/                   # Surowe dane
│   └── 📁 processed/             # Dane przetworzone
├── 📁 models/                    # Zapisane modele ML
├── 📁 notebooks/                 # Jupyter notebooks
├── 📁 src/                       # Kod źródłowy
│   ├── data_generator.py         # Generator danych
│   ├── preprocessing.py          # Preprocessing
│   └── basic_model.py           # Model ML
├── requirements.txt              # Zależności
└── README.md                    # Dokumentacja
```

## 🔌 API

```python
from src.basic_model import StaffPredictionModel

# Inicjalizacja modelu
model = StaffPredictionModel()
model.load_model('models/staff_prediction_model.joblib')

# Przewidywanie dla konkretnej godziny
prediction = model.predict({
    'hour': 17,
    'day_of_week_encoded': 0,  # poniedziałek
    'is_weekend': 0,
    'weather_encoded': 0       # słonecznie
})

# Przewidywania na tydzień
week_predictions = model.predict_next_week()
```

## 📄 Licencja

Ten projekt jest licencjonowany na licencji MIT.

---

<div align="center">

**⭐ Jeśli ten projekt Ci pomógł, zostaw gwiazdkę! ⭐**

</div>
