import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import timedelta
import os

# Konfiguracja strony
st.set_page_config(page_title="Predykcja Awarii DB05", layout="wide")
st.title("📊 Analiza Awarii Maszyn")

# Funkcje pomocnicze
@st.cache_data
def load_data():
    """Wczytanie danych z pliku"""
    try:
        # Ścieżka względna do pliku w folderze app/
        return pd.read_excel("app/Dane_z_mes.xlsx")
    except Exception as e:
        st.error(f"Błąd ładowania danych: {e}")
        return None

@st.cache_data
def prepare_data(df):
    """Przygotowanie danych do analizy"""
    df_awarie = df[df["Dispatch type"] == "01 Awaria"].copy()
    df_awarie["Time reported"] = pd.to_datetime(df_awarie["Time reported"])
    df_awarie["Time completed"] = pd.to_datetime(df_awarie["Time completed"])
    return df_awarie.sort_values("Time reported")

# Ładowanie danych
df_raw = load_data()
if df_raw is None:
    st.stop()

df = prepare_data(df_raw)

# Sekcja 1: Podstawowe statystyki
st.header("Podstawowe statystyki")
col1, col2 = st.columns(2)
with col1:
    st.metric("Liczba stacji", df["Station"].nunique())
with col2:
    st.metric("Łączna liczba awarii", len(df))

# Sekcja 2: Najbardziej awaryjne stacje
st.header("Top 5 najbardziej awaryjnych stacji")
top_stations = df["Station"].value_counts().nlargest(5)
st.bar_chart(top_stations)

# Sekcja 3: Analiza czasowa
st.header("Analiza czasowa awarii")
selected_station = st.selectbox("Wybierz stację", df["Station"].unique())

station_data = df[df["Station"] == selected_station]
if not station_data.empty:
    fig = px.line(
        station_data,
        x="Time reported",
        y="Duration",
        title=f"Czasy awarii dla stacji {selected_station}"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Brak danych dla wybranej stacji")

# Sekcja 4: Predykcja awarii (uproszczona)
st.header("Predykcja ryzyka awarii")

if st.button("Oblicz ryzyko awarii"):
    # Uproszczony model predykcji
    risk_by_station = df.groupby("Station").size() / len(df)
    st.write("Ryzyko względne awarii:")
    st.dataframe(risk_by_station.sort_values(ascending=False))
