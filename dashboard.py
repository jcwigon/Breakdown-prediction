import streamlit as st
import pandas as pd
import plotly.express as px

# Konfiguracja aplikacji
st.set_page_config(
    page_title="Predykcja Awarii DB05", 
    layout="wide",
    page_icon="🔧"
)

# Tytuł
st.title("Analiza Awarii Maszyn Produkcyjnych")

# Funkcja ładowania danych
@st.cache_data
def load_data():
    """Wczytuje dane z pliku Excel"""
    try:
        return pd.read_excel("app/Dane_z_mes.xlsx")
    except Exception as e:
        st.error(f"Błąd ładowania danych: {str(e)}")
        return None

# Główna logika
def main():
    df = load_data()
    if df is None:
        return

    # Filtruj tylko awarie
    df_awarie = df[df["Dispatch type"] == "01 Awaria"].copy()
    
    # Sekcja wizualizacji
    st.header("Statystyki awarii")
    
    # Top 5 stacji
    top_stacje = df_awarie["Station name"].value_counts().nlargest(5)
    fig = px.bar(
        top_stacje,
        title="Najczęściej awariujące stacje",
        labels={'value': 'Liczba awarii', 'index': 'Stacja'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Analiza per stacja
    selected_station = st.selectbox(
        "Wybierz stację do analizy",
        df_awarie["Station name"].unique()
    )
    
    st.subheader(f"Historia awarii dla: {selected_station}")
    station_data = df_awarie[df_awarie["Station name"] == selected_station]
    st.dataframe(station_data)

# Uruchomienie aplikacji
if __name__ == "__main__":
    main()
