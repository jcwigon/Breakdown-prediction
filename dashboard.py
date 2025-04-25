#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import timedelta

# Wczytanie danych
df = pd.read_excel("Dane z mes.xlsx")

# Filtrujemy tylko awarie
df_awarie = df[df["Dispatch type"] == "01 Awaria"].copy()

# Konwersja kolumn czasowych na datetime
df_awarie["Time reported"] = pd.to_datetime(df_awarie["Time reported"])
df_awarie["Time completed"] = pd.to_datetime(df_awarie["Time completed"])

# Sortowanie po czasie zgłoszenia
df_awarie = df_awarie.sort_values("Time reported")

# Podgląd danych
print(df_awarie.head())
print("\nLiczba awarii na stację:\n", df_awarie["Station name"].value_counts())


# In[2]:


# Zakres dat od początku danych do dziś
date_range = pd.date_range(
    start=df_awarie["Time reported"].min().floor("D"),
    end=df_awarie["Time reported"].max().ceil("D"),
    freq="D"
)

# Lista unikalnych stacji
stacje = df_awarie["Station name"].unique()

# Przygotowanie pustej ramki danych
df_model = pd.DataFrame()

for stacja in stacje:
    # Dane dla konkretnej stacji
    df_stacja = df_awarie[df_awarie["Station name"] == stacja].copy()
    
    # Dla każdego dnia w zakresie dat
    for date in date_range:
        # Dane do tego dnia (włącznie)
        mask = df_stacja["Time reported"] <= date
        df_history = df_stacja[mask].copy()
        
        if len(df_history) > 0:
            # Ostatnia awaria przed tym dniem
            last_breakdown = df_history["Time reported"].max()
            days_since_last = (date - last_breakdown).days
            
            # Liczba awarii w ostatnich 7/14/30 dniach
            awarie_7d = df_history[
                df_history["Time reported"] >= (date - timedelta(days=7))
            ].shape[0]
            
            awarie_14d = df_history[
                df_history["Time reported"] >= (date - timedelta(days=14))
            ].shape[0]
            
            awarie_30d = df_history[
                df_history["Time reported"] >= (date - timedelta(days=30))
            ].shape[0]
            
            # Czy w ciągu 7 dni po tym dniu wystąpi awaria? (target)
            future_breakdowns = df_stacja[
                (df_stacja["Time reported"] > date) & 
                (df_stacja["Time reported"] <= date + timedelta(days=7))
            ]
            target = 1 if len(future_breakdowns) > 0 else 0
            
            # Dodanie do ramki wynikowej
            df_model = pd.concat([df_model, pd.DataFrame({
                "Date": date,
                "Station": stacja,
                "Days_since_last": days_since_last,
                "Awarii_7d": awarie_7d,
                "Awarii_14d": awarie_14d,
                "Awarii_30d": awarie_30d,
                "Target": target
            }, index=[0])], ignore_index=True)

# Zapis do pliku (opcjonalnie)
df_model.to_csv("dane_modelowe.csv", index=False)
print("Dane modelowe zapisane!")


# In[3]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Usuwanie braków danych (jeśli są)
df_model = df_model.dropna()

# Podział na cechy (X) i target (y)
X = df_model[["Days_since_last", "Awarii_7d", "Awarii_14d", "Awarii_30d"]]
y = df_model["Target"]

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trenowanie modelu
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predykcje i ewaluacja
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # Prawdopodobieństwo awarii

print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))


# In[4]:


import matplotlib.pyplot as plt

# Ważność cech
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

plt.barh(feature_importance["Feature"], feature_importance["Importance"])
plt.title("Ważność cech w modelu")
plt.show()


# In[9]:


for stacja in df["Station"].unique():
    df_stacja = df[df["Station"] == stacja]
    print(f"\n🔧 Stacja: {stacja}")
    
    # Upewnij się, że używasz poprawnej nazwy kolumny!
    print(f"Średnie ryzyko awarii: {df_stacja['Target'].mean():.0%}")  # Zmień 'Target' na właściwą nazwę
    
    print(f"Ostatnia awaria: {df_stacja['Time reported'].max()}")


# In[10]:


print("Dostępne kolumny w danych:")
print(df.columns.tolist())


# In[11]:


# Sortuj dane po stacji i czasie
df = df.sort_values(['Station', 'Time reported'])

# Stwórz kolumnę Target
df['Target'] = 0

for stacja in df['Station'].unique():
    stacja_mask = df['Station'] == stacja
    # Znajdź daty awarii dla danej stacji
    awarie_daty = df[stacja_mask & (df['Dispatch type'] == '01 Awaria')]['Time reported']
    
    # Oznacz rekordy, gdzie w ciągu 7 dni była awaria
    for data in awarie_daty:
        mask = (df['Station'] == stacja) & \
               (df['Time reported'] >= data - pd.Timedelta(days=7)) & \
               (df['Time reported'] < data)
        df.loc[mask, 'Target'] = 1


# In[12]:


print("Przykładowe rekordy z Target:")
print(df[['Station', 'Time reported', 'Dispatch type', 'Target']].sample(5))


# In[13]:


for stacja in df["Station"].unique():
    df_stacja = df[df["Station"] == stacja]
    print(f"\n🔧 Stacja: {stacja}")
    print(f"Średnie ryzyko awarii: {df_stacja['Target'].mean():.0%}")
    print(f"Ostatnia awaria: {df_stacja[df_stacja['Dispatch type'] == '01 Awaria']['Time reported'].max()}")


# In[14]:


print(f"Liczba rekordów z ryzykiem awarii: {df['Target'].sum()}")


# In[15]:


stacja_przyklad = df['Station'].iloc[0]
print(df[df['Station'] == stacja_przyklad][['Time reported', 'Target']].head(10))


# In[16]:


print(df.columns.tolist())


# In[17]:


import plotly.express as px
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# 1. Top 5 najbardziej awaryjnych stacji
st.subheader("🔥 Top 5 najbardziej awaryjnych stacji")
top_stacje = df[df['Dispatch type'] == '01 Awaria'].groupby('Station').size().nlargest(5).reset_index(name='Liczba awarii')
fig1 = px.bar(top_stacje, x='Station', y='Liczba awarii', 
             color='Liczba awarii', text_auto=True,
             title='Stacje z największą liczbą awarii')
st.plotly_chart(fig1, use_container_width=True)

# 2. Rozkład czasowy awarii dla wybranej stacji
st.subheader("📅 Historia awarii dla wybranej stacji")
selected_station = st.selectbox('Wybierz stację', df['Station'].unique())

# Filtruj dane dla wybranej stacji
station_data = df[df['Station'] == selected_station]
awarie_data = station_data[station_data['Dispatch type'] == '01 Awaria']

# Wykres czasowy
fig2 = px.scatter(awarie_data, x='Time reported', y='Station',
                 title=f'Awarie dla {selected_station} w czasie',
                 labels={'Time reported': 'Data zgłoszenia'})
st.plotly_chart(fig2, use_container_width=True)

# 3. Heatmapa awarii w ciągu tygodnia
st.subheader("🗓️ Rozkład awarii w dniach tygodnia")
df['DayOfWeek'] = df['Time reported'].dt.day_name()
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data = df[df['Dispatch type'] == '01 Awaria'].groupby(['Station', 'DayOfWeek']).size().unstack().reindex(columns=weekday_order)

fig3 = px.imshow(heatmap_data, 
                labels=dict(x="Dzień tygodnia", y="Stacja", color="Liczba awarii"),
                title='Awarie wg stacji i dnia tygodnia')
st.plotly_chart(fig3, use_container_width=True)

# 4. Wykres ryzyka awarii dla wybranej stacji
st.subheader("📊 Ryzyko awarii w czasie")
if 'Target' in df.columns:
    station_risk = df[df['Station'] == selected_station]
    fig4 = px.line(station_risk, x='Time reported', y='Target',
                  title=f'Ryzyko awarii dla {selected_station}',
                  labels={'Time reported': 'Data', 'Target': 'Ryzyko awarii (0-1)'})
    st.plotly_chart(fig4, use_container_width=True)
else:
    st.warning("Kolumna 'Target' nie istnieje - brak danych o ryzyku awarii")

# 5. Statystyki czasowe napraw
st.subheader("⏱️ Czas trwania awarii")
if 'Duration' in df.columns:
    fig5 = px.box(df[df['Dispatch type'] == '01 Awaria'], 
                 x='Station', y='Duration',
                 title='Rozkład czasów naprawy per stacja')
    st.plotly_chart(fig5, use_container_width=True)
else:
    st.warning("Kolumna 'Duration' nie istnieje - brak danych o czasie trwania awarii")


# In[18]:


import streamlit as st  # To jest kluczowy import!
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Inicjalizacja dashboardu Streamlit
st.set_page_config(page_title="Analiza Awarii", layout="wide")
st.title("📊 Dashboard Analizy Awarii Maszyn")

# 1. Top 5 najbardziej awaryjnych stacji
st.subheader("🔥 Top 5 najbardziej awaryjnych stacji")
top_stacje = df[df['Dispatch type'] == '01 Awaria'].groupby('Station').size().nlargest(5).reset_index(name='Liczba awarii')
fig1 = px.bar(top_stacje, x='Station', y='Liczba awarii', 
             color='Liczba awarii', text_auto=True,
             title='Stacje z największą liczbą awarii')
st.plotly_chart(fig1, use_container_width=True)

# 2. Rozkład czasowy awarii dla wybranej stacji
st.subheader("📅 Historia awarii dla wybranej stacji")
selected_station = st.selectbox('Wybierz stację', df['Station'].unique())

# Reszta Twojego kodu pozostaje bez zmian...


# In[19]:


st.set_page_config(page_title="Analiza Awarii", layout="wide")
st.title("📊 Dashboard Analizy Awarii Maszyn")


# In[20]:



# In[21]:


pip install streamlit


# In[22]:


import streamlit as st  # To jest kluczowy import!
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Inicjalizacja dashboardu Streamlit
st.set_page_config(page_title="Analiza Awarii", layout="wide")
st.title("📊 Dashboard Analizy Awarii Maszyn")

# 1. Top 5 najbardziej awaryjnych stacji
st.subheader("🔥 Top 5 najbardziej awaryjnych stacji")
top_stacje = df[df['Dispatch type'] == '01 Awaria'].groupby('Station').size().nlargest(5).reset_index(name='Liczba awarii')
fig1 = px.bar(top_stacje, x='Station', y='Liczba awarii', 
             color='Liczba awarii', text_auto=True,
             title='Stacje z największą liczbą awarii')
st.plotly_chart(fig1, use_container_width=True)

# 2. Rozkład czasowy awarii dla wybranej stacji
st.subheader("📅 Historia awarii dla wybranej stacji")
selected_station = st.selectbox('Wybierz stację', df['Station'].unique())

# Reszta Twojego kodu pozostaje bez zmian...


# In[23]:


st.set_page_config(page_title="Analiza Awarii", layout="wide")
st.title("📊 Dashboard Analizy Awarii Maszyn")


# In[24]:


streamlit run dashboard.py


# In[25]:


# W komórce notebooka:
get_ipython().system('pip install streamlit  # Jeśli nie masz zainstalowanego')
get_ipython().system('streamlit run dashboard.py')


# In[26]:


import plotly.express as px

# Przykładowy wykres
fig = px.bar(df, x='Station', y='Duration')
fig.show()


# In[27]:


import streamlit as st
import pandas as pd
import plotly.express as px
from joblib import load

# Wczytanie danych i modelu
@st.cache_data
def load_data():
    return pd.read_csv("app/dane_produkcyjne.csv")

@st.cache_resource
def load_model():
    return load("app/model_awarie.pkl")  # Jeśli używasz modelu

df = load_data()
model = load_model() if 'model_awarie.pkl' in os.listdir("app") else None

# Dashboard
st.title("Predykcja awarii maszyn")
st.plotly_chart(px.bar(df, x='Station', y='Duration', title='Czasy awarii per stacja'))


# In[28]:


# app/dashboard.py
import streamlit as st
import pandas as pd

st.title("Predykcja awarii DB05")
# ... reszta Twojego kodu


# In[ ]:




