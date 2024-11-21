#imports
from io import BytesIO
import streamlit as st
# do pracy z sekretami i kluczami
from dotenv import dotenv_values
# do pracy z qdrantem
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Range, Filter, FieldCondition, MatchText, MatchValue, SearchRequest
from qdrant_client.http import models
from openai import OpenAI
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pandas as pd
from dotenv import dotenv_values
from typing import Union, List

# functions

def load_data5():
    if st.session_state.data is None:
        option = st.selectbox("Choose an option", ("Upload .xlsx", "Upload .csv"))
         
        if option == "Upload .xlsx":
            image = st.file_uploader("Upload a photo", type=["xlsx"])
        elif option == "Upload .csv":
            image = st.file_uploader("Upload a photo", type=["xlsx"])

@st.cache_data
def load_data():
    """
    Funkcja do wczytywania plików Excel (.xlsx) lub CSV.
    Zwraca DataFrame lub None w przypadku braku pliku.
    """
    option = st.selectbox(
        "Wybierz format pliku", 
        ("Upload .xlsx", "Upload .csv"),
        key="file_format"
    )
    
    try:
        if option == "Upload .xlsx":
            uploaded_file = st.file_uploader(
                "Wybierz plik Excel",
                type=["xlsx"],
                key="excel_uploader"
            )
            
            if uploaded_file is not None:
                st.session_state.data = pd.read_excel(uploaded_file)
                st.success(f"Pomyślnie wczytano plik: {uploaded_file.name}")
                return st.session_state.data
                
        elif option == "Upload .csv":
            uploaded_file = st.file_uploader(
                "Wybierz plik CSV",
                type=["csv"],
                key="csv_uploader"
            )
            
            if uploaded_file is not None:
                # Dodanie opcji kodowania i separatora dla CSV
                encoding_option = st.selectbox(
                    "Wybierz kodowanie pliku",
                    options=["utf-8", "cp1250", "iso-8859-1"],
                    key="encoding"
                )
                
                separator_option = st.selectbox(
                    "Wybierz separator",
                    options=[",", ";", "|", "\t"],
                    key="separator"
                )
                
                st.session_state.data = pd.read_csv(
                    uploaded_file,
                    encoding=encoding_option,
                    sep=separator_option
                )
                st.success(f"Pomyślnie wczytano plik: {uploaded_file.name}")
                return st.session_state.data
                
        # Jeśli żaden plik nie został wczytany
        if st.session_state.data is None:
            st.info("Proszę wybrać plik do wczytania")
            
    except Exception as e:
        st.error(f"Wystąpił błąd podczas wczytywania pliku: {str(e)}")
        st.session_state.data = None
        return None

# pre main
env = dotenv_values(".env") #load secrets
# st.text(env) 

#
# MAIN
#

# Inicjalizacja session state jeśli nie istnieje
if 'data' not in st.session_state:
    st.session_state.data = None

st.set_page_config(page_title="Test context search", layout="centered")
st.title("Hi, I compare text and context (embedding) search of Your data")
# st.markdown(env) #temporary print secrets (only for test)

def main():
    st.title("Wczytywanie danych z pliku")
    
    # Przycisk do resetowania danych
    if st.button("Resetuj dane"):
        st.session_state.data = None
        st.experimental_rerun()
    
    # Wczytaj dane
    df = load_data()
    
    # Wyświetl dane jeśli są dostępne
    if df is not None:
        st.write("Podgląd wczytanych danych:")
        st.dataframe(df.head())
        
        # Podstawowe informacje o danych
        st.write("Informacje o danych:")
        st.write(f"Liczba wierszy: {df.shape[0]}")
        st.write(f"Liczba kolumn: {df.shape[1]}")
        st.write("Nazwy kolumn:", df.columns.tolist())

if __name__ == "__main__":
    main()