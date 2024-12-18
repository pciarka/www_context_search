from dotenv import dotenv_values
import streamlit as st
import pandas as pd

from qdrant_communication import get_collection_info, sentence_search, sentence_transtormer_load_data, openAI_load_data, open_AI_search
from data_manipulation import load_data
import global_variables

QDRANT_COLLECTION_NAME_SENTENCE = "shop_data_sentence_transformer"
QDRANT_COLLECTION_NAME_AI = "shop_data_openAI"
EMBEDDING_DIM = 1536
EMBEDDING_MODEL = "text-embedding-ada-002" #OpenAI used model

def test_searching():
    """
    Zakładka do testowania wyszukiwania w danych
    """
    # Sprawdź czy dane są załadowane
    if st.session_state.data is None:
        st.warning("Load data first at tab 'Load data'")
        return
    
    # Pobierz aktualnie załadowane dane
    df = st.session_state.data
    
    st.subheader("Narzędzia wyszukiwania")
    
    # Wybór kolumn do przeszukiwania
    columns = df.columns.tolist()
    selected_columns = st.multiselect(
        "Wybierz kolumny do przeszukiwania",
        columns,
        default=columns[:min(3, len(columns))]
    )
    
    # Tryby wyszukiwania
    search_mode = st.radio(
        "Tryb wyszukiwania",
        ["Dokładne dopasowanie", "Częściowe dopasowanie", "Regex"]
    )
    
    # Pole wyszukiwania
    search_query = st.text_input("Wprowadź frazę do wyszukania")
    
    # Przycisk wyszukiwania
    if st.button("Szukaj"):
        if not search_query:
            st.warning("Wprowadź frazę do wyszukania")
            return
        
        # Filtrowanie danych
        if search_mode == "Dokładne dopasowanie":
            mask = df[selected_columns].isin([search_query]).any(axis=1)
        elif search_mode == "Częściowe dopasowanie":
            mask = df[selected_columns].apply(lambda col: col.astype(str).str.contains(search_query, case=False, na=False))
            mask = mask.any(axis=1)
        else:  # Regex
            mask = df[selected_columns].apply(lambda col: col.astype(str).str.contains(search_query, regex=True, case=False, na=False))
            mask = mask.any(axis=1)
        
        # Wyświetl wyniki
        results = df[mask]
        
        st.write(f"Znaleziono {len(results)} wyników:")
        st.dataframe(results)
        
        # Wykresy rozkładu wyników
        if st.checkbox("Pokaż rozkład wyników"):
            for col in selected_columns:
                st.subheader(f"Rozkład {col}")
                st.bar_chart(results[col].value_counts())

def main():
    
    env = dotenv_values(".env")
    
    if 'QDRANT_URL' in st.secrets:
        env['QDRANT_URL'] = st.secrets['QDRANT_URL']
    if 'QDRANT_API_KEY' in st.secrets:
        env['QDRANT_API_KEY'] = st.secrets['QDRANT_API_KEY']

    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'file_buffer' not in st.session_state:
        st.session_state.file_buffer = None
    if "user_input" not in st.session_state:
        st.session_state.user_input = ''
        user_input = st.session_state.user_input
    if "user_input_sentence" not in st.session_state:
        st.session_state.user_input_sentence = ''
        user_input_sentence = st.session_state.user_input_sentence

    st.sidebar.title("Load data")
    st.sidebar.write("Proper file format: id_product, category root, category, name, description.")
    
    # Path to the default Excel file
    default_file_path = "dane wszystkodlazwierzat.pl.xlsx"

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["xlsx", "csv"])

    if uploaded_file is not None:
        # Process uploaded file
        df = load_data()
    else:
        # Load default file if no file is uploaded
        df = pd.read_excel(default_file_path, engine='openpyxl')
        st.session_state.data = df
        st.sidebar.success(f"Successfully loaded default file: {default_file_path}")

    # Display data if available
    if df is not None:
        st.sidebar.write("Look for upload data:")
        st.sidebar.dataframe(df.head())
        
        # Informacje o danych
        st.sidebar.write("Info:")
        st.sidebar.write(f"Rows: {df.shape[0]}")
        st.sidebar.write(f"Columns: {df.shape[1]}")
        st.sidebar.write("Columns name:", df.columns.tolist())

        # Opcjonalnie: statystyki kolumn
        if st.sidebar.checkbox("Statistic of columns"):
            st.sidebar.write(df.describe())

    st.title("Text & Embeddings searching")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Main", "Advanced"])
    
    with tab1:
        # Common search bar
        user_input = st.text_input("Search", st.session_state.user_input, key="common_search")
        if st.button("Confirm", key="common_search_button"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("OpenAI ada 002 Results")
                vector_results, text_results = open_AI_search(
                    query_text=user_input,
                    collection_name=global_variables.QDRANT_COLLECTION_NAME_AI
                )
                st.write("Vector search result:")
                for result in vector_results:
                    st.write('Name:', result.payload["name"], 
                             'ID:', result.payload["id_product"], 
                             'Score:', round(result.score, 3))

                st.write("\nText search result:")
                for result in text_results:
                    st.write('Name:', result.payload["name"], 
                             'ID:', result.payload["id_product"])

            with col2:
                st.subheader("Sentence Transformers Results")
                vector_results, text_results = sentence_search(
                    query_text=user_input,
                    collection_name=global_variables.QDRANT_COLLECTION_NAME_SENTENCE
                )
                st.write("Vector search result:")
                for result in vector_results:
                    st.write('Name:', result.payload["name"], 
                             'ID:', result.payload["id_product"], 
                             'Score:', round(result.score, 3))

                st.write("\nText search result:")
                for result in text_results:
                    st.write('Name:', result.payload["name"], 
                             'ID:', result.payload["id_product"])

    with tab2:
        st.header("Advanced Options")
        test_searching()
        
        st.subheader("Vector Database Operations")

        st.subheader("Sentence transformers database")
        if st.button("ST data -  Load data to Qdrant"):
            sentence_transtormer_load_data()
        if st.button("Look Sententence transformers database"):
            get_collection_info(QDRANT_COLLECTION_NAME_SENTENCE)
        st.subheader("Ada-002 Open AI database")
        if st.button("ADA data -  Load data to Qdrant"):
            openAI_load_data()

if __name__ == "__main__":
    main()