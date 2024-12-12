from dotenv import dotenv_values
import streamlit as st

from qdrant_communication import get_collection_info, sentence_search, sentence_transtormer_load_data, openAI_load_data, open_AI_search
from data_manipulation import load_data
import global_variables

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
    
    # Path to the Excel file
    file_path = "example_file.xlsx"

    # Read the file in binary mode
    with open(file_path, "rb") as file:
        file_data = file.read()

    # Display the download button
    st.sidebar.download_button(
        label="Download example Excel File",
        data=file_data,
        file_name="example_file.xlsx",
        mime="application/vnd.ms-excel"
    )

    # Przycisk do resetowania
    if st.sidebar.button("Reset data"):
        st.session_state.data = None
        st.session_state.file_buffer = None
        #st.experimental_rerun()
    
    # Wczytaj dane
    df = load_data()
    
    # Wyświetl dane jeśli są dostępne
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
    
    # Zakładki
    tab2, tab3, tab4 = st.tabs(["Text searching", "OpenAI ada 002", "Sentence transformers"])
    
    with tab2:
        st.header("Text searching")
        test_searching()

    with tab3:
        st.write("OpenAI ada 002 model - paid commercial model")
        tab31, tab32, tab33 = st.tabs(["Searching", "Current Qdrant data", "Load data to Qdrant"])
        with tab31:
            user_input = st.text_input("Search", st.session_state.user_input, key=f"input_2")
            if st.button("Confirm", key=f"input_3"):
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
        with tab32:
            st.write("Current Qdrant data")
            if st.button("Info about current Qdrant collecion "):
                info = get_collection_info(global_variables.QDRANT_COLLECTION_NAME_AI)
                if info:
                    # st.subheader("Szczegóły kolekcji")
                    for key, value in info.items():
                        st.write(f"{key}: {value}")

        with tab33:
            st.write("Load data to Qdrant")
            openAI_load_data()
    with tab4:
        st.write("multi qa mpnet base dot v1 - open soucre python sentence transformer library")
        tab41, tab42, tab43 = st.tabs(["Searching", "Current Qdrant data", "Load data to Qdrant"])
        with tab41:
            user_input = st.text_input("Search", st.session_state.user_input, key=f"input_10")
            if st.button("Zatwierdź ", key=f"input_11"):
                vector_results, text_results = sentence_search(
                query_text=user_input,
                collection_name=global_variables.QDRANT_COLLECTION_NAME_SENTENCE)
                st.write("Wyniki wyszukiwania wektorowego:")
                for result in vector_results:
                    st.write('Name:', result.payload["name"], 
                    'ID:', result.payload["id_product"], 
                    'Score:', round(result.score, 3))

                st.write("\nWyniki wyszukiwania tekstowego:")
                for result in text_results:
                    st.write('Name:', result.payload["name"], 
                    'ID:', result.payload["id_product"])

        with tab42:
            st.write("Current Qdrant data")
            if st.button("Info about current Qdrant collecion"):
                info=get_collection_info(global_variables.QDRANT_COLLECTION_NAME_SENTENCE)
                if info:
                    #st.subheader("Szczegóły kolekcji")
                    for key, value in info.items():
                        st.write(f"{key}: {value}")
        with tab43:
            st.write("Load data to Qdrant")
            sentence_transtormer_load_data()  

if __name__ == "__main__":
    main()