import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, MatchText
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, SearchRequest
from typing import Union, List
from sentence_transformers import SentenceTransformer
from dotenv import dotenv_values

# only local deploy
#from dotenv import load_dotenv
#from langfuse.openai import OpenAI

#only 4 streamlit deploy
from openai import OpenAI
# only 4 local deploy

# load_dotenv()
# env = dotenv_values(".env")

#streamlit deploy
env = dotenv_values(".env")
key=st.session_state.get("openai_api_key")
### Secrets using Streamlit Cloud Mechanism
# https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
if 'QDRANT_URL' in st.secrets:
    env['QDRANT_URL'] = st.secrets['QDRANT_URL']
if 'QDRANT_API_KEY' in st.secrets:
    env['QDRANT_API_KEY'] = st.secrets['QDRANT_API_KEY']


# Initialisation
#streamlit deploy
# qdrant_client = QdrantClient(url='QDRANT_URL', api_key='QDRANT_API_KEY')
# local deploy
#qdrant_client = QdrantClient(url=os.getenv('QDRANT_URL'), api_key=os.getenv('QDRANT_API_KEY'))
QDRANT_COLLECTION_NAME_AI = "shop_data_openAI"
QDRANT_COLLECTION_NAME_SENTENCE = "shop_data_sentence_transformer"
EMBEDDING_DIM = 1536
EMBEDDING_MODEL = "text-embedding-ada-002" #OpenAI used model


# Session state inistialisation
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

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
    url=env["QDRANT_URL"], 
    api_key=env["QDRANT_API_KEY"],
)

def process_uploaded_file(uploaded_file, file_type):
    """
    Przetwarza wczytany plik do BytesIO i DataFrame z pełnym zakresem danych
    """
    try:
        # Zapisz plik do BytesIO
        bytes_data = BytesIO(uploaded_file.read())
        st.session_state.file_buffer = bytes_data
        
        # Wczytaj dane do DataFrame
        if file_type == "xlsx":
            # Użyj parametrów do wczytania pełnego arkusza
            df = pd.read_excel(
                bytes_data, 
                engine='openpyxl',  # Nowszy silnik dla .xlsx
                header=0,  # Pierwszy wiersz jako nagłówek
                index_col=None,  # Nie używaj pierwszej kolumny jako indeksu
                usecols=None,  # Wczytaj wszystkie kolumny
                dtype=object  # Zachowaj oryginalne typy danych
            )
        else:  # csv
            encoding = st.session_state.get('encoding', 'utf-8')
            separator = st.session_state.get('separator', ',')
            df = pd.read_csv(
                bytes_data, 
                encoding=encoding, 
                sep=separator,
                header=0,
                index_col=False,
                dtype=object  # Zachowaj oryginalne typy danych
            )
        
        return df
    except Exception as e:
        st.error(f"Błąd podczas przetwarzania pliku: {str(e)}")
        return None

def load_data():
    """
    Funkcja do wczytywania plików
    """
    option = st.selectbox(
        "Choose file format",
        ("Upload .xlsx", "Upload .csv"),
        key="file_format"
    )
    
    try:
        if option == "Upload .xlsx":
            uploaded_file = st.file_uploader(
                "Choose Excel file",
                type=["xlsx"],
                key="excel_uploader"
            )
            
            if uploaded_file is not None:
                # Wyświetl dostępne arkusze dla pliku Excel
                xls = pd.ExcelFile(uploaded_file)
                sheet_names = xls.sheet_names
                
                # Wybór arkusza
                selected_sheet = st.selectbox(
                    "Choose sheet",
                    sheet_names,
                    key="sheet_selector"
                )
                
                # Resetuj pozycję pliku
                uploaded_file.seek(0)
                
                # Wczytaj wybrany arkusz
                df = pd.read_excel(
                    uploaded_file, 
                    sheet_name=selected_sheet,
                    engine='openpyxl',
                    header=0,
                    index_col=None,
                    dtype=object  # Zachowaj oryginalne typy danych
                )
                
                if df is not None:
                    st.session_state.data = df
                    st.success(f"Succesfuly load Excel sheet: {selected_sheet}")
                    return df
                    
        elif option == "Upload .csv":
            uploaded_file = st.file_uploader(
                "Choose CSV file",
                type=["csv"],
                key="csv_uploader"
            )
            
            if uploaded_file is not None:
                # Opcje dla CSV
                st.session_state.encoding = st.selectbox(
                    "Choose encoding",
                    options=["utf-8", "cp1250", "iso-8859-1"],
                    key="encoding"
                )
                
                st.session_state.separator = st.selectbox(
                    "Choose separator",
                    options=[",", ";", "|", "\t"],
                    key="separator"
                )
                
                df = process_uploaded_file(uploaded_file, "csv")
                if df is not None:
                    st.session_state.data = df
                    st.success(f"Succesfuly load file: {uploaded_file.name}")
                    return df
                    
        if st.session_state.data is None:
            st.info("Choose file to load")
            
    except Exception as e:
        st.error(f"Load error: {str(e)}")
        return None

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

def reset_collection(COLLECTION_NAME, DIM):
    try:
        qdrant_client = get_qdrant_client()
        collection_name = COLLECTION_NAME
        # Usuń kolekcję, jeśli istnieje
        if collection_name in [col.name for col in qdrant_client.get_collections().collections]:
            qdrant_client.delete_collection(collection_name=collection_name)
            print(f"Collection '{collection_name}' was droped.")

        # Utwórz nową kolekcję
        if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
            print("Tworzę kolekcję")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
    )
        
    except Exception as e:
        print("Connection error:", e)

def get_normalized_embedding(
    text: Union[str, List[str]], 
    model_name: str = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
) -> np.ndarray:
    """
    Generuje znormalizowany wektor embedingu dla podanego tekstu używając określonego modelu.
    
    Args:
        text: Tekst lub lista tekstów do przetworzenia
        model_name: Nazwa modelu sentence-transformers do użycia
        
    Returns:
        Znormalizowany wektor embedingu lub macierz wektorów dla listy tekstów
    """
    # Załaduj model
    model = SentenceTransformer(model_name)
    
    # Wygeneruj embedding
    embedding = model.encode(text)
    
    # Konwertuj na numpy array jeśli nie jest
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding)
    
    # Normalizacja L2 (długość wektora = 1)
    if embedding.ndim == 1:
        # Dla pojedynczego wektora
        norm = np.linalg.norm(embedding)
        normalized_embedding = embedding / norm if norm > 0 else embedding
    else:
        # Dla macierzy wektorów
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        normalized_embedding = np.divide(embedding, norm, where=norm > 0)
    
    return normalized_embedding

def openAI_load_data():
    # Sprawdź czy dane są załadowane
    # if st.session_state.data is None:
    #     st.warning("Load data first at tab 'Load data'")
    #     return
    if st.button("Vector db new data load"):
    # usunięcie \n
    #    st.session_state.data=st.session_state.data.map(lambda x: x.replace('\n', ' ') if isinstance(x, str) else x)
    # przeształcenie data na listę
    #    st.session_state.data=st.session_state.data.to_dict(orient='records')
    #st.write(st.session_state.data) #temporaty list print ok?
    # collection reset        
        reset_collection(QDRANT_COLLECTION_NAME_AI, 1536)
    # load data to collection
        qdrant_client = get_qdrant_client()
        qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME_AI,
        points=[
        PointStruct(
            id=idx,
            vector=get_embedding_ai(f'{row["category root"]} {row["category"]} {row["name"]} {row["description"]}'),  # Generowanie wektora tylko dla nazwy
            payload={
                "id_product": row["id_product"], 
                "name": row["name"]
            }  # Payload jako słownik
        )
        for idx, row in st.session_state.data.iterrows()
        ]
        )
        st.toast("Data was succesufy loaded to Qdrant collection '{QDRANT_COLLECTION_NAME_AI}'", icon="🎉")
    return 
    # st.write(f"Data was succesufy loaded to Qdrant '{QDRANT_COLLECTION_NAME_AI}'.")

def sentence_transtormer_load_data():
    # Sprawdź czy dane są załadowane
    # if st.session_state.data is None:
    #     st.warning("Load data first at tab 'Load data'")
    #     return
    if st.button("Vector db new data load "):
    # usunięcie \n
    #    st.session_state.data=st.session_state.data.map(lambda x: x.replace('\n', ' ') if isinstance(x, str) else x)
    # przeształcenie data na listę
    #    st.session_state.data=st.session_state.data.to_dict(orient='records')
    #st.write(st.session_state.data) #temporaty list print ok?
    # collection reset        
        qdrant_client = get_qdrant_client()
        reset_collection(QDRANT_COLLECTION_NAME_SENTENCE, 768)
    # load data to collection
        qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME_SENTENCE,
        points=[
        PointStruct(
            id=idx,
            vector=get_normalized_embedding(f'{row["category root"]} {row["category"]} {row["name"]} {row["description"]}'),  
            payload={
                "id_product": row["id_product"], 
                "name": row["name"]
            }  # Payload jako słownik
        )
        for idx, row in st.session_state.data.iterrows()
        ]
        )
        st.toast("Data was succesufy loaded to Qdrant collection '{QDRANT_COLLECTION_NAME_AI}'", icon="🎉")
    return 
#st.write(f"Data was succesufy loaded to Qdrant '{QDRANT_COLLECTION_NAME_SENTENCE}'.")

def get_openai_client():
    #local deploy
    #return OpenAI(api_key=env["OPENAI_API_KEY"])
    #streamlit deploy
    return OpenAI(api_key=key)

def get_embedding_ai(text):
    openai_client = get_openai_client()
    result = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
        #dimensions=EMBEDDING_DIM,
    )

    return result.data[0].embedding  

def open_AI_search(query_text: str, collection_name: str, limit: int = 10, score_threshold: float = 0.2):
    """
    Performs both vector similarity search and text search in Qdrant collection.
    
    Args:
        query_text (str): Text to search for
        collection_name (str): Name of the Qdrant collection
        limit (int): Maximum number of results to return
        score_threshold (float): Minimum similarity score threshold
        
    Returns:
        tuple: Two lists of results - vector search results and text search results
    """
   
    # Vector similarity search
    qdrant_client = get_qdrant_client()
    vector_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=get_embedding_ai(query_text),
        limit=limit,
        score_threshold=score_threshold
    )
    
    # Text search using payload field
    text_search_results = qdrant_client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="name",
                    match=MatchText(
                        text=query_text
                    )
                )
            ]
        ),
        limit=limit
    )[0]  # [0] because scroll returns tuple (results, next_page_offset)
    
    return vector_results, text_search_results

def sentence_search(query_text: str, collection_name: str, limit: int = 10, score_threshold: float = 0.2):
    """
    Performs both vector similarity search and text search in Qdrant collection.
    
    Args:
        query_text (str): Text to search for
        collection_name (str): Name of the Qdrant collection
        limit (int): Maximum number of results to return
        score_threshold (float): Minimum similarity score threshold
        
    Returns:
        tuple: Two lists of results - vector search results and text search results
    """
   
    # Vector similarity search
    qdrant_client = get_qdrant_client()
    vector_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=get_normalized_embedding(query_text),
        limit=limit,
        score_threshold=score_threshold
    )
    
    # Text search using payload field
    text_search_results = qdrant_client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="name",
                    match=MatchText(
                        text=query_text
                    )
                )
            ]
        ),
        limit=limit
    )[0]  # [0] because scroll returns tuple (results, next_page_offset)
    
    return vector_results, text_search_results

def get_collection_info(collection_name):

    
    try:
        # Pobieranie informacji o kolekcji
        qdrant_client = get_qdrant_client()
        collection_info = qdrant_client.get_collection(collection_name)
        
        return {
            "Name": collection_name,
            #"Vectors Count": collection_info.vectors_count,
            #"Indexed Vectors Count": collection_info.indexed_vectors_count,
            "Points Count": collection_info.points_count,
            #"Deletion Enabled": collection_info.config.params.vectors.on_disk,
            "Vector Size": collection_info.config.params.vectors.size,
        }
    except Exception as e:
        st.error(f"Error during get Qudrant collection info: {e}")
        return None

def main():
    st.title("Text & Context searching tests")
    
    # Zakładki
    tab1, tab2, tab3, tab4 = st.tabs(["Load data", "Text searching", "OpenAI ada 002", "Sentence transformers"])
    
    with tab1:
        st.write("Proper file format: id_product, category root, category, name, description.")
        
        # Path to the Excel file
        file_path = "example_file.xlsx"

        # Read the file in binary mode
        with open(file_path, "rb") as file:
            file_data = file.read()

        # Display the download button
        st.download_button(
            label="Download example Excel File",
            data=file_data,
            file_name="emample_file.xls",
            mime="application/vnd.ms-excel"
            )
            

        # Przycisk do resetowania
        if st.button("Reset data"):
            st.session_state.data = None
            st.session_state.file_buffer = None
            #st.experimental_rerun()
        
        # Wczytaj dane
        df = load_data()
        
        # Wyświetl dane jeśli są dostępne
        if df is not None:
            st.write("Look for upload data:")
            st.dataframe(df.head())
            
            # Informacje o danych
            st.write("Info:")
            st.write(f"Rows: {df.shape[0]}")
            st.write(f"Columns: {df.shape[1]}")
            st.write("Columns name:", df.columns.tolist())

            # Opcjonalnie: statystyki kolumn
            if st.checkbox("Statistic of columns"):
                st.write(df.describe())
    
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
                collection_name=QDRANT_COLLECTION_NAME_AI
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
                info = get_collection_info(QDRANT_COLLECTION_NAME_AI)
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
                collection_name=QDRANT_COLLECTION_NAME_SENTENCE)
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
                info=get_collection_info(QDRANT_COLLECTION_NAME_SENTENCE)
                if info:
                    st.subheader("Szczegóły kolekcji")
                    for key, value in info.items():
                        st.write(f"{key}: {value}")
        with tab43:
            st.write("Load data to Qdrant")
            sentence_transtormer_load_data()  
            
            
        
        



if __name__ == "__main__":
    main()