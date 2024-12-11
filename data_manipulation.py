import streamlit as st
import pandas as pd
import numpy as np
from typing import Union, List
from sentence_transformers import SentenceTransformer
from io import BytesIO



#create data frame from uploaded file
def process_uploaded_file(uploaded_file, file_type):

    try:
        
        bytes_data = BytesIO(uploaded_file.read())
        st.session_state.file_buffer = bytes_data
        
        #xlsx
        if file_type == "xlsx":
            
            df = pd.read_excel(
                bytes_data, 
                engine='openpyxl',
                header=0,  
                index_col=None, 
                usecols=None,
                dtype=object  # Keep original data types
            )
        #csv
        else: 
            encoding = st.session_state.get('encoding', 'utf-8')
            separator = st.session_state.get('separator', ',')
            df = pd.read_csv(
                bytes_data, 
                encoding=encoding, 
                sep=separator,
                header=0,
                index_col=False,
                dtype=object  # Keep original data types
            )
        
        return df
    
    
    except Exception as e:
        st.error(f"Error while processing file: {str(e)}")
        return None

# Load file with streamlit interface
def load_data():
    
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
                # Show available sheets
                xls = pd.ExcelFile(uploaded_file)
                sheet_names = xls.sheet_names
                
                # Sheet selector
                selected_sheet = st.selectbox(
                    "Choose sheet",
                    sheet_names,
                    key="sheet_selector"
                )
                
                # Possition reset
                uploaded_file.seek(0)
                
                # xls encoding and separator
                df = pd.read_excel(
                    uploaded_file, 
                    sheet_name=selected_sheet,
                    engine='openpyxl',
                    header=0,
                    index_col=None,
                    dtype=object  #Keep original data types
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
                # csv encoding and separator
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