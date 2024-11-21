import streamlit as st
import pandas as pd
from io import BytesIO

# Inicjalizacja session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'file_buffer' not in st.session_state:
    st.session_state.file_buffer = None

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
                # Wyświetl dostępne arkusze dla pliku Excel
                xls = pd.ExcelFile(uploaded_file)
                sheet_names = xls.sheet_names
                
                # Wybór arkusza
                selected_sheet = st.selectbox(
                    "Wybierz arkusz",
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
                    st.success(f"Pomyślnie wczytano arkusz: {selected_sheet}")
                    return df
                    
        elif option == "Upload .csv":
            uploaded_file = st.file_uploader(
                "Wybierz plik CSV",
                type=["csv"],
                key="csv_uploader"
            )
            
            if uploaded_file is not None:
                # Opcje dla CSV
                st.session_state.encoding = st.selectbox(
                    "Wybierz kodowanie pliku",
                    options=["utf-8", "cp1250", "iso-8859-1"],
                    key="encoding"
                )
                
                st.session_state.separator = st.selectbox(
                    "Wybierz separator",
                    options=[",", ";", "|", "\t"],
                    key="separator"
                )
                
                df = process_uploaded_file(uploaded_file, "csv")
                if df is not None:
                    st.session_state.data = df
                    st.success(f"Pomyślnie wczytano plik: {uploaded_file.name}")
                    return df
                    
        if st.session_state.data is None:
            st.info("Proszę wybrać plik do wczytania")
            
    except Exception as e:
        st.error(f"Wystąpił błąd: {str(e)}")
        return None

def main():
    st.title("Wczytywanie pełnych danych z pliku")
    
    # Przycisk do resetowania
    if st.button("Resetuj dane"):
        st.session_state.data = None
        st.session_state.file_buffer = None
        st.experimental_rerun()
    
    # Wczytaj dane
    df = load_data()
    
    # Wyświetl dane jeśli są dostępne
    if df is not None:
        st.write("Podgląd wczytanych danych:")
        st.dataframe(df)
        
        # Informacje o danych
        st.write("Informacje o danych:")
        st.write(f"Liczba wierszy: {df.shape[0]}")
        st.write(f"Liczba kolumn: {df.shape[1]}")
        st.write("Nazwy kolumn:", df.columns.tolist())

        # Opcjonalnie: statystyki kolumn
        if st.checkbox("Pokaż statystyki kolumn"):
            st.write(df.describe())

if __name__ == "__main__":
    main()