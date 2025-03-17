# Użyj oficjalnego obrazu Pythona
FROM python:3.11-slim

# Ustaw katalog roboczy w kontenerze
WORKDIR /app

# Skopiuj pliki aplikacji
COPY . /app

# Zainstaluj zależności
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

# Określ domyślną komendę startową
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]