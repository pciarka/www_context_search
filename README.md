# Text & Embeddings Searching Application

This project is a Streamlit-based web application that shows capabilities of semantic search.

## Features

- Upload and process Excel (.xlsx) or CSV files.
- Display and explore uploaded data.
- Perform text searches with exact, partial, and regex matching.
- Perform embedding searches using OpenAI's ada-002 model and Sentence Transformers.
- Load data into Qdrant collections for vector similarity search.
- Display information about Qdrant collections.


## Usage

1. Open the [application](https://embedding-search.streamlit.app) in your web browser.
2. Use the sidebar to upload an Excel or CSV file(or use preloaded file).
3. Explore the uploaded data in the sidebar.
4. Use the main interface to perform text and embedding searches.
5. Load data into Qdrant collections and perform vector similarity searches.

## Files

- [app.py](http://_vscodecontentref_/8): Main application file.
- [app_by_me.py](http://_vscodecontentref_/9): Alternative application file with similar functionality.
- `app — kopia ok wczytywanie danych do df.py`: Another alternative application file.
- [data_manipulation.py](http://_vscodecontentref_/10): Functions for processing uploaded files and generating embeddings.
- [global_variables.py](http://_vscodecontentref_/11): Global variables for Qdrant collection names and embedding model configurations.
- [qdrant_communication.py](http://_vscodecontentref_/12): Functions for interacting with Qdrant and performing searches.

## License

This project is licensed under the MIT License. See the LICENSE file for details.