from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, MatchText
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, SearchRequest
import streamlit as st
from openai import OpenAI
from dotenv import dotenv_values
import pytz
from datetime import datetime

from data_manipulation import get_normalized_embedding
import global_variables

@st.cache_resource
def get_qdrant_client():
    env = dotenv_values(".env")
    return QdrantClient(
    url=env["QDRANT_URL"], 
    api_key=env["QDRANT_API_KEY"],
)

def reset_collection(COLLECTION_NAME, DIM):
    try:
        qdrant_client = get_qdrant_client()
        collection_name = COLLECTION_NAME
        # Delete collection if exists
        if collection_name in [col.name for col in qdrant_client.get_collections().collections]:
            qdrant_client.delete_collection(collection_name=collection_name)
            print(f"Collection '{collection_name}' was droped.")

        # Create collection
        if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
            print("Tworzę kolekcję")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
    )
        
    except Exception as e:
        print("Connection error:", e)

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
        # Określenie strefy czasowej, np. Europe/Warsaw
        local_timezone = pytz.timezone("Europe/Warsaw")

        # Pobranie aktualnej daty i godziny w UTC
        utc_time = datetime.now(pytz.utc)

        # Konwersja na czas lokalny
        local_time = utc_time.astimezone(local_timezone)

        return {
            "Time of get info": local_time.strftime("%Y-%m-%d %H:%M:%S"),
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
        reset_collection(global_variables.QDRANT_COLLECTION_NAME_SENTENCE, 768)
    # load data to collection
        qdrant_client.upsert(
        collection_name=global_variables.QDRANT_COLLECTION_NAME_SENTENCE,
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
        reset_collection(global_variables.QDRANT_COLLECTION_NAME_AI, 1536)
    # load data to collection
        qdrant_client = get_qdrant_client()
        qdrant_client.upsert(
        collection_name=global_variables.QDRANT_COLLECTION_NAME_AI,
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
        st.toast("Data was succesufy loaded to Qdrant collection '{global_variables.QDRANT_COLLECTION_NAME_AI}'", icon="🎉")
    return 

###
### OpenAI functions
###

def open_AI_search(query_text: str, collection_name: str, limit: int = 10, score_threshold: float = 0.2):

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

def get_embedding_ai(text):
    openai_client = OpenAI(api_key=st.session_state.get("openai_api_key"))
    result = openai_client.embeddings.create(
        input=[text],
        model=global_variables.EMBEDDING_MODEL,
    )

    return result.data[0].embedding  
