import os
from dotenv import load_dotenv
import pandas as pd
import uvicorn
import requests
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer



load_dotenv() # read local .env file
qdrant = encoder = vdb_name = model_name = base_url = None  # declare globals



@asynccontextmanager
async def lifespan(app: FastAPI):
    global qdrant, encoder, vdb_name, model_name, base_url

    load_dotenv()
    base_url = os.getenv("OPENAI_BASE_URL")
    vdb_name = os.getenv("VECTORDB_NAME")
    source_data = os.getenv("SOURCE_DATA_LOC")
    vector_col = os.getenv("VECTORIZE_COLUMN")
    model_name = os.getenv("OPENAI_MODEL")

    print("INFO: base_url ", base_url)
    print("INFO: vdb_name ", vdb_name)
    print("INFO: source_data ", source_data)
    print("INFO: vector_col ", vector_col)
    print("INFO: model_name ", model_name)

    qdrant, encoder = init_vectorDB(source_data, vector_col, vdb_name)
    print("INFO: Vector DB Initialized")

    yield  # control goes to FastAPI app here



app = FastAPI(lifespan=lifespan)



def init_vectorDB(source_data, vector_col, vdb_name):
    global qdrant, encoder
    # import source data
    df = pd.read_csv(source_data)
    df = df.dropna(axis=0) # remove any row with NaN values as it blows up serialization
    data = df.to_dict('records')
    # model to create embeddings
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    # create the vector database client
    qdrant = QdrantClient(":memory:") # Create in-memory Qdrant instance
    # create collection to store data
    qdrant.recreate_collection(
        collection_name=vdb_name,
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(), # Vector size is defined by used model
            distance=models.Distance.COSINE
        )
    )
    # vectorize
    qdrant.upload_points(
        collection_name=vdb_name,
        points=[
            models.PointStruct(
                id=idx,
                vector=encoder.encode(doc[vector_col]).tolist(),
                payload=doc
            ) for idx, doc in enumerate(data)
        ]
    )

    return qdrant, encoder



class Body(BaseModel):
    query: str


@app.get('/')
def root():
    return RedirectResponse(url='/docs', status_code=301)


@app.post('/ask')
def ask(body: Body):
    """
    Use the query parameter to interact with the 
    llamafile based LLM running in background for Retrieval Augmented Generation.
    """
    search_result = search(body.query)
    chat_bot_response = assistant(body.query, search_result)
    return {'response': chat_bot_response}



def search(query):
    """
    Send the query to a llamafile based LLM and return the top result
    """
    hits = qdrant.search(
    collection_name=vdb_name,
    query_vector=encoder.encode(query).tolist(),
    limit=3
    )
    # define a variable to hold the search results
    result = [hit.payload for hit in hits]
    print(result)
    return result


def assistant(query, context):
    messages=[
        # Set the system characteristics for this chat bot
        {"role": "system", "content": "You are chatbot, a movie specialist. Your top priority is to help guide users into selecting amazing movies and guide them with their requests. Use only the movies provided here nothing else."},

        # Set the query so that the chatbot can respond to it
        {"role": "user", "content": query},

        # Add the context from the vector search results so that the chatbot can use
        # it as part of the response for an augmented context
        {"role": "assistant", "content": str(context)}
    ]

    response = requests.post(
        base_url,   # port/url on which LLM is being served
        headers={"Content-Type": "application/json"},
        json={
            "model": model_name,    # whatever model name your llamafile serves
            "messages": messages
        }
    )
    return response.json()["choices"][0]["message"]["content"]



if __name__ == "__main__":
    print("INFO: Starting FastAPI Application...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000)