import os
from pinecone import Pinecone, ServerlessSpec
import time
import PyPDF2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "ricky"

pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    print("Creating Pinecone index...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    time.sleep(10)

index = pc.Index(INDEX_NAME)

embed_model = SentenceTransformer("intfloat/multilingual-e5-large", trust_remote_code=True)

def get_embedding(text):
    """Generate embedding for given text."""
    return embed_model.encode(text).tolist()

def upsert_screenplay_vectors(screenplay_id, text, genre):
    """Store screenplay embeddings in Pinecone."""
    try:
        vector = get_embedding(text)
        index.upsert([
            {"id": screenplay_id, "values": vector, "metadata": {"genre": genre, "content": text}}
        ])
    except Exception as e:
        print(f"Error upserting to Pinecone: {e}")

def retrieve_relevant_data(keywords, genre):
    """Retrieve relevant screenplay context from Pinecone."""
    try:
        query_vector = get_embedding(" ".join(keywords))
        results = index.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True,
            filter={"genre": {"$eq": genre}},
        )
        retrieved_texts = [match["metadata"].get("content", "") for match in results.get("matches", [])]
        return " ".join(retrieved_texts) if retrieved_texts else ""
    except Exception as e:
        print(f"Error retrieving from Pinecone: {e}")
        return ""

def extract_text(file):
    """Extract text from PDF or TXT file."""
    if file.filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    else:
        text = file.read().decode("utf-8")
    return text

def process_and_store(file, genre):
    """Extract text, generate embeddings, and upsert to Pinecone."""
    text = extract_text(file)
    if not text:
        return "No text found in file."

    embedding = get_embedding(text)
    file_id = str(hash(text))

    index.upsert([
        {"id": file_id, "values": embedding, "metadata": {"genre": genre, "content": text}}
    ])
    return "File uploaded and stored successfully!"
