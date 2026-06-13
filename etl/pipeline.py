import os
# import uuid
import hashlib
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.models.models import KeywordIndexParams, IntegerIndexParams

try:
    from etl.chunker import semantic_chunk, extract_metadata_from_filename
except ModuleNotFoundError:
    from chunker import semantic_chunk, extract_metadata_from_filename

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def create_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    # Ensure payload indexes exist for metadata filters used by the agent
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="company_name",
            field_schema=KeywordIndexParams(),
            wait=True,
        )
        print("Ensured payload index: company_name (keyword)")
    except Exception:
        # ignore if index already exists or Qdrant reports non-fatal error
        pass

    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="document_year",
            field_schema=IntegerIndexParams(),
            wait=True,
        )
        print("Ensured payload index: document_year (integer)")
    except Exception:
        pass

def embed_and_insert(chunks: list[dict], model: SentenceTransformer, client: QdrantClient):
    if not chunks:
        return

    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=False, batch_size=32)

    points = [
        PointStruct(
            id = hashlib.md5(f"{chunk['metadata']['source_file']}_{chunk['metadata']['chunk_index']}".encode()).hexdigest(),
            vector=vector.tolist(),
            payload={"text": chunk["text"], **chunk["metadata"]}
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    # Upsert in batches of 100
    for i in range(0, len(points), 100):
        client.upsert(collection_name=COLLECTION_NAME, points=points[i:i+100])


def run_pipeline(documents_dir: str = "data/cleaned"):
 
    print("=" * 60)
    print("Starting Vector ETL Pipeline")
    print("=" * 60)

    print(f"\n[1/4] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"\n[2/4] Connecting to Qdrant at {QDRANT_URL}")
    client = get_qdrant_client()
    create_collection(client)

    doc_files = list(Path(documents_dir).glob("*.txt"))
    print(f"\n[3/4] Processing {len(doc_files)} cleaned documents...")

    total_chunks = 0
    for doc_path in tqdm(doc_files, desc="Documents"):
        cleaned_text = doc_path.read_text(encoding="utf-8", errors="ignore")
        metadata = extract_metadata_from_filename(doc_path.name)
        chunks = semantic_chunk(cleaned_text, metadata)
        embed_and_insert(chunks, model, client)
        total_chunks += len(chunks)

    print(f"\n[4/4] Done! Inserted {total_chunks} chunks across {len(doc_files)} documents.")
    info = client.get_collection(COLLECTION_NAME)
    print(f"Qdrant collection '{COLLECTION_NAME}' now has {info.points_count} vectors.")


if __name__ == "__main__":
    run_pipeline()