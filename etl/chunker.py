from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path

_similarity_model = None

def get_similarity_model():
    global _similarity_model
    if _similarity_model is None:
        _similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _similarity_model


def extract_metadata_from_filename(filename: str) -> dict:

    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) >= 2:
        company_name = " ".join(parts[:-1]).title()
        year_str = parts[-1]
        year = int(year_str) if year_str.isdigit() else 2025
    else:
        company_name = stem.title()
        year = 2025

    return {
        "company_name": company_name,
        "document_year": year,
        "source_file": filename,
        "document_type": "10-K",
    }


def semantic_chunk(text: str, metadata: dict, chunk_size: int = 800, overlap: int = 100) -> list[dict]:

    # Step 1: Initial split on structural boundaries
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n\n", "\n\n", "\n", ". ", " "],
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    raw_chunks = splitter.split_text(text)

    if not raw_chunks:
        return []

    # Step 2: Compute embeddings for similarity check
    model = get_similarity_model()
    embeddings = model.encode(raw_chunks, show_progress_bar=False)

    # Step 3: Merge semantically similar adjacent chunks
    merged_chunks = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        cos_sim = np.dot(embeddings[i-1], embeddings[i]) / (
            np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i]) + 1e-9
        )
        candidate_merge = merged_chunks[-1] + " " + raw_chunks[i]
        if cos_sim > 0.85 and len(candidate_merge) <= chunk_size * 1.5:
            merged_chunks[-1] = candidate_merge
        else:
            merged_chunks.append(raw_chunks[i])

    # Step 4: Attach metadata to every chunk
    result = []
    for idx, chunk_text in enumerate(merged_chunks):
        if len(chunk_text.strip()) < 50:
            continue
        result.append({
            "text": chunk_text.strip(),
            "metadata": {
                **metadata,
                "chunk_index": idx,
                "chunk_total": len(merged_chunks),
            }
        })

    return result