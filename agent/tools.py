import os
import re
import sqlite3
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv
from langchain.tools import tool
from pydantic import BaseModel, Field

import instructor
from google import genai as google_genai
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

load_dotenv()


# ─── Config ──────────────────────────────────────────────────────────────────

_collection = os.getenv("QDRANT_COLLECTION", "documents")
_db_path = os.getenv("SQLITE_DB_PATH", "data/financials.db")
gemini_model = os.getenv("GEMINI_EXTRACTION_MODEL", "gemini-2.5-flash-lite")


# ─── Lazy Shared Resources ───────────────────────────────────────────────────
# Do NOT initialize these at import time.
# If they fail globally, the whole agent crashes before error handling can help.

_embed_model: Optional[SentenceTransformer] = None
_qdrant_client: Optional[QdrantClient] = None
_instructor_client = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model

    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    return _embed_model


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client

    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY") or None,
            timeout=10,
        )

    return _qdrant_client


def _get_instructor_client():
    global _instructor_client

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")

    if _instructor_client is None:
        genai_client = google_genai.Client(api_key=api_key)
        _instructor_client = instructor.from_genai(
            client=genai_client,
            mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
        )

    return _instructor_client


# ─── Tool 1: execute_sql ─────────────────────────────────────────────────────

def _is_read_only_sql(query: str) -> bool:
    """
    Allows only SELECT/WITH queries.

    This prevents the LLM from accidentally mutating or deleting your database.
    """
    cleaned = query.strip().rstrip(";").strip()

    if not cleaned:
        return False

    # Disallow multiple statements.
    if ";" in cleaned:
        return False

    # Only SELECT or WITH CTE queries.
    if not re.match(r"^(select|with)\b", cleaned, flags=re.IGNORECASE):
        return False

    forbidden = re.compile(
        r"\b("
        r"insert|update|delete|drop|alter|create|replace|truncate|"
        r"attach|detach|pragma|vacuum|reindex"
        r")\b",
        flags=re.IGNORECASE,
    )

    return forbidden.search(cleaned) is None


@tool
def execute_sql(query: str) -> str:
    """
    Execute a read-only SQL query against the company financials SQLite database.

    The database has one table: company_financials.

    Columns:
    - company_name TEXT
    - year INT
    - revenue_billions REAL
    - net_income_billions REAL
    - total_assets_billions REAL
    - employees_thousands REAL
    - rd_expense_billions REAL
    - operating_margin_pct REAL

    Use this for precise numerical lookups like:
    - What was Apple's revenue in 2025?
    - Which company had the highest net income in 2025?
    - Compare R&D spending across companies.
    """
    query = query.strip()

    if not _is_read_only_sql(query):
        return (
            "SQL TOOL ERROR [UnsafeQuery]: Only read-only SELECT/WITH queries are allowed. "
            "Rewrite the request as a SELECT query against the company_financials table."
        )

    if not Path(_db_path).exists():
        return (
            f"SQL TOOL ERROR [MissingDatabase]: SQLite database not found at path: {_db_path}. "
            "Check SQLITE_DB_PATH in your environment."
        )

    try:
        max_rows = 50

        with sqlite3.connect(_db_path, timeout=10) as conn:
            conn.row_factory = sqlite3.Row

            # Defense in depth: even if validation misses something,
            # SQLite will reject writes in this connection.
            conn.execute("PRAGMA query_only = ON")

            cursor = conn.cursor()
            cursor.execute(query)

            rows = cursor.fetchmany(max_rows + 1)

        if not rows:
            return "SQL query succeeded but returned no rows."

        truncated = len(rows) > max_rows
        rows = rows[:max_rows]

        columns = list(rows[0].keys())
        header = " | ".join(columns)
        separator = "-" * len(header)

        data_rows = [
            " | ".join(str(row[col]) for col in columns)
            for row in rows
        ]

        output = "\n".join([header, separator] + data_rows)

        if truncated:
            output += f"\n\n[Truncated: showing first {max_rows} rows.]"

        return output

    except sqlite3.Error as error:
        return (
            f"SQL TOOL ERROR [{type(error).__name__}]: {error}\n"
            f"Failed query: {query}\n"
            "Fix the SQL once using only the company_financials table and documented columns."
        )

    except Exception as error:
        return (
            f"SQL TOOL ERROR [{type(error).__name__}]: Unexpected SQL tool failure: {error}"
        )


# ─── Tool 2: search_vector_db ────────────────────────────────────────────────

class VectorSearchParams(BaseModel):
    """
    Structured search parameters extracted from a natural language query.
    """

    query_text: str = Field(
        description="The core semantic search query, cleaned of company/year filters."
    )

    company_name: Optional[str] = Field(
        default=None,
        description=(
            "Company name to filter by if mentioned, e.g. Apple, Microsoft, Amazon. "
            "None if not specified."
        ),
    )

    document_year: Optional[int] = Field(
        default=None,
        description="4-digit document year to filter by if mentioned. None if not specified.",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return.",
    )


_COMPANY_ALIASES = {
    "apple": "Apple",
    "microsoft": "Microsoft",
    "amazon": "Amazon",
    "google": "Google",
    "alphabet": "Alphabet",
    "meta": "Meta",
    "facebook": "Meta",
    "nvidia": "NVIDIA",
    "tesla": "Tesla",
}


def _fallback_extract_params(user_query: str) -> VectorSearchParams:
    """
    Simple fallback parser used when Gemini/Instructor extraction fails.

    This keeps the vector tool usable even if structured extraction is broken.
    """
    lowered = user_query.lower()

    document_year = None
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", user_query)
    if year_match:
        document_year = int(year_match.group(1))

    company_name = None
    matched_alias = None

    for alias, canonical_name in _COMPANY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            company_name = canonical_name
            matched_alias = alias
            break

    query_text = user_query

    if document_year:
        query_text = re.sub(rf"\b{document_year}\b", "", query_text)

    if matched_alias:
        query_text = re.sub(
            rf"\b{re.escape(matched_alias)}\b",
            "",
            query_text,
            flags=re.IGNORECASE,
        )

    query_text = re.sub(r"\s+", " ", query_text).strip()

    if not query_text:
        query_text = user_query

    return VectorSearchParams(
        query_text=query_text,
        company_name=company_name,
        document_year=document_year,
        top_k=5,
    )


def _extract_search_params(user_query: str) -> Tuple[VectorSearchParams, Optional[str]]:
    """
    Attempts Gemini/Instructor extraction first.
    Falls back to regex parsing if anything fails.
    """
    prompt = (
        "Extract search parameters from this query. "
        "Be precise with company names and years. "
        "The query_text should be the semantic search query WITHOUT company/year constraints.\n\n"
        f"Query: {user_query}"
    )

    try:
        client = _get_instructor_client()

        params = client.create(
            response_model=VectorSearchParams,
            messages=[{"role": "user", "content": prompt}],
            model=gemini_model,
        )

        if not params.query_text.strip():
            params.query_text = user_query

        return params, None

    except Exception as error:
        fallback_params = _fallback_extract_params(user_query)

        warning = (
            f"Structured extraction failed; used fallback parsing instead. "
            f"Reason: {type(error).__name__}: {error}"
        )

        return fallback_params, warning


def _build_qdrant_filter(params: VectorSearchParams) -> Optional[Filter]:
    must_conditions = []

    if params.company_name:
        must_conditions.append(
            FieldCondition(
                key="company_name",
                match=MatchValue(value=params.company_name),
            )
        )

    if params.document_year:
        must_conditions.append(
            FieldCondition(
                key="document_year",
                match=MatchValue(value=params.document_year),
            )
        )

    if not must_conditions:
        return None

    return Filter(must=must_conditions)


def _query_qdrant(
    client: QdrantClient,
    query_vector: list[float],
    qdrant_filter: Optional[Filter],
    limit: int,
):
    return client.query_points(
        collection_name=_collection,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=limit,
        with_payload=True,
    )


@tool
def search_vector_db(user_query: str) -> str:
    """
    Search the document vector database using semantic similarity.

    Use this for qualitative questions like:
    - What are Apple's main risk factors?
    - How does Amazon describe its cloud strategy?
    - What does Google say about competition in 2025?
    - What is Amazon's stance on cybersecurity?

    The tool attempts to extract company/year filters automatically.
    If extraction fails, it falls back to simple parsing.
    """
    try:
        params, extraction_warning = _extract_search_params(user_query)

        top_k = max(1, min(params.top_k or 5, 20))
        qdrant_filter = _build_qdrant_filter(params)

        try:
            embed_model = _get_embed_model()
            query_vector = embed_model.encode(params.query_text).tolist()
        except Exception as error:
            return (
                f"VECTOR TOOL ERROR [EmbeddingError]: Failed to embed query text.\n"
                f"Details: {type(error).__name__}: {error}"
            )

        try:
            qdrant_client = _get_qdrant_client()
        except Exception as error:
            return (
                f"VECTOR TOOL ERROR [QdrantClientInitError]: Failed to initialize Qdrant client.\n"
                f"Details: {type(error).__name__}: {error}"
            )

        used_unfiltered_fallback = False

        try:
            results = _query_qdrant(
                client=qdrant_client,
                query_vector=query_vector,
                qdrant_filter=qdrant_filter,
                limit=top_k,
            )

        except UnexpectedResponse as error:
            msg = str(error)

            # Qdrant can reject filtered queries if payload indexes are missing.
            # Fall back to unfiltered semantic search so the agent still gets evidence.
            if qdrant_filter and (
                "Index required" in msg
                or "index required" in msg.lower()
                or "index" in msg.lower()
            ):
                try:
                    results = _query_qdrant(
                        client=qdrant_client,
                        query_vector=query_vector,
                        qdrant_filter=None,
                        limit=top_k,
                    )
                    used_unfiltered_fallback = True

                except Exception as fallback_error:
                    return (
                        "VECTOR TOOL ERROR [QdrantFilteredSearchFailed]: "
                        "Filtered search failed, and unfiltered fallback also failed.\n"
                        f"Filtered error: {type(error).__name__}: {error}\n"
                        f"Fallback error: {type(fallback_error).__name__}: {fallback_error}"
                    )
            else:
                return (
                    f"VECTOR TOOL ERROR [QdrantUnexpectedResponse]: Qdrant rejected the query.\n"
                    f"Details: {type(error).__name__}: {error}"
                )

        except Exception as error:
            return (
                f"VECTOR TOOL ERROR [QdrantQueryError]: Could not query Qdrant.\n"
                f"Details: {type(error).__name__}: {error}"
            )

        points = getattr(results, "points", None) or []

        # If metadata filter worked but returned no hits, retry unfiltered once.
        if not points and qdrant_filter:
            try:
                results = _query_qdrant(
                    client=qdrant_client,
                    query_vector=query_vector,
                    qdrant_filter=None,
                    limit=top_k,
                )
                points = getattr(results, "points", None) or []
                used_unfiltered_fallback = True

            except Exception as error:
                return (
                    "VECTOR TOOL ERROR [NoFilteredResultsAndFallbackFailed]: "
                    "Filtered search returned no hits, and unfiltered fallback failed.\n"
                    f"Details: {type(error).__name__}: {error}"
                )

        if not points:
            return (
                f"No vector results found for query='{params.query_text}' "
                f"with filters company={params.company_name}, year={params.document_year}."
            )

        notes = []

        if extraction_warning:
            notes.append(f"[Warning] {extraction_warning}")

        if used_unfiltered_fallback:
            notes.append(
                "[Warning] Metadata-filtered search failed or returned no hits; "
                "showing unfiltered semantic results instead."
            )

        formatted_results = []

        for i, hit in enumerate(points, 1):
            payload = hit.payload or {}

            company = payload.get("company_name", "Unknown company")
            year = payload.get("document_year", "unknown year")
            section = (
                payload.get("section")
                or payload.get("document_section")
                or payload.get("source_section")
                or "unknown section"
            )
            text = payload.get("text") or "[No text payload found]"

            formatted_results.append(
                f"[Result {i} | Score: {hit.score:.3f} | "
                f"Source: {company}, {year}, {section}]\n"
                f"{text}"
            )

        prefix = ""
        if notes:
            prefix = "\n".join(notes) + "\n\n"

        return prefix + "\n---\n".join(formatted_results)

    except Exception as error:
        return (
            f"VECTOR TOOL ERROR [{type(error).__name__}]: Unexpected vector tool failure.\n"
            f"Details: {error}\n"
            "The agent should continue with SQL if useful, or explain that document search is unavailable."
        )