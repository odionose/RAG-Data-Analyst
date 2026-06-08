import sqlite3
import os
from typing import Optional
from dotenv import load_dotenv

import instructor
from google import genai as google_genai
from pydantic import BaseModel, Field
from langchain.tools import tool
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

load_dotenv()

# ─── Shared resources (loaded once at import) ────────────────────────────────
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
_qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY", None),
)
_collection = os.getenv("QDRANT_COLLECTION", "documents")
_db_path = os.getenv("SQLITE_DB_PATH", "data/financials.db")

# Instructor with Gemini for structured extraction - review
_genai_client = google_genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
_instructor_client = instructor.from_genai(
    client=_genai_client,
    mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
)
gemini_model = "gemini-2.5-flash-lite"


# ─── Tool 1: execute_sql ─────────────────────────────────────────────────────

@tool
def execute_sql(query: str) -> str:
    """
    Execute a SQL SELECT query against the company financials SQLite database.
    The database has one table: company_financials
    Columns: company_name (TEXT), year (INT), revenue_billions (REAL),
             net_income_billions (REAL), total_assets_billions (REAL),
             employees_thousands (REAL), rd_expense_billions (REAL),
             operating_margin_pct (REAL)

    Use this for precise numerical lookups like:
    - "What was Apple's revenue in 2023?"
    - "Which company had the highest net income in 2022?"
    - "Compare R&D spending across companies"
    """
    max_retries = 3

    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return "Query executed successfully but returned no rows."

            columns = rows[0].keys()
            header = " | ".join(columns)
            separator = "-" * len(header)
            data_rows = [" | ".join(str(row[col]) for col in columns) for row in rows]
            return "\n".join([header, separator] + data_rows)

        except sqlite3.Error as e:
            if attempt < max_retries - 1:
                # Error recovery: return the error so the LLM can fix the query
                return (
                    f"SQL ERROR (attempt {attempt + 1}/{max_retries}): {str(e)}\n"
                    f"The query was: {query}\n"
                    "Please fix the SQL syntax and try again."
                )

    return f"SQL failed after {max_retries} attempts."


# ─── Tool 2: search_vector_db ─────────────────────────────────────────────────

class VectorSearchParams(BaseModel):
    """
    Structured search parameters extracted from a natural language query.
    Instructor uses Gemini to populate this model before the semantic search runs,
    so the search is filtered to the right company and year automatically.
    """
    query_text: str = Field(
        description="The core semantic search query, cleaned of company/year filters"
    )
    company_name: Optional[str] = Field(
        default=None,
        description="Company name to filter by if mentioned (e.g. 'Apple', 'Microsoft'). None if not specified."
    )
    document_year: Optional[int] = Field(
        default=None,
        description="4-digit year to filter by if mentioned (e.g. 2023). None if not specified."
    )
    top_k: int = Field(
        default=5,
        ge=1, le=20,
        description="Number of results to return. Use 3-5 for specific questions, 8-10 for broad surveys."
    )


@tool
def search_vector_db(user_query: str) -> str:
    """
    Search the document vector database using semantic similarity.
    Use this for qualitative questions like:
    - "What are Apple's main risk factors?"
    - "How does Microsoft describe its cloud strategy?"
    - "What does Tesla say about competition in 2023?"

    The tool automatically extracts company/year filters from your query.
    """
    # Step 1: Use the Instructor wrapper to extract structured params directly
    prompt = (
        "Extract search parameters from this query. "
        "Be precise with company names and years. "
        "The query_text should be the semantic search query "
        "WITHOUT the company/year constraints.\n\n"
        f"Query: {user_query}"
    )

    try:
        params = _instructor_client.create(
            response_model=VectorSearchParams,
            messages=[{"role": "user", "content": prompt}],
            model=gemini_model,
        )
    except Exception as e:
        return (
            f"Failed to extract structured search parameters from the query: {e}\n"
            "Consider installing 'jsonref' or using a simpler fallback parsing approach."
        )

    # Step 2: Build Qdrant metadata filters
    must_conditions = []
    if params.company_name:
        must_conditions.append(
            FieldCondition(key="company_name", match=MatchValue(value=params.company_name))
        )
    if params.document_year:
        must_conditions.append(
            FieldCondition(key="document_year", match=MatchValue(value=params.document_year))
        )

    qdrant_filter = Filter(must=must_conditions) if must_conditions else None

    # Step 3: Embed the semantic query
    query_vector = _embed_model.encode(params.query_text).tolist()

    # Step 4: Search with both semantic + metadata filtering
    results = _qdrant_client.query_points(
        collection_name=_collection,
        query=query_vector,
        query_filter=qdrant_filter,
        limit=params.top_k,
        with_payload=True,
    )

    if not results or not getattr(results, 'points', None):
        return (
            f"No results found for: '{params.query_text}' "
            f"with filters: company={params.company_name}, year={params.document_year}"
        )

    formatted = []
    for i, hit in enumerate(results.points, 1):
        payload = hit.payload or {}
        formatted.append(
            f"[Result {i} | Score: {hit.score:.3f} | "
            f"{payload.get('company_name', 'Unknown')} {payload.get('document_year', '')}]\n"
            f"{payload.get('text', '')}\n"
        )

    return "\n---\n".join(formatted)