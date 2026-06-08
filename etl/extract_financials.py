import os
import re
import json
import sqlite3
from pathlib import Path
from urllib import response
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

DB_PATH = os.getenv("SQLITE_DB_PATH", "data/financials.db")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract raw text from PDF."""
    reader = PdfReader(str(pdf_path))
    return "\n\n".join(
        page.extract_text() for page in reader.pages if page.extract_text()
    )


def find_financial_section(text: str) -> str:
  
    markers = [
        "income statements",                              # catches Microsoft's index header
        "consolidated statements of operations",
        "consolidated statements of income",
        "consolidated statements of earnings",
        "consolidated balance sheet",
        "financial statements and supplementary data",
        "item 8",
    ]

    text_lower = text.lower()
    best_idx = -1

    for marker in markers:
        # Find ALL occurrences of this marker
        start = 0
        while True:
            idx = text_lower.find(marker, start)
            if idx == -1:
                break
            # Skip TOC hits — real financial sections are never in the first 10% of the doc
            if idx > len(text) * 0.10:
                best_idx = max(best_idx, idx)
            start = idx + 1

    if best_idx != -1:
        print(f"  Financial section found at position {best_idx} "
              f"({round(best_idx/len(text)*100)}% into document)")
        return text[best_idx:best_idx + 40000]

    # Fallback: last 40% of the document
    print("  No marker found — using fallback (last 40%)")
    start = int(len(text) * 0.6)
    return text[start:start + 40000]
def extract_financials_with_gemini(text: str, company: str, year: int) -> dict | None:
    """
    Send the financial section to Gemini and parse the JSON response.
    Returns a dict with all fields, or None if parsing fails.
    """
    financial_text = find_financial_section(text)

    # TEMPORARY: print what's being sent to Gemini
    print(f"\n  --- SECTION PREVIEW (first 1000 chars) ---")
    print(financial_text[:1000])
    print(f"  --- END PREVIEW ---\n")

    prompt = f"""
You are a financial data extractor. From the 10-K filing text below for {company} ({year}),
extract ONLY these specific figures. Return ONLY valid JSON, no explanation, no markdown.

Required fields:
- revenue_billions: Total revenue/net sales in billions USD (divide millions by 1000)
- net_income_billions: Net income in billions USD
- total_assets_billions: Total assets in billions USD
- employees_thousands: Total employees in thousands (divide by 1000)
- rd_expense_billions: Research and development expense in billions USD
- operating_margin_pct: Operating income / Revenue * 100 (as a percentage)

If a figure cannot be found, use null.

Filing text:
{financial_text}

Return JSON only:
"""

    response = client.models.generate_content(
    model="gemini-2.5-flash-lite", # review
    contents=prompt,
    )
    raw = response.text.strip()
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw)
    raw = raw.strip()

    try:
        data = json.loads(raw)
        data["company_name"] = company
        data["year"] = year
        return data
    except json.JSONDecodeError as e:
        print(f"  WARNING: Could not parse JSON for {company} {year}: {e}")
        print(f"  Raw response: {raw[:200]}")
        return None


def create_table(cursor: sqlite3.Cursor):
    """
    Create the table if it doesn't exist.
    Remove legacy duplicates and enforce a unique index on company/year.
    """
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS company_financials (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        company_name         TEXT    NOT NULL,
        year                 INTEGER NOT NULL,
        revenue_billions     REAL,
        net_income_billions  REAL,
        total_assets_billions REAL,
        employees_thousands  REAL,
        rd_expense_billions  REAL,
        operating_margin_pct REAL
    )
    """)

    cursor.execute("""
    DELETE FROM company_financials
    WHERE id NOT IN (
        SELECT MIN(id)
        FROM company_financials
        GROUP BY company_name, year
    )
    """)

    cursor.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS idx_company_year
    ON company_financials(company_name, year)
    """)


def populate_database(pdf_dir: str = "data/documents", pdf_filter: str = None):

    pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))

    # Apply optional filter (e.g. python extract_financials.py apple)
    if pdf_filter:
        pdf_files = [f for f in pdf_files if pdf_filter.lower() in f.name.lower()]

    if not pdf_files:
        print(f"No PDFs found in '{pdf_dir}'.")
        print("Make sure your files are named COMPANY_YEAR.pdf (e.g. apple_2025.pdf)")
        return

    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    create_table(cursor)

    print(f"Processing {len(pdf_files)} PDF(s) from '{pdf_dir}'...")
    print("=" * 60)

    inserted = 0
    skipped = 0

    for pdf_path in pdf_files:
        # Parse company name and year from filename: apple_2025.pdf → Apple, 2025
        parts = pdf_path.stem.split("_")
        company = " ".join(parts[:-1]).title()
        year = int(parts[-1])

        print(f"\n[{pdf_files.index(pdf_path) + 1}/{len(pdf_files)}] {company} {year}")
        text = extract_text_from_pdf(pdf_path)
        data = extract_financials_with_gemini(text, company, year)

        if data:
            # Check if Gemini returned all nulls — means it couldn't find anything
            values = [data.get(k) for k in [
             "revenue_billions", "net_income_billions", "total_assets_billions",
             "employees_thousands", "rd_expense_billions", "operating_margin_pct"
            ]]
            if all(v is None for v in values):
                print(f"  ✗ SKIPPED — all fields returned null, financial section not found")
                skipped += 1
                continue
            cursor.execute("""
                INSERT OR REPLACE INTO company_financials
                    (company_name, year, revenue_billions, net_income_billions,
                     total_assets_billions, employees_thousands,
                     rd_expense_billions, operating_margin_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["company_name"],
                data["year"],
                data.get("revenue_billions"),
                data.get("net_income_billions"),
                data.get("total_assets_billions"),
                data.get("employees_thousands"),
                data.get("rd_expense_billions"),
                data.get("operating_margin_pct"),
            ))
            inserted += 1
            print(f"  ✓ revenue={data.get('revenue_billions')}B | "
                  f"net_income={data.get('net_income_billions')}B | "
                  f"employees={data.get('employees_thousands')}k")
        else:
            skipped += 1
            print(f"  ✗ SKIPPED — extraction failed")

    conn.commit()

    # Summary — shows only the rows for THIS run, not all historical rows
    print(f"\n{'=' * 60}")
    print(f"Run complete: {inserted} inserted/updated, {skipped} skipped")
    print(f"\nCurrent database contents ({DB_PATH}):")
    cursor.execute("""
        SELECT company_name, year, revenue_billions, net_income_billions
        FROM company_financials
        ORDER BY company_name, year
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]:<12} {row[1]}  revenue=${row[2]}B  net_income=${row[3]}B")

    conn.close()


if __name__ == "__main__":
    import sys
    # Run all PDFs:         python etl/extract_financials.py
    # Run one company:      python etl/extract_financials.py apple
    pdf_filter = sys.argv[1] if len(sys.argv) > 1 else None
    populate_database(pdf_filter=pdf_filter)