import argparse
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

load_dotenv()

DB_PATH = os.getenv("SQLITE_DB_PATH", "data/financials.db")
API_KEY = os.getenv("GOOGLE_API_KEY")
CLIENT = genai.Client(api_key=API_KEY) if API_KEY else None

SECTION_MARKERS = [
    "income statements",
    "consolidated statements of operations",
    "consolidated statements of income",
    "consolidated statements of earnings",
    "consolidated balance sheet",
    "financial statements and supplementary data",
    "item 8",
]

NUMERIC_FIELDS = {
    "revenue_billions": [
        "total revenue", "net sales", "revenues", "total net sales",
        "net sales and revenue",
    ],
    "net_income_billions": [
        "net income", "net loss", "net earnings", "net loss attributable",
    ],
    "total_assets_billions": [
        "total assets", "assets total", "total consolidated assets",
    ],
    "rd_expense_billions": [
        "research and development", "research & development", "rd expense", "r&d",
    ],
    "employees_thousands": [
        "employees", "number of employees", "total employees",
    ],
}

INFER_FIELDS = [
    "revenue_billions",
    "net_income_billions",
    "total_assets_billions",
    "employees_thousands",
    "rd_expense_billions",
    "operating_margin_pct",
]


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def find_financial_section(text: str) -> str:
    text_lower = text.lower()
    best_idx = -1
    for marker in SECTION_MARKERS:
        idx = text_lower.find(marker)
        if idx != -1 and idx > best_idx:
            best_idx = idx
    if best_idx != -1:
        print(f"  Financial section found at position {best_idx} ({round(best_idx / len(text) * 100)}% into document)")
        return text[best_idx:best_idx + 40000]

    print("  No marker found — using fallback (last 40%)")
    start = int(len(text) * 0.6)
    return text[start:start + 40000]


def parse_number(token: str) -> float | None:
    if not token:
        return None
    token = token.replace(',', '').replace('$', '').strip()
    negative = token.startswith('(') and token.endswith(')')
    if negative:
        token = token[1:-1]
    try:
        return -float(token) if negative else float(token)
    except ValueError:
        return None


def scale_number(field: str, value: float | None, context: str | None = None) -> float | None:
    if value is None:
        return None
    ctx = (context or '').lower()
    if field == "employees_thousands":
        if abs(value) > 1000:
            return value / 1000.0
        return value
    if "thousand" in ctx:
        return value / 1_000_000.0
    if "million" in ctx:
        return value / 1000.0
    if "billion" in ctx:
        return value
    if abs(value) >= 1_000_000_000:
        return value / 1_000_000_000.0
    if abs(value) > 1000:
        return value / 1000.0
    return value


def find_year_header(lines: list[str], target_year: int) -> tuple[int | None, int | None]:
    for i, line in enumerate(lines[:80]):
        cols = re.split(r"\s{2,}", line.strip())
        if any(str(target_year) in col for col in cols):
            for j, col in enumerate(cols):
                if str(target_year) in col:
                    return i, j
    return None, None


def extract_from_table(section: str, target_year: int) -> dict[str, float | None]:
    lines = [line for line in section.splitlines() if line.strip()]
    header_idx, year_col = find_year_header(lines, target_year)
    result: dict[str, float | None] = {k: None for k in NUMERIC_FIELDS}

    if header_idx is not None:
        for line in lines[header_idx + 1: header_idx + 250]:
            cols = re.split(r"\s{2,}", line.strip())
            if len(cols) < 2:
                continue
            label = cols[0].lower()
            for field, patterns in NUMERIC_FIELDS.items():
                if any(pattern in label for pattern in patterns):
                    target_token = None
                    if year_col is not None and year_col < len(cols):
                        target_token = cols[year_col]
                    else:
                        for candidate in cols[1:]:
                            if re.search(r"[0-9()\.,]+", candidate):
                                target_token = candidate
                                break
                    if not target_token:
                        continue
                    value = parse_number(re.sub(r"[^0-9().,\-]", '', target_token))
                    result[field] = scale_number(field, value, ' '.join(cols))
    return result


def extract_from_labels(section: str) -> dict[str, float | None]:
    result: dict[str, float | None] = {k: None for k in NUMERIC_FIELDS}
    for field, patterns in NUMERIC_FIELDS.items():
        for pattern in patterns:
            regex = re.compile(
                rf"{re.escape(pattern)}[\s\S]{{0,180}}?([\d,\(\)\.]+)(?:[^\d\n\r]{{0,80}}(million|billion|thousand))?",
                re.IGNORECASE,
            )
            match = regex.search(section)
            if match:
                raw_value = parse_number(match.group(1))
                context = match.group(0)
                result[field] = scale_number(field, raw_value, context)
                break
    return result


def local_extract_financials(text: str, company: str, year: int) -> dict[str, Any]:
    section = find_financial_section(text)
    table_data = extract_from_table(section, year)
    label_data = extract_from_labels(section)
    merged = {k: table_data.get(k) if table_data.get(k) is not None else label_data.get(k) for k in NUMERIC_FIELDS}
    merged["company_name"] = company
    merged["year"] = year
    merged["operating_margin_pct"] = None
    return merged


def extract_financials_with_gemini(text: str, company: str, year: int) -> dict[str, Any] | None:
    if not CLIENT:
        return None
    section = find_financial_section(text)
    prompt = f"""
You are a financial data extractor. From the 10-K filing text below for {company} ({year}),
extract ONLY valid JSON with these fields:
- revenue_billions
- net_income_billions
- total_assets_billions
- employees_thousands
- rd_expense_billions
- operating_margin_pct
If a figure cannot be found, use null.

Filing text:
{section}
"""
    response = CLIENT.models.generate_content(
        model="gemini-2.5-flash-lite",
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
    except json.JSONDecodeError:
        return None


def create_table(cursor: sqlite3.Cursor) -> None:
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS company_financials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT NOT NULL,
            year INTEGER NOT NULL,
            revenue_billions REAL,
            net_income_billions REAL,
            total_assets_billions REAL,
            employees_thousands REAL,
            rd_expense_billions REAL,
            operating_margin_pct REAL
        )
    """)
    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_company_year
        ON company_financials(company_name, year)
    """)


def insert_or_update(cursor: sqlite3.Cursor, data: dict[str, Any]) -> None:
    cursor.execute("""
        INSERT OR REPLACE INTO company_financials
            (company_name, year, revenue_billions, net_income_billions,
             total_assets_billions, employees_thousands, rd_expense_billions,
             operating_margin_pct)
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


def summarize(data: dict[str, Any]) -> str:
    return (
        f"revenue={data.get('revenue_billions')}B | "
        f"net_income={data.get('net_income_billions')}B | "
        f"employees={data.get('employees_thousands')}k | "
        f"assets={data.get('total_assets_billions')}B | "
        f"rd={data.get('rd_expense_billions')}B"
    )


def populate_database(pdf_dir: str = "data/documents", pdf_filter: str | None = None, local_only: bool = False) -> None:
    pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))
    if pdf_filter:
        pdf_files = [path for path in pdf_files if pdf_filter.lower() in path.name.lower()]
    if not pdf_files:
        print(f"No PDFs found in '{pdf_dir}'.")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    create_table(cursor)

    print(f"Processing {len(pdf_files)} PDF(s) from '{pdf_dir}'...")
    print("=" * 60)

    inserted = 0
    skipped = 0

    for index, pdf_path in enumerate(pdf_files, start=1):
        parts = pdf_path.stem.split("_")
        company = " ".join(parts[:-1]).title()
        year = int(parts[-1])

        print(f"\n[{index}/{len(pdf_files)}] {company} {year}")
        text = extract_text_from_pdf(pdf_path)
        result = None

        if not local_only and CLIENT:
            result = extract_financials_with_gemini(text, company, year)
            if result:
                metrics = [result.get(k) for k in INFER_FIELDS]
                if all(v is None for v in metrics):
                    result = None
                else:
                    print("  Gemini extraction succeeded")

        if result is None:
            print("  Falling back to local extraction")
            result = local_extract_financials(text, company, year)

        if result:
            insert_or_update(cursor, result)
            inserted += 1
            print("  ✓", summarize(result))
        else:
            skipped += 1
            print("  ✗ Skipped — no data extracted")

    conn.commit()

    print(f"\n{'=' * 60}")
    print(f"Run complete: {inserted} inserted/updated, {skipped} skipped")
    print(f"\nCurrent database contents ({DB_PATH}):")
    cursor.execute("""
        SELECT company_name, year, revenue_billions, net_income_billions
        FROM company_financials
        ORDER BY company_name, year
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]:<12} {row[1]}  revenue={row[2]}B  net_income={row[3]}B")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract financial metrics from 10-K PDFs.")
    parser.add_argument("filter", nargs="?", help="Optional PDF name filter")
    parser.add_argument("--filter", dest="filter", help="Optional PDF name filter")
    parser.add_argument("--local-only", action="store_true", help="Use only local extraction heuristics")
    parser.add_argument("--pdf-dir", default="data/documents", help="Directory containing PDF files")
    args = parser.parse_args()

    populate_database(pdf_dir=args.pdf_dir, pdf_filter=args.filter, local_only=args.local_only)
