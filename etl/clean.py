import re
from pathlib import Path

def clean_document(text:str) -> str:
   
    # Fix PDF hyphenation breaks: "busi-\nness" → "business"
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # Remove standalone page numbers
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\bPage\s+\d+\s+of\s+\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*[-—]\s*\d+\s*[-—]\s*$', '', text, flags=re.MULTILINE)

    # Remove running headers/footers (short ALL CAPS lines appearing alone)
    text = re.sub(r'^[A-Z\s,\.]{5,60}$\n', '', text, flags=re.MULTILINE)

     # Remove table of contents dot leaders: "Item 1A......... 22"
    text = re.sub(r'\.{3,}\s*\d+', '', text)

     # Fix Unicode ligatures introduced by PDF fonts
    text = text.replace('\ufb01', 'fi')   # ﬁ → fi
    text = text.replace('\ufb02', 'fl')   # ﬂ → fl
    text = text.replace('\u2019', "'")    # Right single quote
    text = text.replace('\u2018', "'")    # Left single quote
    text = text.replace('\u201c', '"')    # Left double quote
    text = text.replace('\u201d', '"')    # Right double quote
    text = text.replace('\u2014', '--')   # Em dash

     # Remove XBRL / HTML remnants
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-z]+;', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

def clean_all_documents(input_dir: str = "data/extracted", output_dir: str = "data/cleaned"):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    txt_files = list(input_path.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {input_dir}. Run etl/extract_pdfs.py first.")
        return

    for file in txt_files:
        print(f"Cleaning: {file.name}")
        raw_text = file.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_document(raw_text)
        out_file = output_path / file.name
        out_file.write_text(cleaned, encoding="utf-8")
        reduction = round((1 - len(cleaned) / max(len(raw_text), 1)) * 100, 1)
        print(f"  -> {out_file.name} ({len(cleaned):,} chars, {reduction}% noise removed)")


if __name__ == "__main__":
    clean_all_documents()

