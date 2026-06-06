from pypdf import PdfReader
from pathlib import Path
from tqdm import tqdm


def extract_pdf_to_text(pdf_path: Path, output_path: Path):
   
   reader = PdfReader(str(pdf_path))
   pages_text = []

   for page in reader.pages:
      page_text = page.extract_text()
      if page_text:
            pages_text.append(page_text)

   full_text = "\n".join(pages_text)

   # Save with the same stem but .txt extension
   output_file = output_path / (pdf_path.stem + ".txt")
   output_file.write_text(full_text, encoding="utf-8")
   print(f"Extracted text from {pdf_path.name} to {output_file.name}")
   return output_file

def extract_all_pdfs(input_dir: str = "data/documents/", output_dir: str = "data/extracted/"):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {input_dir}. Make sure files are named COMPANY_YEAR.pdf")
        return

    print(f"Found {len(pdf_files)} PDFs to extract...")

    for pdf_file in tqdm(pdf_files, desc="Extracting PDFs"):
        try:
            out_file = extract_pdf_to_text(pdf_file, output_path)
            size_kb = out_file.stat().st_size // 1024
            print(f"  {pdf_file.name} -> {out_file.name} ({size_kb} KB)")
        except Exception as e:
            print(f"  ERROR extracting {pdf_file.name}: {e}")

    print(f"\nDone. Extracted text files are in: {output_dir}")

if __name__ == "__main__":
    extract_all_pdfs()