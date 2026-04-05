# src/ocr_loader.py

from pathlib import Path
import pytesseract
from pdf2image import convert_from_path

from langchain_core.documents import Document



BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "pdfs"


def ocr_pdf(pdf_path: Path):
    """
    Convert PDF pages to text using OCR
    """
    images = convert_from_path(pdf_path)
    documents = []

    for page_number, image in enumerate(images, start=1):
        text = pytesseract.image_to_string(image)

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "filename": pdf_path.name,
                        "page_number": page_number,
                        "source": str(pdf_path),
                        "loader": "ocr",
                    },
                )
            )

    return documents


def load_all_pdfs_ocr():
    all_docs = []
    pdf_files = list(PDF_DIR.glob("*.pdf"))

    print(f"Loading {len(pdf_files)} PDFs using OCR...")

    for pdf in pdf_files:
        print(f"→ OCR processing: {pdf.name}")
        docs = ocr_pdf(pdf)
        all_docs.extend(docs)

    return all_docs


if __name__ == "__main__":
    docs = load_all_pdfs_ocr()
    print(f"\nTotal OCR documents loaded: {len(docs)}")
    print("\n--- Sample OCR Text ---\n")
    print(docs[0].page_content[:500])
