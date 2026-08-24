import pymupdf
from pathlib import Path


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF page by page.

    Returns:
        list: A list of dictionaries containing:
              - document
              - page
              - text
    """

    pdf_path = Path(pdf_path)

    documents = []

    pdf = pymupdf.open(pdf_path)
    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text("text").strip()

        if text:
            documents.append({
                "document": pdf_path.name,
                "page": page_number,
                "text": text
            })

    pdf.close()

    return documents


def load_all_pdfs(directory):
    """
    Extract text from all PDFs inside a directory.
    """

    directory = Path(directory)

    all_documents = []

    pdf_files = sorted(directory.glob("*.pdf"))

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")

        pages = extract_text_from_pdf(pdf_file)

        all_documents.extend(pages)

        print(f"  Pages extracted: {len(pages)}")

    return all_documents


if __name__ == "__main__":
    data_directory = Path(__file__).parent.parent / "data" / "documents"

    documents = load_all_pdfs(data_directory)

    print("\n" + "=" * 60)
    print("PDF EXTRACTION COMPLETE")
    print("=" * 60)

    print(f"Total pages extracted: {len(documents)}")

    for document in documents:
        print(
            f"{document['document']} | "
            f"Page {document['page']} | "
            f"{len(document['text'])} characters"
        )
