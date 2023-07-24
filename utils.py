from PyPDF2 import PdfReader
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_file: object) -> str:
    """Function To extract texts from pdf paged

    Args:
        pdf_file (object): pdf file object

    Returns:
        str: text combined from all pages in pdf
    """
    combined_texts = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        combined_texts += page_text
    return combined_texts


def split_text_to_chunks(combined_texts: str, chunk_size: int, chunk_overlap: int) -> List:
    """Function to split texts from pdf into a list of strings(chunks)

    Args:
        combined_texts (str): texts generated from pdf
        chunk_size (int): max token of each chunk size
        chunk_overlap (int): number of token overlap between chunks

    Returns:
        list: list of texts in parts
    """
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    chunks = text_splitter.split_text(combined_texts)
    return chunks
