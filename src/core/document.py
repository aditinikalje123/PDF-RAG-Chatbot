"""Document processing functionality."""

import logging
from pathlib import Path
from typing import List

import pymupdf  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles PDF document loading and processing."""

    def __init__(self, chunk_size: int = 7500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_pdf(self, file_path: Path) -> List[Document]:
        """Load PDF document using PyMuPDF."""

        try:
            logger.info(f"Loading PDF from {file_path}")

            pdf = pymupdf.open(str(file_path))
            documents = []

            for page_number, page in enumerate(pdf):
                text = page.get_text()

                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(file_path),
                                "page": page_number + 1
                            }
                        )
                    )

            pdf.close()

            logger.info(f"Loaded {len(documents)} pages from PDF")
            return documents

        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            raise

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""

        try:
            logger.info("Splitting documents into chunks")

            return self.splitter.split_documents(documents)

        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raises