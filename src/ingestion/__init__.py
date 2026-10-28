"""
Ingestion Module
Handles PDF document loading and text extraction.
"""

from src.ingestion.pdf_extractor import PDFExtractor, PDFIngestionError, ingest_pdfs

__all__ = ["PDFExtractor", "PDFIngestionError", "ingest_pdfs"]
