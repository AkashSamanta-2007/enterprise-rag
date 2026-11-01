"""
PDF Ingestion Module
Handles PDF document loading and text extraction using PyMuPDF.
also file name and page number metadata for each extracted page.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFIngestionError(Exception):
    """Custom exception for PDF ingestion errors."""
    pass


class PDFExtractor:
    """
    PDF text extractor using PyMuPDF (fitz).
    Extracts page-level text with source filename and page number metadata.
    """
    
    def __init__(self, pdf_dir: Optional[str] = None):
        """
        Initialize PDF extractor.
        
        Args:
            pdf_dir: Directory containing PDF files. Defaults to config.PDF_INPUT_DIR.
        """
        if fitz is None:
            raise PDFIngestionError(
                "PyMuPDF (fitz) is not installed. Please install it with: pip install PyMuPDF"
            )
        
        if pdf_dir is None:
            from src.config import config
            pdf_dir = config.PDF_INPUT_DIR
        
        self.pdf_dir = Path(pdf_dir)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"PDFExtractor initialized with directory: {self.pdf_dir}")
    
    def extract_from_file(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract text from a single PDF file.
        
        Args:
            pdf_path: Path to PDF file.
            
        Returns:
            List of document records with text, source, and page_number.
        """
        records = []
        source_name = pdf_path.name
        
        try:
            logger.info(f"Processing PDF: {source_name}")
            doc = fitz.open(pdf_path)
            
            if len(doc) == 0:
                logger.warning(f"Empty PDF detected: {source_name}")
                doc.close()
                return records
            
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    text = page.get_text("text")
                    
                    # Check for scanned PDF with no text
                    if not text or not text.strip():
                        logger.warning(
                            f"No extractable text found on page {page_num + 1} of {source_name}. "
                            "This may be a scanned PDF requiring OCR."
                        )
                        # Still create record with empty text for tracking, or skip
                        # Skip empty pages to avoid noise
                        continue
                    
                    record = {
                        "text": text.strip(),
                        "source": source_name,
                        "page_number": page_num + 1
                    }
                    records.append(record)
                    logger.debug(f"Extracted page {page_num + 1} from {source_name} ({len(text)} chars)")
                    
                except Exception as e:
                    logger.error(f"Error extracting page {page_num + 1} from {source_name}: {e}")
                    # Continue processing other pages
                    continue
            
            doc.close()
            logger.info(f"Successfully extracted {len(records)} pages from {source_name}")
            
        except Exception as e:
            logger.error(f"Error opening/processing PDF {source_name}: {e}")
            raise PDFIngestionError(f"Failed to process {source_name}: {e}")
        
        return records
    
    def extract_all(self) -> List[Dict[str, Any]]:
        """
        Extract text from all PDF files in the pdf_dir.
        
        Returns:
            List of all document records from all PDFs.
        """
        all_records = []
        
        if not self.pdf_dir.exists():
            logger.warning(f"PDF directory does not exist: {self.pdf_dir}")
            return all_records
        
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.info(f"No PDF files found in {self.pdf_dir}")
            return all_records
        
        logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
        
        for pdf_file in pdf_files:
            try:
                records = self.extract_from_file(pdf_file)
                all_records.extend(records)
            except PDFIngestionError as e:
                logger.error(f"Skipping {pdf_file.name} due to error: {e}")
                # Continue with other files
                continue
        
        logger.info(f"Total extracted records: {len(all_records)}")
        return all_records
    
    def get_pdf_files(self) -> List[Path]:
        """
        Get list of all PDF files in the pdf_dir.
        
        Returns:
            List of PDF file paths.
        """
        if not self.pdf_dir.exists():
            return []
        return sorted(self.pdf_dir.glob("*.pdf"))


def ingest_pdfs(pdf_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Convenience function to ingest all PDFs from directory.
    
    Args:
        pdf_dir: Directory containing PDF files.
        
    Returns:
        List of document records with text, source, and page_number.
    """
    extractor = PDFExtractor(pdf_dir=pdf_dir)
    return extractor.extract_all()
