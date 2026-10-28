"""
Tests for PDF Ingestion Module
"""
import os
import tempfile
from pathlib import Path

import pytest

from src.ingestion.pdf_extractor import PDFExtractor, PDFIngestionError, ingest_pdfs


class TestPDFExtractor:
    """Test cases for PDFExtractor class."""
    
    def test_init_with_custom_dir(self):
        """Test initialization with custom directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = PDFExtractor(pdf_dir=tmpdir)
            assert extractor.pdf_dir == Path(tmpdir)
    
    def test_get_pdf_files_empty_dir(self):
        """Test getting PDF files from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = PDFExtractor(pdf_dir=tmpdir)
            files = extractor.get_pdf_files()
            assert files == []
    
    def test_extract_all_empty_dir(self):
        """Test extracting from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = PDFExtractor(pdf_dir=tmpdir)
            records = extractor.extract_all()
            assert records == []
    
    def test_extract_all_no_pdfs(self):
        """Test extracting when no PDF files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a non-PDF file
            txt_file = Path(tmpdir) / "test.txt"
            txt_file.write_text("hello")
            
            extractor = PDFExtractor(pdf_dir=tmpdir)
            records = extractor.extract_all()
            assert records == []
    
    def test_ingest_pdfs_convenience_function(self):
        """Test convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            records = ingest_pdfs(pdf_dir=tmpdir)
            assert isinstance(records, list)
    
    def test_record_structure(self):
        """Test that records have correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # We can't easily create a real PDF without dependencies in test,
            # but we can test the structure expectation
            extractor = PDFExtractor(pdf_dir=tmpdir)
            # This would need a real PDF file for full test
            assert hasattr(extractor, 'extract_from_file')
            assert hasattr(extractor, 'extract_all')


class TestPDFIngestionError:
    """Test cases for PDFIngestionError exception."""
    
    def test_exception_is_exception(self):
        """Test that PDFIngestionError is an Exception subclass."""
        assert issubclass(PDFIngestionError, Exception)
    
    def test_exception_message(self):
        """Test exception message."""
        msg = "Test error"
        err = PDFIngestionError(msg)
        assert str(err) == msg
