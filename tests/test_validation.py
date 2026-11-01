"""
Tests for Input Validation
"""
import pytest
from pathlib import Path
from src.validation import InputValidator, ValidationError, validate_question, validate_filename


class TestInputValidator:
    """Test input validation."""
    
    def test_validate_question_valid(self):
        """Test valid question."""
        q = validate_question("What is the main topic?")
        assert q == "What is the main topic?"
    
    def test_validate_question_strips_whitespace(self):
        """Test whitespace stripping."""
        q = validate_question("  What is this?  ")
        assert q == "What is this?"
    
    def test_validate_question_empty_raises(self):
        """Test empty question raises error."""
        with pytest.raises(ValidationError):
            validate_question("")
    
    def test_validate_question_too_short_raises(self):
        """Test too short question raises error."""
        with pytest.raises(ValidationError):
            validate_question("Hi")
    
    def test_validate_question_too_long_raises(self):
        """Test too long question raises error."""
        long_q = "a" * 2001
        with pytest.raises(ValidationError):
            validate_question(long_q)
    
    def test_validate_question_removes_null_bytes(self):
        """Test null byte removal."""
        q = validate_question("What is\x00 this?")
        assert '\x00' not in q
    
    def test_validate_filename_valid(self):
        """Test valid filename."""
        f = validate_filename("document.pdf")
        assert f == "document.pdf"

    def test_validate_filename_uppercase_extension(self):
        """Test uppercase PDF extension is accepted."""
        f = validate_filename("document.PDF")
        assert f == "document.PDF"

    def test_validate_filename_with_spaces(self):
        """Test spaces in a PDF filename are accepted."""
        f = validate_filename("Company Handbook.pdf")
        assert f == "Company Handbook.pdf"
    
    def test_validate_filename_with_path_traversal(self):
        """Test path traversal blocked."""
        with pytest.raises(ValidationError):
            validate_filename("../../etc/passwd.pdf")
    
    def test_validate_filename_not_pdf(self):
        """Test non-PDF rejected."""
        with pytest.raises(ValidationError):
            validate_filename("document.txt")
    
    def test_validate_filename_invalid_chars(self):
        """Test unsupported characters are rejected."""
        with pytest.raises(ValidationError):
            validate_filename("doc@ument.pdf")
    
    def test_validate_pdf_path_within_dir(self):
        """Test path validation within directory."""
        base = Path("/tmp/docs")
        file_path = Path("/tmp/docs/test.pdf")
        assert InputValidator.validate_pdf_path(file_path, base) is True
    
    def test_sanitize_metadata(self):
        """Test metadata sanitization."""
        meta = {"source": "doc.pdf", "page_number": 1, "bad-key!": "x"}
        sanitized = InputValidator.sanitize_metadata(meta)
        assert "source" in sanitized
        assert "page_number" in sanitized
        assert "bad-key!" not in sanitized
