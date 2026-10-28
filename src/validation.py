"""
Input Validation and Sanitization
Provides validation utilities for user inputs and documents.
"""

import re
from typing import Optional, List, Dict, Any
from pathlib import Path


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class InputValidator:
    """Validates and sanitizes user inputs."""
    
    # Maximum lengths
    MAX_QUESTION_LENGTH = 2000
    MAX_FILENAME_LENGTH = 255
    MIN_QUESTION_LENGTH = 3
    
    # Allowed filename pattern (alphanumeric, dash, underscore, dot)
    FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9._-]*\.pdf$')
    
    @classmethod
    def validate_question(cls, question: str) -> str:
        """
        Validate and sanitize user question.
        
        Args:
            question: User question string.
        
        Returns:
            Sanitized question.
        
        Raises:
            ValidationError: If question is invalid.
        """
        if not question:
            raise ValidationError("Question cannot be empty")
        
        if not isinstance(question, str):
            raise ValidationError("Question must be a string")
        
        # Strip whitespace
        sanitized = question.strip()
        
        if len(sanitized) < cls.MIN_QUESTION_LENGTH:
            raise ValidationError(f"Question must be at least {cls.MIN_QUESTION_LENGTH} characters")
        
        if len(sanitized) > cls.MAX_QUESTION_LENGTH:
            raise ValidationError(f"Question must be less than {cls.MAX_QUESTION_LENGTH} characters")
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        return sanitized
    
    @classmethod
    def validate_filename(cls, filename: str) -> str:
        """
        Validate uploaded filename.
        
        Args:
            filename: Filename to validate.
        
        Returns:
            Sanitized filename.
        
        Raises:
            ValidationError: If filename is invalid.
        """
        if not filename:
            raise ValidationError("Filename cannot be empty")
        
        if not isinstance(filename, str):
            raise ValidationError("Filename must be a string")
        
        # Check for path traversal attempts before extracting basename
        if '..' in filename or '/' in filename or '\\' in filename:
            raise ValidationError("Invalid filename")
        
        # Get just the basename (prevent path traversal)
        basename = Path(filename).name
        
        if basename != filename:
            raise ValidationError("Invalid filename")
        
        if len(basename) > cls.MAX_FILENAME_LENGTH:
            raise ValidationError(f"Filename must be less than {cls.MAX_FILENAME_LENGTH} characters")
        
        if not basename.lower().endswith('.pdf'):
            raise ValidationError("Only PDF files are allowed")
        
        if not cls.FILENAME_PATTERN.match(basename):
            raise ValidationError("Invalid filename. Use only alphanumeric characters, dots, dashes, and underscores")
        
        return basename
    
    @classmethod
    def validate_pdf_path(cls, file_path: Path, base_dir: Path) -> bool:
        """
        Validate that file path is within allowed directory.
        
        Args:
            file_path: File path to check.
            base_dir: Allowed base directory.
        
        Returns:
            True if valid.
        
        Raises:
            ValidationError: If path is outside allowed directory.
        """
        try:
            resolved_path = file_path.resolve()
            resolved_base = base_dir.resolve()
            resolved_path.relative_to(resolved_base)
            return True
        except ValueError:
            raise ValidationError("File path is outside allowed directory")
    
    @classmethod
    def sanitize_metadata(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata to prevent injection.
        
        Args:
            metadata: Metadata dict.
        
        Returns:
            Sanitized metadata.
        """
        sanitized = {}
        for key, value in metadata.items():
            # Only allow alphanumeric and underscore in keys
            if not re.match(r'^[a-zA-Z0-9_]+$', key):
                continue
            # Convert to string and limit length
            if isinstance(value, str):
                sanitized[key] = value[:500].replace('\x00', '')
            elif isinstance(value, (int, float)):
                sanitized[key] = value
            elif isinstance(value, bool):
                sanitized[key] = value
        return sanitized


def validate_question(question: str) -> str:
    """Convenience function to validate question."""
    return InputValidator.validate_question(question)


def validate_filename(filename: str) -> str:
    """Convenience function to validate filename."""
    return InputValidator.validate_filename(filename)
