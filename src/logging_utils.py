"""
Secure Logging Utilities
Provides redaction for sensitive data in logs.
"""

import logging
import re
from typing import Any


class RedactingFilter(logging.Filter):
    """Filter to redact sensitive information from log messages."""
    
    # Patterns to redact
    PATTERNS = [
        (re.compile(r'sk-[a-zA-Z0-9]{20,}'), '[REDACTED_API_KEY]'),
        (re.compile(r'gsk_[a-zA-Z0-9]{20,}'), '[REDACTED_API_KEY]'),
        (re.compile(r'Bearer\s+[^\s]+', re.IGNORECASE), 'Bearer [REDACTED]'),
        (re.compile(r'api[_-]?key["\']?\s*[:=]\s*["\']?[^\s"\',}]+', re.IGNORECASE), 'api_key=[REDACTED]'),
        (re.compile(r'token["\']?\s*[:=]\s*["\']?[^\s"\',}]+', re.IGNORECASE), 'token=[REDACTED]'),
        (re.compile(r'password["\']?\s*[:=]\s*["\']?[^\s"\',}]+', re.IGNORECASE), 'password=[REDACTED]'),
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Redact message
        if isinstance(record.msg, str):
            msg = record.msg
            for pattern, replacement in self.PATTERNS:
                msg = pattern.sub(replacement, msg)
            record.msg = msg
        
        # Redact args
        if record.args:
            new_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    s = arg
                    for pattern, replacement in self.PATTERNS:
                        s = pattern.sub(replacement, s)
                    new_args.append(s)
                else:
                    new_args.append(arg)
            record.args = tuple(new_args)
        
        return True


def setup_secure_logging(name: str = None) -> logging.Logger:
    """
    Setup logger with redaction filter.
    
    Args:
        name: Logger name.
    
    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    # Add filter if not already added
    if not any(isinstance(f, RedactingFilter) for f in logger.filters):
        logger.addFilter(RedactingFilter())
    return logger
