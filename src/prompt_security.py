"""
Prompt Injection Resistance
Utilities to detect and sanitize potential prompt injection attempts.
"""

import re
from typing import Tuple


class PromptInjectionDetector:
    """Detects potential prompt injection patterns."""
    
    # Patterns that may indicate prompt injection attempts
    INJECTION_PATTERNS = [
        r'ignore\s+(?:previous|above|all)\s+.*(?:instructions?|prompts?|rules?)',
        r'\bsystem\s*:',
        r'<\s*system\s*>',
        r'\[\s*system\s*\]',
        r'forget\s+(?:everything|all|previous)',
        r'new\s+instructions?',
        r'you\s+are\s+now',
        r'act\s+as\s+(?!a\s+helpful\b)',
        r'jailbreak',
        r'DAN\s+mode',
        r'prompt\s+leak',
        r'reveal\s+(?:your|the)\s+(?:prompt|instructions)',
    ]
    
    @classmethod
    def detect_injection(cls, text: str) -> Tuple[bool, list]:
        """
        Detect potential prompt injection in text.
        
        Args:
            text: Input text to check.
        
        Returns:
            Tuple of (is_suspicious, list of matched patterns)
        """
        if not text:
            return False, []
        
        text_lower = text.lower()
        matches = []
        
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                matches.append(pattern)
        
        return len(matches) > 0, matches
    
    @classmethod
    def sanitize_for_prompt(cls, text: str) -> str:
        """
        Sanitize text to reduce prompt injection risk.
        This is defense in depth - not a complete solution.
        
        Args:
            text: Text to sanitize.
        
        Returns:
            Sanitized text.
        """
        if not text:
            return text
        
        # Remove potential role markers
        sanitized = text
        # Replace system/user/assistant role markers
        sanitized = re.sub(r'(?i)^\s*(system|user|assistant)\s*:?\s*', '', sanitized, flags=re.MULTILINE)
        sanitized = re.sub(r'(?i)<\s*(system|user|assistant)\s*>', '', sanitized)
        sanitized = re.sub(r'(?i)\[\s*(system|user|assistant)\s*\]', '', sanitized)
        
        # Limit consecutive newlines
        sanitized = re.sub(r'\n{3,}', '\n\n', sanitized)
        
        return sanitized.strip()


def check_injection(text: str) -> Tuple[bool, list]:
    """Convenience function."""
    return PromptInjectionDetector.detect_injection(text)


def sanitize_prompt_input(text: str) -> str:
    """Convenience function."""
    return PromptInjectionDetector.sanitize_for_prompt(text)
