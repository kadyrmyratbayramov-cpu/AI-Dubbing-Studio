"""Text processing utility functions."""

import re
from typing import List, Tuple


class TextUtils:
    """Text processing and manipulation utilities."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into words.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        return text.split()

    @staticmethod
    def normalize_punctuation(text: str) -> str:
        """Normalize punctuation in text.

        Args:
            text: Input text

        Returns:
            Text with normalized punctuation
        """
        # Replace multiple punctuation with single
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        return text
