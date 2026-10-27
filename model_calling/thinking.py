"""
Thinking mode capabilities and utilities.
"""
from enum import Enum


class ThinkingStyle(Enum):
    """Different styles of thinking mode supported by models"""
    NONE = "none"
    GRANITE = "granite"  # Uses <think></think> tags
    # Add more styles as needed, e.g.:
    # MISTRAL = "mistral"  # If Mistral uses a different format
