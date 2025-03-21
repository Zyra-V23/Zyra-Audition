"""
Conversation Analytics Package

A Python package for analyzing conversation data from messaging applications.
The package provides functionality for data processing, text vectorization,
vector similarity analysis, and more.

Author: ZYRAV23
Date: 2025-03-21
"""

__version__ = '0.1.0'

# Import main classes for easier access
from .api.conversation_analyzer import ConversationAnalyzer
from .core.data_processor import DataProcessor
from .core.text_vectorizer import TextVectorizer, AdaptiveTextVectorizer
from .core.vector_similarity import SimilarityIndex, VectorClusterer

# Import utility functions
try:
    from .utils.visualizations import plot_similarity_matrix, plot_clusters
except ImportError:
    pass  # Optional visualization functionality
