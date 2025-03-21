#!/usr/bin/env python3
"""
Text Vectorization Module for Conversation Analytics

This module provides text vectorization functionality using various methods:
- Bag of Words (BoW)
- TF-IDF
- Transformer-based embeddings

Features include adaptive vectorization based on text characteristics and
an intelligent caching system for improved performance.

Authors: Zyra V23 and Zyxel 7B
Date: 2025-03-21
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple
import pickle
import time
import hashlib
from pathlib import Path
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("Warning: sentence-transformers not available. Some vectorization methods will be disabled.")

class IntelligentVectorCache:
    """
    Intelligent caching system for text vectors.
    
    Features:
    - Frequency-based retention policy
    - Time-based aging
    - Automatic pruning
    - Usage statistics
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600,
                 prune_threshold: float = 0.8):
        """
        Initialize the cache with configuration parameters.
        
        Args:
            max_size (int): Maximum number of vectors to store
            ttl (int): Time-to-live for cache entries in seconds
            prune_threshold (float): When cache reaches this % of max_size, prune oldest entries
        """
        self.cache = {}  # Hash -> vector mapping
        self.access_count = {}  # Hash -> access count
        self.last_access = {}  # Hash -> last access timestamp
        self.creation_time = {}  # Hash -> creation timestamp
        self.max_size = max_size
        self.ttl = ttl
        self.prune_threshold = prune_threshold
        self.hits = 0
        self.misses = 0
        
    def _compute_hash(self, text: str) -> str:
        """
        Compute a hash for the given text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Hash string
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
        
    def get(self, text: str) -> tuple:
        """
        Get vector from cache if it exists.
        
        Args:
            text (str): Text to look up
            
        Returns:
            tuple: (vector, cache_hit) where vector is the cached vector
                  or None if not cached, and cache_hit is a boolean
        """
        text_hash = self._compute_hash(text)
        current_time = time.time()
        
        # Check if in cache and not expired
        if text_hash in self.cache:
            if current_time - self.creation_time[text_hash] < self.ttl:
                # Update stats
                self.access_count[text_hash] += 1
                self.last_access[text_hash] = current_time
                self.hits += 1
                return self.cache[text_hash], True
            else:
                # Remove expired entry
                self._remove_entry(text_hash)
                
        # Cache miss
        self.misses += 1
        return None, False
        
    def put(self, text: str, vector: np.ndarray) -> None:
        """
        Store vector in cache.
        
        Args:
            text (str): Text associated with the vector
            vector (np.ndarray): Vector to cache
        """
        text_hash = self._compute_hash(text)
        current_time = time.time()
        
        # If cache is getting full, prune it
        if len(self.cache) >= int(self.max_size * self.prune_threshold):
            self._prune()
            
        # Add to cache
        self.cache[text_hash] = vector
        self.access_count[text_hash] = 1
        self.last_access[text_hash] = current_time
        self.creation_time[text_hash] = current_time
        
    def _remove_entry(self, text_hash: str) -> None:
        """
        Remove a single entry from cache.
        
        Args:
            text_hash (str): Hash of the entry to remove
        """
        if text_hash in self.cache:
            del self.cache[text_hash]
            del self.access_count[text_hash]
            del self.last_access[text_hash]
            del self.creation_time[text_hash]
        
    def _prune(self) -> None:
        """
        Prune cache using a scoring algorithm that considers frequency and recency.
        """
        if not self.cache:
            return
            
        current_time = time.time()
        scores = {}
        
        # Calculate scores for each entry (higher is better to keep)
        for text_hash in self.cache:
            # Compute score based on frequency and recency
            frequency_score = self.access_count[text_hash]
            recency_score = 1 / (current_time - self.last_access[text_hash] + 1)
            scores[text_hash] = 0.7 * frequency_score + 0.3 * recency_score
            
        # Sort by score (ascending)
        sorted_hashes = sorted(scores.keys(), key=lambda h: scores[h])
        
        # Remove 30% of the lowest scoring entries
        entries_to_remove = int(len(sorted_hashes) * 0.3)
        for text_hash in sorted_hashes[:entries_to_remove]:
            self._remove_entry(text_hash)
            
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            dict: Cache statistics
        """
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
            'hits': self.hits,
            'misses': self.misses,
            'most_frequent': self._get_most_frequent(5),
            'memory_usage_estimate': sum(v.nbytes for v in self.cache.values()) if self.cache else 0
        }
        
    def _get_most_frequent(self, n: int = 5) -> list:
        """
        Get the most frequently accessed cache entries.
        
        Args:
            n (int): Number of entries to return
            
        Returns:
            list: List of (hash, count) tuples
        """
        if not self.access_count:
            return []
            
        # Sort by access count (descending)
        return sorted(
            self.access_count.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
        
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_count.clear()
        self.last_access.clear()
        self.creation_time.clear()
        # Keep stats for information purposes

class BaseVectorizer:
    """Base class for text vectorization."""
    
    def __init__(self):
        """Initialize the base vectorizer."""
        pass
        
    def fit(self, texts: List[str]) -> None:
        """
        Fit the vectorizer on training texts.
        
        Args:
            texts (List[str]): List of text documents to fit on
        """
        raise NotImplementedError
        
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts into vectors.
        
        Args:
            texts (List[str]): List of text documents to transform
            
        Returns:
            np.ndarray: Document vectors
        """
        raise NotImplementedError
        
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit the vectorizer and transform the texts in one step.
        
        Args:
            texts (List[str]): List of text documents
            
        Returns:
            np.ndarray: Document vectors
        """
        self.fit(texts)
        return self.transform(texts)

class TextVectorizer(BaseVectorizer):
    """Class for text vectorization using various methods."""
    
    VECTORIZATION_METHODS = ['bow', 'tfidf', 'transformer']
    
    def __init__(self, method: str = 'tfidf', max_features: int = 1000,
                transformer_model: str = 'all-MiniLM-L6-v2',
                normalize: bool = True, use_idf: bool = True,
                min_df: int = 2, max_df: float = 0.95):
        """
        Initialize the text vectorizer.
        
        Args:
            method (str): Vectorization method ('bow', 'tfidf', or 'transformer')
            max_features (int): Maximum number of features for BoW/TF-IDF
            transformer_model (str): Name of the transformer model to use
            normalize (bool): Whether to normalize vectors
            use_idf (bool): Use Inverse Document Frequency weighting
            min_df (int): Minimum document frequency for terms
            max_df (float): Maximum document frequency for terms (0-1)
        """
        super().__init__()
        if method not in self.VECTORIZATION_METHODS:
            raise ValueError(f"Method must be one of {self.VECTORIZATION_METHODS}")
        
        if method == 'transformer' and not TRANSFORMER_AVAILABLE:
            print("Warning: Transformer method requested but not available. Falling back to TF-IDF.")
            method = 'tfidf'
            
        self.method = method
        self.max_features = max_features
        self.transformer_model = transformer_model
        self.normalize = normalize
        self.use_idf = use_idf
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None
        self.feature_names = None
        self.vocab_size = None
        
        if method == 'bow':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                min_df=min_df,
                max_df=max_df
            )
        elif method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                use_idf=use_idf,
                min_df=min_df,
                max_df=max_df,
                norm='l2' if normalize else None
            )
        elif method == 'transformer' and TRANSFORMER_AVAILABLE:
            try:
                self.vectorizer = SentenceTransformer(transformer_model)
            except Exception as e:
                print(f"Error loading transformer model: {e}")
                print("Falling back to TF-IDF vectorization.")
                self.method = 'tfidf'
                self.vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    use_idf=use_idf,
                    min_df=min_df,
                    max_df=max_df,
                    norm='l2' if normalize else None
                )
            
    def fit(self, texts: List[str]) -> None:
        """
        Fit the vectorizer on the training texts.
        
        Args:
            texts (List[str]): List of text documents to fit on
        """
        if self.method in ['bow', 'tfidf']:
            self.vectorizer.fit(texts)
            self.feature_names = self.vectorizer.get_feature_names_out()
            self.vocab_size = len(self.feature_names)
        # Transformer models don't need fitting
            
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts into vectors.
        
        Args:
            texts (List[str]): List of text documents to transform
            
        Returns:
            np.ndarray: Document vectors
        """
        if self.method in ['bow', 'tfidf']:
            return self.vectorizer.transform(texts).toarray()
        else:
            return self.vectorizer.encode(texts)
            
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit the vectorizer and transform the texts in one step.
        
        Args:
            texts (List[str]): List of text documents
            
        Returns:
            np.ndarray: Document vectors
        """
        if self.method in ['bow', 'tfidf']:
            vectors = self.vectorizer.fit_transform(texts).toarray()
            self.feature_names = self.vectorizer.get_feature_names_out()
            self.vocab_size = len(self.feature_names)
            return vectors
        else:
            return self.transform(texts)
            
    def get_feature_names(self) -> Optional[List[str]]:
        """Get feature names for BoW/TF-IDF vectorization."""
        return self.feature_names if self.method in ['bow', 'tfidf'] else None
        
    def get_top_features(self, vector: np.ndarray, n: int = 10) -> List[Dict[str, float]]:
        """
        Get top N features for a given vector (only for BoW/TF-IDF).
        
        Args:
            vector (np.ndarray): Input vector
            n (int): Number of top features to return
            
        Returns:
            List[Dict[str, float]]: List of top features with their scores
        """
        if self.method not in ['bow', 'tfidf'] or self.feature_names is None:
            raise ValueError("Top features only available for BoW/TF-IDF")
            
        top_indices = np.argsort(vector)[-n:][::-1]
        return [
            {'feature': self.feature_names[i], 'score': vector[i]}
            for i in top_indices
        ]
        
    def save_model(self, path: str) -> None:
        """
        Save the vectorizer model to disk.
        
        Args:
            path (str): Path to save the model
        """
        model_data = {
            'method': self.method,
            'max_features': self.max_features,
            'transformer_model': self.transformer_model,
            'vectorizer': self.vectorizer,
            'feature_names': self.feature_names,
            'vocab_size': self.vocab_size,
            'normalize': self.normalize,
            'use_idf': self.use_idf,
            'min_df': self.min_df,
            'max_df': self.max_df
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
            
    @classmethod
    def load_model(cls, path: str) -> 'TextVectorizer':
        """
        Load a saved vectorizer model.
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            TextVectorizer: Loaded vectorizer instance
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            
        instance = cls(
            method=model_data['method'],
            max_features=model_data['max_features'],
            transformer_model=model_data['transformer_model'],
            normalize=model_data['normalize'],
            use_idf=model_data['use_idf'],
            min_df=model_data['min_df'],
            max_df=model_data['max_df']
        )
        
        instance.vectorizer = model_data['vectorizer']
        instance.feature_names = model_data['feature_names']
        instance.vocab_size = model_data['vocab_size']
        
        return instance

class AdaptiveTextVectorizer:
    """
    Adaptive text vectorization class that automatically selects the best method
    based on text characteristics. Features intelligent selection between different
    vectorization approaches depending on text length and complexity.
    """
    
    def __init__(self, cache_size: int = 1000, cache_ttl: int = 3600,
                normalize: bool = True, use_idf: bool = True,
                min_df: int = 2, max_df: float = 0.95):
        """
        Initialize vectorizers and thresholds for adaptive text processing.
        
        Args:
            cache_size (int): Maximum number of vectors to cache
            cache_ttl (int): Time-to-live for cache entries in seconds
            normalize (bool): Whether to normalize vectors
            use_idf (bool): Use Inverse Document Frequency weighting
            min_df (int): Minimum document frequency for terms
            max_df (float): Maximum document frequency for terms (0-1)
        """
        self.vectorizers = {
            'short_simple': TextVectorizer(
                method='tfidf',
                max_features=500,
                normalize=normalize,
                use_idf=use_idf,
                min_df=min_df,
                max_df=max_df
            ),
            'short_complex': TextVectorizer(
                method='transformer',
                transformer_model='all-MiniLM-L6-v2',
                normalize=normalize
            ),
            'long_simple': TextVectorizer(
                method='tfidf',
                max_features=1000,
                normalize=normalize,
                use_idf=use_idf,
                min_df=min_df,
                max_df=max_df
            ),
            'long_complex': TextVectorizer(
                method='transformer',
                transformer_model='all-mpnet-base-v2',
                normalize=normalize
            )
        }
        
        # Thresholds for text classification
        self.length_threshold = 100  # characters
        self.complexity_features = {
            'technical_terms': set(['blockchain', 'cryptocurrency', 'token', 'smart contract', 'protocol']),
            'special_chars': set(['$', '€', '¥', '₿', '%']),
            'code_indicators': set(['function', 'class', 'def', 'return', 'import'])
        }
        
        # Initialize cache system
        self.cache = IntelligentVectorCache(max_size=cache_size, ttl=cache_ttl)
        self.cache_stats = {'lookups': 0, 'hits': 0, 'vectorizations_saved': 0}
        
        # Store vectorization parameters
        self.normalize = normalize
        self.use_idf = use_idf
        self.min_df = min_df
        self.max_df = max_df
        
    def analyze_text_complexity(self, text: str) -> Dict[str, float]:
        """
        Analyze text complexity based on multiple factors, returning detailed metrics.
        
        This method evaluates text complexity using various metrics:
        - Average word length
        - Presence of technical terms
        - Special characters usage
        - Code-like content
        - Sentence structure
        - Vocabulary diversity
        - Question vs statement analysis
        - Conversational tone markers
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Dictionary of complexity metrics
        """
        # Handle empty or very short texts
        if not text or len(text) < 3:
            return {
                'avg_word_length': 0.0,
                'technical_terms': 0.0,
                'special_chars': 0.0,
                'code_presence': 0.0,
                'sentence_complexity': 0.0,
                'vocabulary_diversity': 0.0,
                'question_ratio': 0.0,
                'conversational_tone': 0.0,
                'overall_complexity': 0.0
            }
            
        # Normalize text
        text_lower = text.lower()
        
        # Get words (handle case with no words)
        words = text_lower.split()
        if not words:
            return {
                'avg_word_length': 0.0,
                'technical_terms': 0.0,
                'special_chars': 0.0,
                'code_presence': 0.0,
                'sentence_complexity': 0.0,
                'vocabulary_diversity': 0.0,
                'question_ratio': 0.0,
                'conversational_tone': 0.0,
                'overall_complexity': 0.0
            }
        
        # Calculate metrics safely
        try:
            # Word length metric
            avg_word_length = min(np.mean([len(w) for w in words]) / 10, 1.0)
            
            # Technical term presence
            technical_term_count = sum(term in text_lower for term in self.complexity_features['technical_terms'])
            technical_terms = technical_term_count / max(len(self.complexity_features['technical_terms']), 1)
            
            # Special characters
            special_char_count = sum(char in text for char in self.complexity_features['special_chars'])
            special_chars = special_char_count / max(len(self.complexity_features['special_chars']), 1)
            
            # Code presence
            code_indicator_count = sum(indicator in text_lower for indicator in self.complexity_features['code_indicators'])
            code_presence = code_indicator_count / max(len(self.complexity_features['code_indicators']), 1)
            
            # Sentence complexity (length and structure)
            sentences = text.split('.')
            avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0
            sentence_complexity = min(avg_sentence_length / 15, 1.0)
            
            # Vocabulary diversity (unique words ratio)
            unique_words = set(words)
            vocabulary_diversity = len(unique_words) / len(words)
            
            # Question vs statement analysis
            question_count = sum(text_lower.count(q) for q in ['?', ' why ', ' how ', ' what ', ' when '])
            question_ratio = min(question_count / max(len(sentences), 1), 1.0)
            
            # Conversational tone
            conversation_marker_count = sum(marker in text_lower for marker in self.complexity_features['conversation_markers'])
            conversational_tone = conversation_marker_count / max(len(self.complexity_features['conversation_markers']), 1)
            
            # Compile all metrics
            metrics = {
                'avg_word_length': avg_word_length,
                'technical_terms': technical_terms,
                'special_chars': special_chars,
                'code_presence': code_presence,
                'sentence_complexity': sentence_complexity,
                'vocabulary_diversity': vocabulary_diversity,
                'question_ratio': question_ratio,
                'conversational_tone': conversational_tone
            }
            
            # Calculate overall complexity (weighted average)
            weights = {
                'avg_word_length': 0.1,
                'technical_terms': 0.25,
                'special_chars': 0.05,
                'code_presence': 0.2,
                'sentence_complexity': 0.15,
                'vocabulary_diversity': 0.15,
                'question_ratio': 0.05,
                'conversational_tone': 0.05
            }
            
            overall_complexity = sum(score * weights[metric] for metric, score in metrics.items())
            metrics['overall_complexity'] = min(max(overall_complexity, 0), 1)  # Normalize between 0 and 1
            
            return metrics
            
        except Exception as e:
            print(f"Warning: Error calculating complexity for text: {text[:30]}... - {str(e)}")
            return {
                'avg_word_length': 0.0,
                'technical_terms': 0.0,
                'special_chars': 0.0,
                'code_presence': 0.0,
                'sentence_complexity': 0.0,
                'vocabulary_diversity': 0.0,
                'question_ratio': 0.0,
                'conversational_tone': 0.0,
                'overall_complexity': 0.0
            }
    
    def select_vectorizer_based_on_complexity(self, complexity_score: float) -> TextVectorizer:
        """Select appropriate vectorizer based on text complexity score."""
        if complexity_score > 0.8:
            return self.vectorizers['long_complex']
        elif complexity_score > 0.6:
            return self.vectorizers['long_simple']
        elif complexity_score > 0.4:
            return self.vectorizers['short_complex']
        else:
            return self.vectorizers['short_simple']
            
    def select_vectorizer(self, text: str) -> TextVectorizer:
        """
        Select the best vectorizer based on text characteristics.
        
        The selection is based on:
        - Text length (short/long)
        - Text complexity (simple/complex)
        
        Args:
            text (str): Text to vectorize
            
        Returns:
            TextVectorizer: Selected vectorizer instance
        """
        # Handle very short or empty text
        if not text or len(text) < 5:
            return self.vectorizers['short_simple']
            
        # Get detailed complexity metrics
        complexity_metrics = self.analyze_text_complexity(text)
        overall_complexity = complexity_metrics['overall_complexity']
        
        # Analyze length
        text_length = len(text)
        
        # Adjust complexity based on length
        if text_length < self.length_threshold:
            complexity_adjusted = min(overall_complexity, 0.3)
        else:
            complexity_adjusted = min(overall_complexity * 1.1, 1.0)  # Slight boost for medium texts
            
        # For crypto conversations, prioritize technical term presence
        if complexity_metrics['technical_terms'] > 0.3:
            complexity_adjusted = max(complexity_adjusted, 0.6)  # Ensure at least medium complexity
            
        # For conversational texts with questions, ensure good semantic understanding
        if complexity_metrics['question_ratio'] > 0.5 and complexity_metrics['conversational_tone'] > 0.3:
            complexity_adjusted = max(complexity_adjusted, 0.5)  # Ensure at least medium complexity
            
        return self.select_vectorizer_based_on_complexity(complexity_adjusted)
            
    def fit(self, texts: List[str]) -> None:
        """
        Fit all vectorizers with the training texts.
        
        Args:
            texts (List[str]): List of texts for training
        """
        for vectorizer in self.vectorizers.values():
            vectorizer.fit(texts)
            
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts using the best vectorizer for each one.
        
        This method:
        1. Processes each text individually
        2. Selects the best vectorizer for each text
        3. Normalizes vector dimensions
        4. Handles padding for different vector sizes
        
        Args:
            texts (List[str]): List of texts to transform
            
        Returns:
            np.ndarray: Resulting vector matrix
        """
        if not texts:
            return np.array([])
            
        # Process each text individually with error handling and caching
        vectors = []
        vectorizers_used = []
        cache_hits = 0
        
        for idx, text in enumerate(texts):
            try:
                self.cache_stats['lookups'] += 1
                
                # Check cache first
                if text and isinstance(text, str):
                    cached_vector, cache_hit = self.cache.get(text)
                    if cache_hit:
                        vectors.append(cached_vector)
                        vectorizers_used.append('cache_hit')
                        cache_hits += 1
                        self.cache_stats['hits'] += 1
                        continue
                
                # Handle empty text
                if not text or not isinstance(text, str):
                    # Use simplest vectorizer with empty string
                    vectorizer = self.vectorizers['short_simple']
                    vector = vectorizer.transform([''])[0]
                    vectorizers_used.append('short_simple (empty text)')
                else:
                    # Select and apply appropriate vectorizer
                    vectorizer = self.select_vectorizer(text)
                    vector = vectorizer.transform([text])[0]
                    vectorizers_used.append(next(k for k, v in self.vectorizers.items() if v == vectorizer))
                    
                    # Store in cache
                    self.cache.put(text, vector)
                    
                vectors.append(vector)
                
            except Exception as e:
                print(f"Warning: Error vectorizing text at index {idx}: {str(e)}")
                # Fallback to simplest vectorizer
                vectorizer = self.vectorizers['short_simple']
                vector = vectorizer.transform([''])[0]
                vectors.append(vector)
                vectorizers_used.append('short_simple (fallback)')
        
        # Print vectorizer usage stats
        vectorizer_counts = {}
        for v_name in vectorizers_used:
            vectorizer_counts[v_name] = vectorizer_counts.get(v_name, 0) + 1
        
        print("\nVectorizer usage statistics:")
        for v_name, count in vectorizer_counts.items():
            print(f"  - {v_name}: {count} texts ({count/len(texts)*100:.1f}%)")
            
        # Print cache stats
        self.cache_stats['vectorizations_saved'] += cache_hits
        print(f"\nCache statistics:")
        print(f"  - Hit rate: {self.cache.get_stats()['hit_rate']:.1%}")
        print(f"  - Cache size: {self.cache.get_stats()['size']} / {self.cache.get_stats()['max_size']}")
        print(f"  - Total lookups: {self.cache_stats['lookups']}")
        print(f"  - Total hits: {self.cache_stats['hits']}")
        print(f"  - Vectorizations saved: {self.cache_stats['vectorizations_saved']}")
        
        # Normalize dimensions if necessary
        if not vectors:
            return np.array([])
            
        try:
            # Handle case where vectors have different dimensions
            vector_dims = [v.shape[0] for v in vectors]
            if len(set(vector_dims)) > 1:
                print(f"Note: Vectors have varying dimensions {set(vector_dims)}. Normalizing to max dimension.")
                
            max_dim = max(vector_dims)
            normalized_vectors = []
            
            for vector in vectors:
                if vector.shape[0] < max_dim:
                    # Pad with zeros if necessary
                    padded = np.zeros(max_dim)
                    padded[:vector.shape[0]] = vector
                    normalized_vectors.append(padded)
                else:
                    normalized_vectors.append(vector)
                    
            return np.array(normalized_vectors)
        except Exception as e:
            print(f"Error normalizing vectors: {str(e)}")
            # Last resort fallback
            return np.zeros((len(texts), 1))
            
    def save_model(self, path: str) -> None:
        """
        Save the adaptive model to disk.
        
        Args:
            path (str): Path to save the model
        """
        # Get cache stats before saving
        cache_stats = self.cache.get_stats()
        
        model_data = {
            'vectorizers': self.vectorizers,
            'length_threshold': self.length_threshold,
            'complexity_features': self.complexity_features,
            'cache_size': self.cache.max_size,
            'cache_ttl': self.cache.ttl,
            'cache_stats': self.cache_stats,
            'cache_contents': self.cache.cache,
            'cache_access_count': self.cache.access_count,
            'cache_last_access': self.cache.last_access,
            'cache_creation_time': self.cache.creation_time,
            'normalize': self.normalize,
            'use_idf': self.use_idf,
            'min_df': self.min_df,
            'max_df': self.max_df
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
            
    @classmethod
    def load_model(cls, path: str) -> 'AdaptiveTextVectorizer':
        """
        Load a saved adaptive model.
        
        Args:
            path (str): Path to the saved model
            
        Returns:
            AdaptiveTextVectorizer: Loaded vectorizer instance
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            
        instance = cls(
            cache_size=model_data['cache_size'],
            cache_ttl=model_data['cache_ttl'],
            normalize=model_data['normalize'],
            use_idf=model_data['use_idf'],
            min_df=model_data['min_df'],
            max_df=model_data['max_df']
        )
        
        instance.vectorizers = model_data['vectorizers']
        instance.length_threshold = model_data['length_threshold']
        instance.complexity_features = model_data['complexity_features']
        instance.cache_stats = model_data['cache_stats']
        
        # Restore cache if available
        if 'cache_contents' in model_data:
            instance.cache.cache = model_data['cache_contents']
            instance.cache.access_count = model_data['cache_access_count']
            instance.cache.last_access = model_data['cache_last_access']
            instance.cache.creation_time = model_data['cache_creation_time']
        
        return instance

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit vectorizers and transform texts in one step.
        
        Args:
            texts (List[str]): List of texts
            
        Returns:
            np.ndarray: Resulting vector matrix
        """
        # Fit all vectorizers first
        for vectorizer in self.vectorizers.values():
            vectorizer.fit(texts)
            
        # Then transform using the best vectorizer for each text
        return self.transform(texts) 