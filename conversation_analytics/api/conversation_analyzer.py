#!/usr/bin/env python3
"""
Conversation Analyzer API for Conversation Analytics

This module provides a high-level API for analyzing conversation data using
the core functionality of the conversation analytics package. It integrates
data processing, text vectorization, and vector similarity analysis.

Authors: Zyra V23 and Zyxel 7B
Date: 2025-03-21
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any
import pickle
from pathlib import Path
import datetime
import json
import time
import random
import logging
import yaml

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("Warning: sentence-transformers not available. Some features will be disabled.")

# Import core modules
try:
    from ..core.data_processor import DataProcessor
    from ..core.text_vectorizer import AdaptiveTextVectorizer, TextVectorizer
    from ..core.vector_similarity import SimilarityIndex, VectorClusterer, ConversationClusterer, batch_process_vectors, find_similar_texts
except ImportError:
    # For standalone usage
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.data_processor import DataProcessor
    from core.text_vectorizer import AdaptiveTextVectorizer, TextVectorizer
    from core.vector_similarity import SimilarityIndex, VectorClusterer, ConversationClusterer, batch_process_vectors, find_similar_texts


class ConversationAnalyzer:
    """
    High-level API for analyzing conversation data.
    
    This class provides a unified interface to the conversation analytics
    functionality, including data processing, text vectorization, and
    vector similarity analysis.
    """
    
    def __init__(self, output_dir: str = "conversation_analytics_output",
                cache_size: int = 1000, config_path: Optional[str] = None):
        """
        Initialize the conversation analyzer.
        
        Args:
            output_dir (str): Directory to store outputs
            cache_size (int): Size of the adaptive vectorizer cache
            config_path (str, optional): Path to YAML configuration file
        """
        self.output_dir = output_dir
        self.cache_size = cache_size
        self.config_path = config_path
        
        # Load configuration if provided
        self.config = {}
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.data_processor = None
        self.vectorizer = None
        self.similarity_index = None
        self.clusterer = None
        
        # Data containers
        self.data = None
        self.vectors = None
        self.processed_data_path = None
        self.vectorization_method = None
        self.similarity_metric = 'cosine'
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load conversation data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        self.data_processor = DataProcessor()
        self.data = self.data_processor.load_data(file_path)
        return self.data
        
    def process_data(self, remove_stopwords: bool = True,
                    spam_threshold: float = 0.8) -> Tuple[pd.DataFrame, Dict]:
        """
        Process the loaded conversation data.
        
        Args:
            remove_stopwords (bool): Whether to remove stopwords during tokenization
            spam_threshold (float): Threshold for spam detection
            
        Returns:
            Tuple[pd.DataFrame, Dict]: Processed data and processing report
        """
        if self.data_processor is None:
            raise ValueError("No data loaded. Call load_data first.")
            
        # Clean data
        clean_data, clean_report = self.data_processor.clean_data()
        
        # Tokenize text
        tokenized_data = self.data_processor.tokenize_text(remove_stopwords=remove_stopwords)
        
        # Detect spam
        processed_data = self.data_processor.detect_spam(threshold=spam_threshold)
        
        # Save processed data
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.processed_data_path = os.path.join(
            self.output_dir,
            f"processed_messages_{timestamp}.csv"
        )
        self.data_processor.save_processed_data(self.processed_data_path)
        
        # Update data reference
        self.data = processed_data
        
        return processed_data, clean_report
        
    def vectorize_texts(self, method: str = 'adaptive',
                       load_existing: Optional[str] = None,
                       model_name: Optional[str] = None,
                       normalize: bool = True,
                       use_idf: bool = True,
                       min_df: int = 2,
                       max_df: float = 0.95) -> np.ndarray:
        """
        Convert processed text data into vectors.
        
        Args:
            method (str): Vectorization method ('bow', 'tfidf', 'transformer', or 'adaptive')
            load_existing (str): Path to existing model to load
            model_name (str): Name of the transformer model to use (for transformer method)
            normalize (bool): Whether to normalize vectors
            use_idf (bool): Use Inverse Document Frequency weighting
            min_df (int): Minimum document frequency for terms
            max_df (float): Maximum document frequency for terms (0-1)
            
        Returns:
            np.ndarray: Text vectors
        """
        if self.data is None:
            raise ValueError("No data available. Load and process data first.")
            
        if 'clean_text' not in self.data.columns:
            raise ValueError("Data doesn't contain 'clean_text' column. Process data first.")
            
        # Get texts to vectorize
        texts = self.data['clean_text'].fillna('').tolist()
        
        # Store vectorization method
        self.vectorization_method = method
        
        # Create vectorizer based on method
        if method == 'adaptive':
            if load_existing:
                print(f"Loading existing adaptive vectorizer from {load_existing}")
                self.vectorizer = AdaptiveTextVectorizer.load_model(load_existing)
            else:
                print("Initializing adaptive vectorizer...")
                self.vectorizer = AdaptiveTextVectorizer(
                    cache_size=self.cache_size,
                    normalize=normalize,
                    use_idf=use_idf,
                    min_df=min_df,
                    max_df=max_df
                )
                
                # Custom specialized models for conversation analysis
                if TRANSFORMER_AVAILABLE:
                    print("Using transformer-based embeddings for complex text")
                    # Override default vectorizers with conversation-optimized models
                    self.vectorizer.vectorizers['medium'] = TextVectorizer(
                        method='transformer',
                        transformer_model='paraphrase-MiniLM-L6-v2',  # Better for conversational text
                        normalize=normalize
                    )
                    self.vectorizer.vectorizers['complex'] = TextVectorizer(
                        method='transformer',
                        transformer_model='paraphrase-multilingual-MiniLM-L12-v2',  # For multilingual support
                        normalize=normalize
                    )
                    
                    # Check if specialized conversation models are requested
                    use_specialized = True
                    if use_specialized:
                        try:
                            # Add specialized model for conversation understanding
                            print("Adding specialized conversation embeddings model")
                            self.vectorizer.vectorizers['conversation'] = TextVectorizer(
                                method='transformer', 
                                transformer_model='sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
                                normalize=normalize
                            )
                            
                            # Override selector to use conversation model for appropriate texts
                            original_selector = self.vectorizer.select_vectorizer
                            
                            def conversation_enhanced_selector(text):
                                # Get detailed metrics and original selection
                                metrics = self.vectorizer.analyze_text_complexity(text)
                                
                                # For highly conversational text with questions, use the conversation model
                                if (metrics['conversational_tone'] > 0.4 and 
                                    metrics['question_ratio'] > 0.3 and
                                    len(text) > 40):
                                    return self.vectorizer.vectorizers['conversation']
                                    
                                # Otherwise use the normal selection logic
                                return original_selector(text)
                                
                            # Replace the selector method
                            self.vectorizer.select_vectorizer = conversation_enhanced_selector
                            
                        except Exception as e:
                            print(f"Warning: Could not set up specialized conversation model: {str(e)}")
                            
        elif method in ['bow', 'tfidf']:
            self.vectorizer = TextVectorizer(
                method=method,
                normalize=normalize,
                use_idf=use_idf,
                min_df=min_df,
                max_df=max_df
            )
        elif method == 'transformer':
            if not TRANSFORMER_AVAILABLE:
                raise ValueError("Transformer-based vectorization not available. Install sentence-transformers.")
            self.vectorizer = TextVectorizer(
                method='transformer',
                transformer_model=model_name or 'all-MiniLM-L6-v2',
                normalize=normalize
            )
        else:
            raise ValueError(f"Unknown vectorization method: {method}")
            
        # Vectorize texts
        print(f"Vectorizing {len(texts)} texts...")
        self.vectors = self.vectorizer.fit_transform(texts)
        
        # Save vectors
        vector_path = os.path.join(self.output_dir, f"text_vectors_{method}.npy")
        np.save(vector_path, self.vectors)
        print(f"Vectors saved to {vector_path}")
        
        return self.vectors
        
    def analyze_similarity(self, n_clusters: int = 3,
                          metric: str = 'cosine') -> Dict:
        """
        Analyze similarities between vectorized texts.
        
        Args:
            n_clusters (int): Number of clusters for clustering
            metric (str): Similarity metric to use
            
        Returns:
            Dict: Analysis results
        """
        if self.vectors is None:
            raise ValueError("No vectors available. Vectorize texts first.")
            
        # Store similarity metric
        self.similarity_metric = metric
        
        # Create similarity output directory
        similarity_dir = os.path.join(self.output_dir, "similarity")
        os.makedirs(similarity_dir, exist_ok=True)
        
        # Process vectors in batch
        results = batch_process_vectors(
            vectors=self.vectors,
            metric=metric,
            n_clusters=n_clusters,
            output_dir=similarity_dir
        )
        
        # Create analysis report
        report = {
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data_source': self.processed_data_path,
            'vectorization_method': self.vectorization_method,
            'similarity_metric': metric,
            'vector_shape': self.vectors.shape,
            'n_clusters': n_clusters,
            'similarity_stats': results['similarity_stats'],
            'cluster_stats': {str(k): v for k, v in results['cluster_stats'].items()},
            'processing_time': results['processing_time']
        }
        
        # Save report
        report_path = os.path.join(similarity_dir, "similarity_analysis_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
            
        return report
        
    def find_similar_messages(self, query_text: str,
                            k: int = 5,
                            min_similarity: float = 0.3) -> List[Dict]:
        """
        Find messages similar to a query text.
        
        Args:
            query_text (str): Query text
            k (int): Number of results to return
            min_similarity (float): Minimum similarity threshold
            
        Returns:
            List[Dict]: List of similar messages with metadata
        """
        if self.vectors is None or self.vectorizer is None:
            raise ValueError("No vectors or vectorizer available. Vectorize texts first.")
            
        if self.data is None:
            raise ValueError("No data available. Load and process data first.")
            
        # Vectorize query text
        query_vector = self.vectorizer.transform([query_text])[0]
        
        # Get texts for results
        texts = self.data['clean_text'].fillna('').tolist()
        
        # Find similar texts
        similar_texts = find_similar_texts(
            query_vector=query_vector,
            vectors=self.vectors,
            texts=texts,
            metric=self.similarity_metric,
            k=k,
            min_similarity=min_similarity
        )
        
        # Enhance results with message metadata
        for result in similar_texts:
            idx = result['index']
            if idx < len(self.data):
                message_data = self.data.iloc[idx]
                result['message'] = {
                    'original_text': message_data.get('text', ''),
                    'clean_text': message_data.get('clean_text', ''),
                    'sender': message_data.get('sender', ''),
                    'timestamp': message_data.get('timestamp', ''),
                    'is_spam': message_data.get('is_spam', False)
                }
                
        return similar_texts
        
    def analyze_conversation(self, file_path: str, 
                           vectorization_method: str = 'adaptive',
                           n_clusters: int = 3,
                           similarity_metric: str = 'cosine') -> Dict:
        """
        Perform a complete conversation analysis workflow.
        
        Args:
            file_path (str): Path to conversation data file
            vectorization_method (str): Method for text vectorization
            n_clusters (int): Number of clusters for similarity analysis
            similarity_metric (str): Metric for similarity calculation
            
        Returns:
            Dict: Analysis results summary
        """
        # Start timing
        start_time = datetime.datetime.now()
        
        # Load data
        print(f"Loading data from {file_path}")
        self.load_data(file_path)
        
        # Process data
        print("Processing conversation data")
        processed_data, clean_report = self.process_data()
        
        # Vectorize texts
        print(f"Vectorizing texts using {vectorization_method} method")
        self.vectorize_texts(method=vectorization_method)
        
        # Analyze similarity
        print(f"Analyzing similarities using {similarity_metric} metric")
        similarity_report = self.analyze_similarity(
            n_clusters=n_clusters,
            metric=similarity_metric
        )
        
        # Calculate time taken
        end_time = datetime.datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Prepare summary report
        summary = {
            'data_file': file_path,
            'output_directory': self.output_dir,
            'record_count': len(self.data),
            'processing_time': processing_time,
            'cleaning_report': clean_report,
            'vectorization': {
                'method': vectorization_method,
                'vector_shape': self.vectors.shape
            },
            'similarity_analysis': similarity_report
        }
        
        # Save summary report
        summary_path = os.path.join(self.output_dir, "analysis_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
            
        print(f"Analysis complete. Results saved to {self.output_dir}")
        print(f"Total processing time: {processing_time:.2f} seconds")
        
        return summary
        
    def save_state(self, path: str) -> None:
        """
        Save the analyzer state to disk.
        
        Args:
            path (str): Path to save the state
        """
        state = {
            'output_dir': self.output_dir,
            'cache_size': self.cache_size,
            'processed_data_path': self.processed_data_path,
            'vectorization_method': self.vectorization_method,
            'similarity_metric': self.similarity_metric
        }
        
        # Save state data
        with open(path, 'wb') as f:
            pickle.dump(state, f)
            
        print(f"Analyzer state saved to {path}")
        
    @classmethod
    def load_state(cls, path: str) -> 'ConversationAnalyzer':
        """
        Load analyzer state from disk.
        
        Args:
            path (str): Path to saved state
            
        Returns:
            ConversationAnalyzer: Loaded analyzer instance
        """
        # Load state data
        with open(path, 'rb') as f:
            state = pickle.load(f)
            
        # Create new instance with saved parameters
        instance = cls(
            output_dir=state['output_dir'],
            cache_size=state['cache_size']
        )
        
        # Restore state
        instance.processed_data_path = state['processed_data_path']
        instance.vectorization_method = state['vectorization_method']
        instance.similarity_metric = state['similarity_metric']
        
        # Load data if available
        if instance.processed_data_path and os.path.exists(instance.processed_data_path):
            instance.data_processor = DataProcessor()
            instance.data = instance.data_processor.load_data(instance.processed_data_path)
            
        print(f"Analyzer state loaded from {path}")
        return instance 

    def analyze_conversations(self, input_file: str, output_dir: str) -> pd.DataFrame:
        """
        Analyze and cluster messages into conversations using both semantic and temporal information.
        
        This method uses advanced clustering techniques specifically designed for conversation data,
        taking into account both the semantic similarity between messages and their temporal proximity.
        
        Args:
            input_file (str): Path to input CSV file
            output_dir (str): Directory to store output files
            
        Returns:
            pd.DataFrame: Data with conversation IDs and confidence scores
        """
        # Load data
        print(f"\nLoading data from {input_file}")
        self.load_data(input_file)
        
        # Process data
        print("\nProcessing conversation data")
        processed_data, clean_report = self.process_data()
        
        # Get configuration parameters
        hdbscan_config = self.config.get('hdbscan', {})
        temporal_config = self.config.get('temporal', {})
        vector_config = self.config.get('vectors', {})
        
        # HDBSCAN parameters
        min_cluster_size = hdbscan_config.get('min_cluster_size', 3)
        min_samples = hdbscan_config.get('min_samples', 2)
        cluster_selection_epsilon = hdbscan_config.get('cluster_selection_epsilon', 0.3)
        
        # Temporal parameters
        time_weight = temporal_config.get('time_weight', 0.4)
        max_time_gap = temporal_config.get('max_time_gap', 3600)
        similarity_threshold = temporal_config.get('similarity_threshold', 0.65)
        
        # Vector parameters
        normalize = vector_config.get('normalize', True)
        use_idf = vector_config.get('use_idf', True)
        min_df = vector_config.get('min_df', 2)
        max_df = vector_config.get('max_df', 0.95)
        
        # Vectorize texts using configuration
        vectorization_config = self.config.get('vectorization', {})
        method = vectorization_config.get('method', 'adaptive')
        print(f"\nVectorizing texts using {method} method")
        self.vectorize_texts(
            method=method,
            normalize=normalize,
            use_idf=use_idf,
            min_df=min_df,
            max_df=max_df
        )
        
        if self.vectors is None:
            raise ValueError("No vectors available. Vectorization failed.")
            
        print("\nClustering messages into conversations...")
        
        # Convert timestamps to epoch seconds for temporal analysis
        if 'datetime' not in self.data.columns:
            if 'Timestamp' in self.data.columns:
                self.data['datetime'] = pd.to_datetime(self.data['Timestamp'])
            else:
                raise ValueError("No timestamp information available in data")
                
        # Get timestamps in seconds since epoch
        timestamps = self.data['datetime'].apply(lambda x: int(x.timestamp())).tolist()
        
        # Get message IDs
        message_ids = self.data['ID'].tolist() if 'ID' in self.data.columns else None
        
        # Get original texts for analysis
        texts = self.data['clean_text'].fillna('').tolist()
        
        # Create clusterer with specified parameters
        clusterer = ConversationClusterer(
            config_path=self.config_path
        )
        
        # Cluster conversations
        clustering_results = clusterer.cluster_conversations(
            vectors=self.vectors,
            timestamps=timestamps,
            texts=texts,
            message_ids=message_ids
        )
        
        # Extract results
        conversation_ids = clustering_results['labels']
        confidence_scores = clustering_results['confidence_scores']
        num_conversations = len(np.unique([x for x in conversation_ids if x >= 0]))
        cluster_stats = clustering_results['cluster_stats']
        
        # Add results to dataframe
        self.data['conversation_id'] = conversation_ids
        self.data['confidence'] = confidence_scores
        
        # Generate topic labels for each conversation
        topic_labels = self._generate_topic_labels(cluster_stats, texts)
        
        # Add topics to data
        self.data['topic'] = [
            topic_labels.get(cid, "Uncategorized") 
            for cid in conversation_ids
        ]
        
        # Save cluster statistics
        self.conversation_stats = cluster_stats
        
        # Print clustering summary
        print(f"\nClustering completed: {num_conversations} conversations detected")
        print("\nConversation Statistics:")
        for cid, stats in cluster_stats.items():
            if cid == 0:  # Skip spam messages (conversation_id = 0)
                continue
                
            # Format times for readability
            start_time = datetime.datetime.fromtimestamp(stats['start_time']).strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"- Conversation {cid}: {stats['size']} messages, Topic: '{topic_labels.get(cid, 'Unknown')}'")
            print(f"  Started: {start_time}, Duration: {self._format_duration(stats['duration_seconds'])}")
            print(f"  Cohesion: {stats['cohesion']:.3f}")
            
        # Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"conversation_results_{timestamp}.csv")
        
        output_df = self.data[['ID', 'conversation_id', 'topic', 'confidence', 'datetime', 'clean_text']].copy()
        output_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        return self.data
        
    def _generate_topic_labels(self, cluster_stats: Dict, texts: List[str]) -> Dict[int, str]:
        """
        Generate descriptive topic labels for each conversation cluster.
        
        Args:
            cluster_stats (Dict): Statistics for each cluster
            texts (List[str]): Original message texts
            
        Returns:
            Dict[int, str]: Mapping from conversation ID to topic label
        """
        topic_labels = {}
        
        # If we have vectorizer with feature extraction capability
        has_feature_extraction = (hasattr(self.vectorizer, 'get_feature_names') and 
                                self.vectorization_method in ['bow', 'tfidf'])
        
        for cid, stats in cluster_stats.items():
            if cid == 0:  # Skip spam cluster
                topic_labels[cid] = "Spam Messages"
                continue
                
            if len(stats['indices']) == 0:
                topic_labels[cid] = "Empty Conversation"
                continue
                
            # Try to extract key terms for this conversation
            if has_feature_extraction:
                # Get indices of messages in this conversation
                indices = stats['indices']
                
                # Extract vectors for these messages
                if hasattr(self.vectorizer, 'get_top_features'):
                    # For conversation with multiple messages, use centroid
                    if len(indices) > 1:
                        centroid = np.mean(self.vectors[indices], axis=0)
                        top_features = self.vectorizer.get_top_features(centroid, n=5)
                        terms = [f['feature'] for f in top_features]
                    else:
                        # For single message, get features directly
                        top_features = self.vectorizer.get_top_features(self.vectors[indices[0]], n=5)
                        terms = [f['feature'] for f in top_features]
                        
                    # Create topic from terms
                    topic = " ".join(terms[:3]).title()
                    topic_labels[cid] = topic
                else:
                    # Fallback to simple approach
                    topic_labels[cid] = f"Conversation {cid}"
            else:
                # Without feature extraction, use basic approach
                # Extract sample texts from this conversation
                sample_texts = [texts[i] for i in stats['indices'][:3]]
                
                # Extract key words based on frequency
                all_words = []
                for text in sample_texts:
                    all_words.extend(text.lower().split())
                    
                # Count word frequencies
                word_counts = {}
                for word in all_words:
                    if len(word) > 3:  # Filter short words
                        word_counts[word] = word_counts.get(word, 0) + 1
                        
                # Sort by count
                sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
                
                # Get top words
                top_words = [word for word, count in sorted_words[:3] if count > 1]
                
                if top_words:
                    topic = " ".join(top_words).title()
                    topic_labels[cid] = topic
                else:
                    topic_labels[cid] = f"Conversation {cid}"
                    
        return topic_labels
        
    def _format_duration(self, seconds: int) -> str:
        """
        Format duration in seconds to a readable string.
        
        Args:
            seconds (int): Duration in seconds
            
        Returns:
            str: Formatted duration string
        """
        if seconds < 60:
            return f"{seconds} seconds"
        elif seconds < 3600:
            minutes = seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        elif seconds < 86400:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours} hour{'s' if hours != 1 else ''} {minutes} minute{'s' if minutes != 1 else ''}"
        else:
            days = seconds // 86400
            hours = (seconds % 86400) // 3600
            return f"{days} day{'s' if days != 1 else ''} {hours} hour{'s' if hours != 1 else ''}" 