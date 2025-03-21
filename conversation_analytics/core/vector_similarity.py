#!/usr/bin/env python3
"""
Vector Similarity Module for Conversation Analytics

This module provides functionality for analyzing similarities between text vectors:
- Building and using similarity search indices
- Finding nearest neighbors for a query vector
- Performing batch similarity searches
- Clustering similar vectors
- Visualizing similarity clusters

Author: Zyxel 7B audited by Zyra
Date: 2025-03-21
"""

import numpy as np
import os
import pickle
from typing import List, Dict, Tuple, Union, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.cm as cm
import time
import yaml
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing HDBSCAN with proper error handling
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
    logger.info("HDBSCAN successfully imported")
except ImportError as e:
    HDBSCAN_AVAILABLE = False
    logger.warning("HDBSCAN import failed. Using DBSCAN as fallback. Error: %s", str(e))

# Import FAISS if available for faster similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS successfully imported")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Using numpy for similarity calculations (slower).")

class SimilarityIndex:
    """
    Class for building and using similarity search indices.
    
    Features:
    - Fast neighbor search using FAISS (if available) or numpy
    - Support for different similarity metrics
    - Persistence of indices for reuse
    - Hierarchical indices for efficient searching
    """
    
    METRICS = ['cosine', 'euclidean', 'inner_product']
    
    def __init__(self, metric: str = 'cosine', use_gpu: bool = False, 
                 n_lists: int = 100, nprobe: int = 10):
        """
        Initialize the similarity index.
        
        Args:
            metric (str): Similarity metric to use ('cosine', 'euclidean', or 'inner_product')
            use_gpu (bool): Whether to use GPU acceleration if available
            n_lists (int): Number of Voronoi cells for IVF index (only used with FAISS)
            nprobe (int): Number of cells to visit during search (only used with FAISS)
        """
        if metric not in self.METRICS:
            raise ValueError(f"Metric must be one of {self.METRICS}")
            
        self.metric = metric
        self.index = None
        self.original_vectors = None
        self.vector_dim = None
        self.use_faiss = FAISS_AVAILABLE
        self.use_gpu = use_gpu and self.use_faiss
        
        # FAISS specific parameters for hierarchical index
        self.n_lists = n_lists
        self.nprobe = nprobe
        self.is_trained = False
        
        # GPU resources if needed
        self.gpu_resources = None
        if self.use_gpu:
            try:
                self.gpu_resources = faiss.StandardGpuResources()
                print("GPU resources initialized for similarity search")
            except Exception as e:
                print(f"Failed to initialize GPU resources: {str(e)}")
                self.use_gpu = False
        
    def build_index(self, vectors: np.ndarray, use_hierarchical: bool = True) -> None:
        """
        Build a similarity search index from vectors.
        
        Args:
            vectors (np.ndarray): Input vectors (n_samples, n_features)
            use_hierarchical (bool): Whether to use hierarchical index for faster search
        """
        if vectors.shape[0] == 0:
            raise ValueError("Cannot build index with empty vector set")
            
        # Store original vectors for metrics not supported by FAISS
        self.original_vectors = vectors.copy()
        self.vector_dim = vectors.shape[1]
        
        if self.use_faiss:
            # Use FAISS for fast similarity search
            try:
                # Determine number of lists for IVF index
                n_vectors = vectors.shape[0]
                if use_hierarchical and n_vectors >= 1000:
                    # For hierarchical indexing, adjust n_lists based on dataset size
                    # Rule of thumb: sqrt(n) for smaller datasets, 4*sqrt(n) for larger ones
                    self.n_lists = min(
                        max(int(4 * np.sqrt(n_vectors)), 8), 
                        n_vectors // 10  # Don't use more than 1/10th of vectors
                    )
                    
                    # Adjust nprobe based on n_lists
                    self.nprobe = max(1, min(self.n_lists // 4, 50))
                    
                    print(f"Building hierarchical index with {self.n_lists} clusters and nprobe={self.nprobe}")
                
                # Create the appropriate index based on metric and dataset size
                if self.metric == 'cosine':
                    # Normalize vectors for cosine similarity
                    vectors_norm = vectors.copy()
                    
                    # Handle zero vectors to avoid division by zero
                    norms = np.linalg.norm(vectors_norm, axis=1)
                    zero_norm_indices = norms == 0
                    norms[zero_norm_indices] = 1.0  # Set to 1 to avoid division by zero
                    
                    # Normalize each vector
                    vectors_norm = vectors_norm / norms[:, np.newaxis]
                    
                    # Use different index types based on dataset size
                    if use_hierarchical and n_vectors >= 1000:
                        # Use IVF (inverted file index) for larger datasets
                        quantizer = faiss.IndexFlatIP(self.vector_dim)
                        self.index = faiss.IndexIVFFlat(quantizer, self.vector_dim, self.n_lists, faiss.METRIC_INNER_PRODUCT)
                        
                        # Train the index if we have enough vectors
                        self.index.train(vectors_norm.astype(np.float32))
                        self.is_trained = True
                        
                        # Set search parameters
                        self.index.nprobe = self.nprobe
                    else:
                        # Use flat index for smaller datasets
                        self.index = faiss.IndexFlatIP(self.vector_dim)
                    
                    # Add vectors to index
                    self.index.add(vectors_norm.astype(np.float32))
                    
                elif self.metric == 'euclidean':
                    if use_hierarchical and n_vectors >= 1000:
                        quantizer = faiss.IndexFlatL2(self.vector_dim)
                        self.index = faiss.IndexIVFFlat(quantizer, self.vector_dim, self.n_lists, faiss.METRIC_L2)
                        self.index.train(vectors.astype(np.float32))
                        self.is_trained = True
                        self.index.nprobe = self.nprobe
                    else:
                        self.index = faiss.IndexFlatL2(self.vector_dim)
                    
                    self.index.add(vectors.astype(np.float32))
                    
                elif self.metric == 'inner_product':
                    if use_hierarchical and n_vectors >= 1000:
                        quantizer = faiss.IndexFlatIP(self.vector_dim)
                        self.index = faiss.IndexIVFFlat(quantizer, self.vector_dim, self.n_lists, faiss.METRIC_INNER_PRODUCT)
                        self.index.train(vectors.astype(np.float32))
                        self.is_trained = True
                        self.index.nprobe = self.nprobe
                    else:
                        self.index = faiss.IndexFlatIP(self.vector_dim)
                    
                    self.index.add(vectors.astype(np.float32))
                
                # Convert to GPU index if requested
                if self.use_gpu and self.gpu_resources is not None:
                    print("Moving index to GPU...")
                    self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
                    
            except Exception as e:
                print(f"Failed to build FAISS index: {str(e)}")
                print("Falling back to numpy implementation")
                self.use_faiss = False
                
        if not self.use_faiss:
            # Simple numpy implementation (slower)
            self.index = vectors
            
    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors to the query vector.
        
        Args:
            query_vector (np.ndarray): Query vector
            k (int): Number of neighbors to return
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (indices, distances/similarities)
        """
        if self.index is None:
            raise ValueError("Index not built yet. Call build_index first.")
            
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
            
        # Ensure k doesn't exceed the number of vectors
        k = min(k, self.original_vectors.shape[0])
        
        if self.use_faiss:
            # Use FAISS for search
            try:
                if self.metric == 'cosine':
                    # Normalize query vector for cosine similarity
                    query_norm = np.linalg.norm(query_vector)
                    if query_norm > 0:
                        query_vector = query_vector / query_norm
                    
                # FAISS returns (distances, indices)
                distances, indices = self.index.search(query_vector.astype(np.float32), k)
                
                # Convert distances to similarities for cosine and inner product
                if self.metric in ['cosine', 'inner_product']:
                    return indices[0], distances[0]
                else:
                    # For euclidean, smaller is better
                    return indices[0], 1 / (1 + distances[0])
                
            except Exception as e:
                print(f"FAISS search failed: {str(e)}. Falling back to numpy.")
                self.use_faiss = False
                
        # Numpy implementation
        if not self.use_faiss:
            if self.metric == 'cosine':
                # Normalize vectors for cosine similarity
                query_norm = np.linalg.norm(query_vector)
                if query_norm > 0:
                    query_vector = query_vector / query_norm
                    
                # Calculate dot product for normalized vectors (equivalent to cosine)
                similarities = np.dot(self.index, query_vector.T).flatten()
                
                # Get top-k indices
                top_indices = np.argsort(similarities)[-k:][::-1]
                top_similarities = similarities[top_indices]
                
                return top_indices, top_similarities
                
            elif self.metric == 'euclidean':
                # Calculate euclidean distances
                distances = np.sqrt(np.sum((self.index - query_vector) ** 2, axis=1))
                
                # Get top-k indices (smallest distances)
                top_indices = np.argsort(distances)[:k]
                top_similarities = 1 / (1 + distances[top_indices])  # Convert to similarity
                
                return top_indices, top_similarities
                
            elif self.metric == 'inner_product':
                # Calculate dot product
                similarities = np.dot(self.index, query_vector.T).flatten()
                
                # Get top-k indices
                top_indices = np.argsort(similarities)[-k:][::-1]
                top_similarities = similarities[top_indices]
                
                return top_indices, top_similarities
    
    def batch_search(self, query_vectors: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform batch search for multiple query vectors.
        
        Args:
            query_vectors (np.ndarray): Batch of query vectors
            k (int): Number of neighbors to return per query
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (indices, distances/similarities)
        """
        if self.index is None:
            raise ValueError("Index not built yet. Call build_index first.")
            
        # Ensure k doesn't exceed the number of vectors
        k = min(k, self.original_vectors.shape[0])
        
        if self.use_faiss:
            try:
                if self.metric == 'cosine':
                    # Normalize query vectors for cosine similarity
                    norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0  # Avoid division by zero
                    query_vectors = query_vectors / norms
                
                # FAISS returns (distances, indices)
                distances, indices = self.index.search(query_vectors.astype(np.float32), k)
                
                # Convert distances to similarities for cosine and inner product
                if self.metric in ['cosine', 'inner_product']:
                    return indices, distances
                else:
                    # For euclidean, smaller is better - convert to similarity
                    return indices, 1 / (1 + distances)
            
            except Exception as e:
                print(f"FAISS batch search failed: {str(e)}. Falling back to numpy.")
                self.use_faiss = False
        
        # Numpy implementation (slower)
        if not self.use_faiss:
            n_queries = query_vectors.shape[0]
            all_indices = np.zeros((n_queries, k), dtype=np.int64)
            all_similarities = np.zeros((n_queries, k))
            
            # Process each query vector individually
            for i in range(n_queries):
                indices, similarities = self.search(query_vectors[i:i+1], k)
                all_indices[i] = indices
                all_similarities[i] = similarities
                
            return all_indices, all_similarities
            
    def save(self, path: str) -> None:
        """
        Save the index to disk.
        
        Args:
            path (str): Path to save the index
        """
        save_data = {
            'metric': self.metric,
            'vector_dim': self.vector_dim,
            'use_faiss': self.use_faiss,
            'original_vectors': self.original_vectors,
            'n_lists': self.n_lists,
            'nprobe': self.nprobe,
            'is_trained': self.is_trained
        }
        
        # Save metadata and vectors
        with open(path + '.meta', 'wb') as f:
            pickle.dump(save_data, f)
            
        # Save FAISS index separately if available
        if self.use_faiss:
            # Convert to CPU index if on GPU
            cpu_index = self.index
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
            
            faiss.write_index(cpu_index, path + '.faiss')
            
    @classmethod
    def load(cls, path: str) -> 'SimilarityIndex':
        """
        Load a saved index.
        
        Args:
            path (str): Path to the saved index
            
        Returns:
            SimilarityIndex: Loaded index
        """
        # Load metadata and vectors
        with open(path + '.meta', 'rb') as f:
            save_data = pickle.load(f)
            
        # Create instance with saved parameters
        instance = cls(
            metric=save_data['metric'],
            n_lists=save_data['n_lists'],
            nprobe=save_data['nprobe']
        )
        
        instance.vector_dim = save_data['vector_dim']
        instance.original_vectors = save_data['original_vectors']
        instance.use_faiss = save_data['use_faiss']
        instance.is_trained = save_data.get('is_trained', False)
        
        # Load FAISS index if available
        if instance.use_faiss and os.path.exists(path + '.faiss'):
            try:
                instance.index = faiss.read_index(path + '.faiss')
                # Set search parameters
                if hasattr(instance.index, 'nprobe'):
                    instance.index.nprobe = instance.nprobe
                
                # Move to GPU if requested
                if instance.use_gpu and instance.gpu_resources is not None:
                    instance.index = faiss.index_cpu_to_gpu(instance.gpu_resources, 0, instance.index)
            except Exception as e:
                print(f"Failed to load FAISS index: {str(e)}")
                instance.use_faiss = False
                instance.index = instance.original_vectors
        else:
            # Fall back to original vectors if FAISS index not available
            instance.use_faiss = False
            instance.index = instance.original_vectors
            
        return instance

class VectorClusterer:
    """
    Cluster similar vectors and analyze cluster characteristics.
    
    Features:
    - K-means clustering of vectors
    - Analysis of within-cluster similarities
    - Visualization of clusters
    """
    
    def __init__(self, metric: str = 'cosine'):
        """
        Initialize the clusterer.
        
        Args:
            metric (str): Similarity metric to use
        """
        self.metric = metric
        self.kmeans = None
        self.clusters = None
        self.n_clusters = None
        self.vectors = None
        self.labels = None
        
    def cluster_vectors(self, vectors: np.ndarray, n_clusters: int = 3, random_state: int = 42) -> np.ndarray:
        """
        Cluster vectors using K-means.
        
        Args:
            vectors (np.ndarray): Input vectors
            n_clusters (int): Number of clusters
            random_state (int): Random seed for reproducibility
            
        Returns:
            np.ndarray: Cluster labels
        """
        if vectors.shape[0] == 0:
            raise ValueError("Cannot cluster empty vector set")
            
        # Adjust number of clusters if there are fewer vectors than requested clusters
        if vectors.shape[0] < n_clusters:
            n_clusters = vectors.shape[0]
            print(f"Warning: Fewer vectors ({vectors.shape[0]}) than requested clusters ({n_clusters}). "
                  f"Adjusting to {n_clusters} clusters.")
                  
        self.vectors = vectors
        self.n_clusters = n_clusters
        
        # Run K-means
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.clusters = self.kmeans.fit_predict(vectors)
        
        return self.clusters
        
    def get_cluster_statistics(self) -> Dict[int, Dict]:
        """
        Get statistics for each cluster.
        
        Returns:
            Dict[int, Dict]: Dictionary of cluster statistics
        """
        if self.clusters is None:
            raise ValueError("No clusters found. Call cluster_vectors first.")
            
        stats = {}
        sim_index = SimilarityIndex(metric=self.metric)
        sim_index.build_index(self.vectors)
        pairwise_sims = sim_index.calculate_pairwise_similarities()
        
        for i in range(self.n_clusters):
            # Get indices of vectors in this cluster
            cluster_indices = np.where(self.clusters == i)[0]
            
            # Skip if cluster is empty
            if len(cluster_indices) == 0:
                stats[i] = {
                    'size': 0,
                    'avg_similarity': 0.0
                }
                continue
                
            # If cluster has only one vector, similarity is 1.0
            if len(cluster_indices) == 1:
                stats[i] = {
                    'size': 1,
                    'avg_similarity': 1.0
                }
                continue
                
            # Calculate within-cluster similarities
            within_sims = []
            for j in range(len(cluster_indices)):
                for k in range(j + 1, len(cluster_indices)):
                    idx1, idx2 = cluster_indices[j], cluster_indices[k]
                    within_sims.append(pairwise_sims[idx1, idx2])
                    
            # Calculate average similarity
            if within_sims:
                avg_sim = np.mean(within_sims)
            else:
                avg_sim = 0.0
                
            stats[i] = {
                'size': len(cluster_indices),
                'avg_similarity': avg_sim
            }
            
        return stats
        
    def visualize_clusters(self, output_path: str = None, labels: List[str] = None,
                           perplexity: int = 30, learning_rate: int = 200) -> None:
        """
        Visualize clusters using t-SNE for dimensionality reduction.
        
        Args:
            output_path (str): Path to save the visualization
            labels (List[str]): Optional labels for data points
            perplexity (int): Perplexity parameter for t-SNE
            learning_rate (int): Learning rate for t-SNE
        """
        if self.clusters is None:
            raise ValueError("No clusters found. Call cluster_vectors first.")
            
        # Store labels
        self.labels = labels
        
        # Use t-SNE to reduce dimensions for visualization
        tsne = TSNE(n_components=2, perplexity=min(perplexity, self.vectors.shape[0]-1),
                    learning_rate=learning_rate, random_state=42)
                    
        try:
            # Reduce vectors to 2D
            vectors_2d = tsne.fit_transform(self.vectors)
            
            # Plot the clusters
            plt.figure(figsize=(10, 8))
            
            # Create scatter plot for each cluster
            colors = cm.rainbow(np.linspace(0, 1, self.n_clusters))
            
            for i in range(self.n_clusters):
                cluster_indices = np.where(self.clusters == i)[0]
                plt.scatter(
                    vectors_2d[cluster_indices, 0],
                    vectors_2d[cluster_indices, 1],
                    color=colors[i],
                    label=f'Cluster {i} (n={len(cluster_indices)})',
                    alpha=0.7
                )
                
            # Add labels if provided
            if labels and len(labels) == self.vectors.shape[0]:
                for i, (x, y) in enumerate(vectors_2d):
                    if i % max(1, len(labels) // 20) == 0:  # Show only some labels to avoid clutter
                        plt.annotate(
                            str(i),  # Use index rather than potentially long text
                            (x, y),
                            fontsize=8
                        )
                        
            plt.title(f'Vector Clusters ({self.metric} similarity)')
            plt.legend()
            plt.tight_layout()
            
            # Save or show
            if output_path:
                plt.savefig(output_path, dpi=300)
                plt.close()
            else:
                plt.show()
                
        except Exception as e:
            print(f"Failed to visualize clusters: {str(e)}")
            
    def save_clustering(self, path: str) -> None:
        """
        Save clustering results to disk.
        
        Args:
            path (str): Path to save the results
        """
        if self.clusters is None:
            raise ValueError("No clusters found. Call cluster_vectors first.")
            
        # Save clustering data
        clustering_data = {
            'kmeans': self.kmeans,
            'clusters': self.clusters,
            'n_clusters': self.n_clusters,
            'metric': self.metric,
            'labels': self.labels
        }
        
        with open(path, 'wb') as f:
            pickle.dump(clustering_data, f)
            
    @classmethod
    def load_clustering(cls, path: str) -> 'VectorClusterer':
        """
        Load saved clustering results.
        
        Args:
            path (str): Path to the saved results
            
        Returns:
            VectorClusterer: Loaded clusterer instance
        """
        with open(path, 'rb') as f:
            clustering_data = pickle.load(f)
            
        instance = cls(metric=clustering_data['metric'])
        instance.kmeans = clustering_data['kmeans']
        instance.clusters = clustering_data['clusters']
        instance.n_clusters = clustering_data['n_clusters']
        instance.labels = clustering_data['labels']
        
        return instance

def batch_process_vectors(vectors: np.ndarray, metric: str = 'cosine',
                         n_clusters: int = 3, output_dir: str = 'similarity_output') -> Dict:
    """
    Process vectors in batch mode with index building, clustering, and visualization.
    
    Args:
        vectors (np.ndarray): Input vectors
        metric (str): Similarity metric to use
        n_clusters (int): Number of clusters
        output_dir (str): Output directory for results
        
    Returns:
        Dict: Results summary
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    results = {
        'vector_shape': vectors.shape,
        'metric': metric
    }
    
    # Build similarity index
    print(f"Building similarity search index for {vectors.shape[0]} vectors using {metric} metric...")
    sim_index = SimilarityIndex(metric=metric)
    sim_index.build_index(vectors)
    
    # Calculate and analyze pairwise similarities
    print("Calculating pairwise similarities...")
    pairwise_sims = sim_index.calculate_pairwise_similarities()
    
    # Get similarity statistics
    sim_stats = {
        'average': np.mean(pairwise_sims),
        'max': np.max(pairwise_sims),
        'min': np.min(pairwise_sims)
    }
    results['similarity_stats'] = sim_stats
    
    print(f"Similarity statistics:")
    print(f"  - Average similarity: {sim_stats['average']:.3f}")
    print(f"  - Maximum similarity: {sim_stats['max']:.3f}")
    print(f"  - Minimum similarity: {sim_stats['min']:.3f}")
    
    # Save similarity index
    sim_index_path = os.path.join(output_dir, f"similarity_index_{metric}.idx")
    sim_index.save(sim_index_path)
    
    # Cluster vectors
    print(f"Clustering vectors into {n_clusters} clusters...")
    clusterer = VectorClusterer(metric=metric)
    cluster_labels = clusterer.cluster_vectors(vectors, n_clusters=n_clusters)
    
    # Get cluster statistics
    cluster_stats = clusterer.get_cluster_statistics()
    results['cluster_stats'] = cluster_stats
    
    print("Cluster statistics:")
    for cluster_id, stats in cluster_stats.items():
        print(f"  - Cluster {cluster_id}: {stats['size']} vectors, avg similarity: {stats['avg_similarity']:.3f}")
        
    # Visualize clusters
    vis_path = os.path.join(output_dir, f"vector_clusters_{metric}.png")
    clusterer.visualize_clusters(output_path=vis_path)
    
    # Save clustering results
    clustering_path = os.path.join(output_dir, f"clustering_results_{metric}.pkl")
    clusterer.save_clustering(clustering_path)
    
    # Record time taken
    results['processing_time'] = time.time() - start_time
    
    print(f"Vector processing completed in {results['processing_time']:.2f} seconds")
    print(f"Results saved to {output_dir}")
    
    return results

def find_similar_texts(query_vector: np.ndarray, vectors: np.ndarray, texts: List[str],
                       metric: str = 'cosine', k: int = 5, min_similarity: float = 0.3) -> List[Dict]:
    """
    Find texts similar to a query vector.
    
    Args:
        query_vector (np.ndarray): Query vector
        vectors (np.ndarray): Vector corpus to search in
        texts (List[str]): Corresponding texts
        metric (str): Similarity metric
        k (int): Number of results to return
        min_similarity (float): Minimum similarity threshold
        
    Returns:
        List[Dict]: List of similar texts with metadata
    """
    # Build index
    sim_index = SimilarityIndex(metric=metric)
    sim_index.build_index(vectors)
    
    # Search for similar vectors
    indices, similarities = sim_index.search(query_vector, k=k)
    
    # Filter by threshold
    filtered_indices = [i for i, s in zip(indices, similarities) if s >= min_similarity]
    filtered_similarities = [s for s in similarities if s >= min_similarity]
    
    # Prepare results
    results = []
    for idx, sim in zip(filtered_indices, filtered_similarities):
        results.append({
            'text': texts[idx] if idx < len(texts) else "Unknown",
            'index': int(idx),
            'similarity': float(sim)
        })
        
    return results

class ConversationClusterer:
    """
    Enhanced conversation clustering using HDBSCAN/DBSCAN and temporal-semantic hybrid approach.
    Falls back to DBSCAN if HDBSCAN is not available.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the conversation clusterer with configuration from YAML file.
        
        Args:
            config_path: Path to the configuration file (optional)
        """
        # Load configuration
        self._load_config(config_path)
        
        # Initialize clustering algorithm
        self._initialize_clusterer()
        
        # Initialize metrics tracking
        self.silhouette_scores = []
        self.cluster_stats = {}
        
    def _load_config(self, config_path: str = None):
        """Load and validate configuration"""
        # Default configuration
        self.config = {
            'hdbscan': {
                'min_cluster_size': 3,
                'min_samples': 2,
                'cluster_selection_epsilon': 0.3,
                'metric': 'precomputed'
            },
            'temporal': {
                'time_weight': 0.4,
                'max_time_gap': 3600,
                'similarity_threshold': 0.65
            },
            'vectors': {
                'normalize': True,
                'use_idf': True,
                'min_df': 2,
                'max_df': 0.95
            },
            'outliers': {
                'reassignment_threshold': 0.6,
                'max_outlier_ratio': 0.2
            },
            'optimization': {
                'use_gpu': False,
                'batch_size': 1000,
                'cache_similarity': True
            }
        }
        
        # Load configuration from file if provided
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    # Update configuration recursively
                    self._update_config_recursive(self.config, file_config)
            except Exception as e:
                logger.warning(f"Error loading config from {config_path}: {str(e)}")
                logger.info("Using default configuration")
        
        # Set instance variables from config
        self._set_instance_variables()
    
    def _update_config_recursive(self, base_config: dict, new_config: dict):
        """Recursively update configuration dictionary"""
        for key, value in new_config.items():
            if key in base_config and isinstance(base_config[key], dict):
                if isinstance(value, dict):
                    self._update_config_recursive(base_config[key], value)
            else:
                base_config[key] = value
    
    def _set_instance_variables(self):
        """Set instance variables from configuration"""
        self.time_weight = self.config['temporal']['time_weight']
        self.similarity_threshold = self.config['temporal']['similarity_threshold']
        self.max_time_gap = self.config['temporal']['max_time_gap']
        self.min_cluster_size = self.config['hdbscan']['min_cluster_size']
        self.min_samples = self.config['hdbscan']['min_samples']
        self.cluster_selection_epsilon = self.config['hdbscan']['cluster_selection_epsilon']
        
    def _initialize_clusterer(self):
        """Initialize the appropriate clustering algorithm"""
        if HDBSCAN_AVAILABLE:
            self.clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                metric='precomputed'
            )
            logger.info("Using HDBSCAN for clustering")
        else:
            # Fall back to DBSCAN with similar parameters
            eps = self.cluster_selection_epsilon
            self.clusterer = DBSCAN(
                eps=eps,
                min_samples=self.min_samples,
                metric='precomputed'
            )
            logger.info("Using DBSCAN as fallback clustering algorithm")
        
    def cluster_conversations(self, 
                            vectors: np.ndarray, 
                            timestamps: List[int],
                            texts: List[str] = None,
                            message_ids: List[int] = None) -> Dict[str, Any]:
        """
        Cluster conversations using enhanced HDBSCAN-based approach.
        
        Args:
            vectors: Text vectors
            timestamps: Message timestamps
            texts: Original message texts (optional)
            message_ids: Message IDs (optional)
            
        Returns:
            Dictionary containing clustering results and metrics
        """
        # Normalize timestamps to [0,1] range
        timestamps_arr = np.array(timestamps)
        timestamps_normalized = (timestamps_arr - timestamps_arr.min()) / (timestamps_arr.max() - timestamps_arr.min())
        
        # Build hybrid similarity matrix
        similarity_matrix = self._build_hybrid_similarity_matrix(vectors, timestamps_normalized)
        
        # Apply clustering
        labels = self.clusterer.fit_predict(1 - similarity_matrix)  # Convert similarity to distance
        probabilities = self.clusterer.probabilities_
        
        # Handle outliers
        outlier_mask = labels == -1
        if outlier_mask.any():
            labels = self._reassign_outliers(vectors, labels, np.where(outlier_mask)[0], similarity_matrix)
        
        # Enforce temporal continuity
        labels = self._enforce_temporal_continuity(labels, timestamps_arr)
        
        # Compute confidence scores
        confidence_scores = self._compute_confidence_scores(vectors, labels, similarity_matrix)
        
        # Compute cluster statistics
        stats = self._compute_cluster_stats(labels, timestamps_arr, similarity_matrix, texts)
        
        # Prepare results
        results = {
            'labels': labels.tolist(),
            'confidence_scores': confidence_scores.tolist(),
            'probabilities': probabilities.tolist(),
            'n_clusters': len(np.unique(labels[labels >= 0])),
            'outlier_percentage': (labels == -1).mean() * 100,
            'cluster_stats': stats,
            'parameters': {
                'time_weight': self.time_weight,
                'similarity_threshold': self.similarity_threshold,
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples
            }
        }
        
        if message_ids is not None:
            results['message_ids'] = message_ids
            
        return results
        
    def _build_hybrid_similarity_matrix(self, vectors: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """
        Build hybrid similarity matrix combining semantic and temporal similarities.
        """
        # Compute semantic similarity
        semantic_sim = cosine_similarity(vectors)
        
        # Compute temporal similarity
        temporal_sim = 1 - np.abs(timestamps[:, np.newaxis] - timestamps)
        
        # Combine similarities
        hybrid_sim = (1 - self.time_weight) * semantic_sim + self.time_weight * temporal_sim
        
        return hybrid_sim
        
    def _reassign_outliers(self, vectors: np.ndarray, labels: np.ndarray, 
                          outlier_indices: np.ndarray, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Reassign outliers to their most similar clusters using enhanced method.
        """
        new_labels = labels.copy()
        unique_clusters = np.unique(labels[labels >= 0])
        
        for idx in outlier_indices:
            # Get similarities to all points
            similarities = similarity_matrix[idx]
            
            # Calculate average similarity to each cluster
            cluster_similarities = []
            for cluster in unique_clusters:
                cluster_mask = labels == cluster
                if cluster_mask.any():
                    avg_sim = similarities[cluster_mask].mean()
                    cluster_similarities.append((cluster, avg_sim))
            
            if cluster_similarities:
                # Assign to most similar cluster if above threshold
                best_cluster, best_sim = max(cluster_similarities, key=lambda x: x[1])
                if best_sim >= self.similarity_threshold:
                    new_labels[idx] = best_cluster
        
        return new_labels
        
    def _enforce_temporal_continuity(self, labels: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """
        Enforce temporal continuity within clusters using enhanced method.
        """
        new_labels = labels.copy()
        unique_clusters = np.unique(labels[labels >= 0])
        
        for cluster in unique_clusters:
            cluster_mask = new_labels == cluster
            cluster_times = timestamps[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_times) > 1:
                # Sort by time
                sorted_idx = np.argsort(cluster_times)
                sorted_times = cluster_times[sorted_idx]
                sorted_indices = cluster_indices[sorted_idx]
                
                # Check for time gaps
                time_gaps = np.diff(sorted_times)
                split_points = np.where(time_gaps > self.max_time_gap)[0]
                
                if len(split_points) > 0:
                    # Create new clusters for temporally disconnected segments
                    new_cluster_id = labels.max() + 1
                    for i, split_point in enumerate(split_points):
                        if i == 0:
                            continue
                        new_labels[sorted_indices[split_point + 1:]] = new_cluster_id
                        new_cluster_id += 1
        
        return new_labels
        
    def _compute_confidence_scores(self, vectors: np.ndarray, labels: np.ndarray, 
                                 similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Compute confidence scores for cluster assignments using enhanced method.
        """
        confidence_scores = np.zeros(len(labels))
        unique_clusters = np.unique(labels[labels >= 0])
        
        for cluster in unique_clusters:
            cluster_mask = labels == cluster
            cluster_indices = np.where(cluster_mask)[0]
            
            for idx in cluster_indices:
                # Calculate average similarity to cluster members
                cluster_similarities = similarity_matrix[idx][cluster_mask]
                avg_cluster_sim = cluster_similarities.mean()
                
                # Calculate average similarity to other clusters
                other_clusters_mask = ~cluster_mask & (labels >= 0)
                if other_clusters_mask.any():
                    other_clusters_sim = similarity_matrix[idx][other_clusters_mask].mean()
                    # Confidence score based on difference between in-cluster and out-cluster similarities
                    confidence_scores[idx] = (avg_cluster_sim - other_clusters_sim + 1) / 2
                else:
                    confidence_scores[idx] = avg_cluster_sim
        
        return confidence_scores
    
    def _compute_cluster_stats(self, labels: np.ndarray, timestamps: np.ndarray, 
                              similarity_matrix: np.ndarray, texts: List[str] = None) -> Dict:
        """
        Compute statistics and metadata for each detected conversation cluster.
        
        Args:
            labels (np.ndarray): Conversation cluster IDs
            timestamps (np.ndarray): Message timestamps
            similarity_matrix (np.ndarray): Similarity matrix
            texts (List[str], optional): Original message texts
            
        Returns:
            Dict: Statistics for each conversation cluster
        """
        cluster_stats = {}
        
        # For each cluster, compute statistics
        for cluster_id in np.unique(labels):
            cluster_indices = np.where(labels == cluster_id)[0]
            
            # Get timestamps for this cluster
            cluster_times = timestamps[cluster_indices]
            
            # Calculate temporal statistics
            duration = max(cluster_times) - min(cluster_times) if len(cluster_times) > 1 else 0
            
            # Calculate cohesion (average similarity within cluster)
            if len(cluster_indices) > 1:
                cohesion = 0
                pairs = 0
                for i in range(len(cluster_indices)):
                    for j in range(i+1, len(cluster_indices)):
                        cohesion += similarity_matrix[cluster_indices[i], cluster_indices[j]]
                        pairs += 1
                cohesion = cohesion / pairs if pairs > 0 else 0
            else:
                cohesion = 1.0  # Default for singleton clusters
            
            # Compute text statistics if texts are provided
            text_stats = {}
            if texts is not None:
                cluster_texts = [texts[i] for i in cluster_indices]
                avg_length = np.mean([len(t) for t in cluster_texts]) if cluster_texts else 0
                text_stats = {
                    'avg_text_length': avg_length,
                    'sample_texts': cluster_texts[:3] if len(cluster_texts) > 0 else []
                }
            
            # Store all statistics
            cluster_stats[int(cluster_id)] = {
                'size': len(cluster_indices),
                'start_time': int(min(cluster_times)) if len(cluster_times) > 0 else 0,
                'end_time': int(max(cluster_times)) if len(cluster_times) > 0 else 0,
                'duration_seconds': int(duration),
                'cohesion': float(cohesion),
                'indices': cluster_indices.tolist(),
                **text_stats
            }
            
        return cluster_stats 