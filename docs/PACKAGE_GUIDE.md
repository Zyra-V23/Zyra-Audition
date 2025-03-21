# Conversation Analytics Package Guide

## Description

Conversation Analytics is a Python package designed to analyze conversation data, providing tools for:

- Processing and cleaning conversation data
- Text tokenization and spam detection
- Text vectorization with adaptive method selection
- Text similarity analysis and clustering
- Semantic search in conversations

## Main Features

- **Robust data processing**: Load, clean, and preprocess conversation data
- **Adaptive text vectorization**: Automatically selects the best vectorization method based on text characteristics
- **Intelligent cache system**: Improves performance by storing previous vectorizations
- **Advanced similarity analysis**: Identifies similar texts using various methods and metrics
- **Text clustering**: Organizes texts into groups based on their semantic similarity
- **Visualizations**: Generates visualizations to better understand data and results
- **Easy-to-use API**: Simple and consistent interface for all functionalities
- **Complete CLI**: Access to all functionalities from the command line

## Installation

```bash
# Clone the repository
git clone https://github.com/zyra-v23/conversation-analytics.git
cd conversation-analytics

# Install the package
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

## Quick Usage

### From Python

```python
from conversation_analytics import ConversationAnalyzer

# Initialize the analyzer
analyzer = ConversationAnalyzer(output_dir="my_analysis")

# Analyze a complete conversation (processing + vectorization + similarity)
results = analyzer.analyze_conversation(
    file_path="my_messages.csv", 
    vectorization_method="adaptive",
    n_clusters=3,
    similarity_metric="cosine"
)

# Or use the components individually
analyzer.load_data("my_messages.csv")
analyzer.process_data(remove_stopwords=True)
analyzer.vectorize_texts(method="tfidf")
similarity_report = analyzer.analyze_similarity(n_clusters=5)

# Search for messages similar to a text
similar_messages = analyzer.find_similar_messages(
    query_text="When is the meeting?",
    k=5,
    min_similarity=0.3
)
```

### From the Command Line

```bash
# Complete analysis
python -m conversation_analytics analyze messages.csv --output-dir my_analysis

# Data processing
python -m conversation_analytics process messages.csv --output-dir processed

# Text vectorization
python -m conversation_analytics vectorize processed/processed_data.csv --method adaptive

# Similarity analysis
python -m conversation_analytics similarity vectors.npy --clusters 5

# Search for similar messages
python -m conversation_analytics search analyzer_state.pkl "When is the meeting?"
```

## Examples

### Data Processing

```python
from conversation_analytics import DataProcessor

processor = DataProcessor()
processor.load_data("messages.csv")
cleaned_data, report = processor.clean_data()
tokenized_data = processor.tokenize_text(remove_stopwords=True)
processed_data = processor.detect_spam(threshold=0.8)
processor.save_processed_data("processed_messages.csv")
```

### Text Vectorization

```python
from conversation_analytics import AdaptiveTextVectorizer

# Adaptive vectorizer with cache
vectorizer = AdaptiveTextVectorizer(cache_size=1000)

# Vectorize texts
texts = ["Hello, how are you?", "I'm fine, thanks", "When is the meeting?"]
vectors = vectorizer.transform(texts)

# Save the model
vectorizer.save_model("my_vectorizer.pkl")

# Load an existing model
vectorizer = AdaptiveTextVectorizer.load_model("my_vectorizer.pkl")
```

### Similarity Analysis

```python
from conversation_analytics import SimilarityIndex
import numpy as np

# Build a similarity index
index = SimilarityIndex(metric="cosine")
index.build_index(vectors)

# Search for similar texts
query_vector = vectors[0]
indices, similarities = index.search(query_vector, k=3)

# Calculate similarities between all texts
similarity_matrix = index.calculate_pairwise_similarities()
```

## Main Modules

### DataProcessor
Handles loading, cleaning, and preprocessing conversation data:
- CSV file loading
- Exploration and statistical analysis
- Text cleaning and normalization
- Tokenization and stopword handling
- Heuristic-based spam detection

### TextVectorizer and AdaptiveTextVectorizer
Transform text into vectors for analysis:
- Multiple vectorization methods (BoW, TF-IDF, transformers)
- Adaptive selection based on text characteristics
- Intelligent cache to improve performance
- Functions to save and load trained models

### SimilarityIndex and VectorClusterer
Analyze similarities between text vectors:
- Building optimized indices for search
- Support for multiple similarity metrics
- Clustering with K-means
- Visualization of clusters with t-SNE
- Detailed similarity statistics

### ConversationAnalyzer
High-level API that integrates all components:
- Complete analysis workflows
- State management and persistence
- Unified interfaces for all functionalities

## Configuration Options

### Data Processing
- `remove_stopwords`: Removes stopwords during tokenization
- `spam_threshold`: Threshold for spam detection (0.0-1.0)
- `min_text_length`: Minimum length for valid texts

### Vectorization
- `method`: Vectorization method ('bow', 'tfidf', 'transformer', 'adaptive')
- `max_features`: Maximum number of features for BoW/TF-IDF
- `transformer_model`: Transformer model to use
- `cache_size`: Cache size for the adaptive vectorizer

### Similarity Analysis
- `metric`: Similarity metric ('cosine', 'euclidean', 'inner_product')
- `n_clusters`: Number of clusters for grouping
- `min_similarity`: Minimum similarity threshold for searches

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Scikit-learn
- NLTK
- Matplotlib
- FAISS (optional, for faster similarity search)
- SentenceTransformers (optional, for transformer-based vectorization) 