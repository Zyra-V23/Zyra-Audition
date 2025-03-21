#!/usr/bin/env python3
"""
Data Processing Module for Conversation Analytics

This module provides data loading, cleaning, and preprocessing functionality for
conversation data analysis. It includes methods for handling timestamps, cleaning text,
and detecting spam messages.

Authors: Zyra V23 and Zyxel 7B
Date: 2025-03-21
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import re
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class DataProcessor:
    """Class for loading, cleaning, and preprocessing conversation data."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the data processor.
        
        Args:
            config (Dict, optional): Configuration parameters
        """
        self.df = None
        self.config = config or {
            'min_text_length': 3,
            'remove_stopwords': True,
            'spam_score_threshold': 0.7
        }
        
        # Ensure NLTK resources are available
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading required NLTK resources...")
            nltk.download('punkt')
            nltk.download('stopwords')
            
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        self.df = pd.read_csv(file_path)
        print(f"Loaded {len(self.df)} messages from {file_path}")
        return self.df
        
    def explore_data(self) -> Dict:
        """
        Explore the data and generate statistics.
        
        Returns:
            Dict: Data exploration statistics
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Get column names with tolerance for case
        username_col = 'Username' if 'Username' in self.df.columns else 'username'
        text_col = 'Text' if 'Text' in self.df.columns else 'text'
        timestamp_col = 'Timestamp' if 'Timestamp' in self.df.columns else 'timestamp'
            
        # Basic statistics
        stats = {
            'total_messages': len(self.df),
            'unique_users': self.df[username_col].nunique() if username_col in self.df.columns else 0,
        }
        
        if timestamp_col in self.df.columns:
            stats['date_range'] = (self.df[timestamp_col].min(), self.df[timestamp_col].max())
        
        if text_col in self.df.columns:
            stats['text_stats'] = {
                'mean_length': self.df[text_col].astype(str).str.len().mean(),
                'max_length': self.df[text_col].astype(str).str.len().max(),
                'min_length': self.df[text_col].astype(str).str.len().min(),
                'empty_texts': len(self.df[self.df[text_col].astype(str).str.len() == 0])
            }
        
        # Time-based statistics
        if timestamp_col in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df[timestamp_col])
            stats['time_stats'] = {
                'messages_by_day': self.df['datetime'].dt.date.value_counts().to_dict(),
                'messages_by_hour': self.df['datetime'].dt.hour.value_counts().to_dict(),
                'avg_time_between': self.df['datetime'].diff().mean().total_seconds() / 60  # minutes
            }
            
        # User-based statistics
        if username_col in self.df.columns:
            stats['user_stats'] = {
                'messages_per_user': self.df[username_col].value_counts().to_dict(),
                'top_users': self.df[username_col].value_counts().head(5).to_dict()
            }
        
        return stats
        
    def clean_data(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean and preprocess the data.
        
        Returns:
            Tuple[pd.DataFrame, Dict]: Cleaned DataFrame and cleaning report
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        cleaning_report = {
            'initial_rows': len(self.df),
            'initial_columns': len(self.df.columns),
            'null_values': self.df.isnull().sum().to_dict(),
            'operations': []
        }
        
        # Normalize column names (handle lowercase)
        column_mapping = {}
        
        # Check if columns exist in lowercase and create mapping
        if 'id' in self.df.columns and 'ID' not in self.df.columns:
            column_mapping['id'] = 'ID'
        
        if 'text' in self.df.columns and 'Text' not in self.df.columns:
            column_mapping['text'] = 'Text'
            
        if 'timestamp' in self.df.columns and 'Timestamp' not in self.df.columns:
            column_mapping['timestamp'] = 'Timestamp'
            
        if 'username' in self.df.columns and 'Username' not in self.df.columns:
            column_mapping['username'] = 'Username'
            
        if 'first_name' in self.df.columns and 'First Name' not in self.df.columns:
            column_mapping['first_name'] = 'First Name'
            
        if 'last_name' in self.df.columns and 'Last Name' not in self.df.columns:
            column_mapping['last_name'] = 'Last Name'
            
        # Rename columns if needed
        if column_mapping:
            self.df = self.df.rename(columns=column_mapping)
            cleaning_report['operations'].append('columns_renamed')
            cleaning_report['column_mapping'] = column_mapping
        
        # Handle timestamps
        if 'Timestamp' in self.df.columns:
            self.df['datetime'] = pd.to_datetime(self.df['Timestamp'])
            self.df['date'] = self.df['datetime'].dt.date
            self.df['time'] = self.df['datetime'].dt.time
            cleaning_report['operations'].append('timestamp_parsing')
            
        # Clean text
        if 'Text' in self.df.columns:
            # Handle missing text
            missing_text = self.df['Text'].isnull().sum()
            if missing_text > 0:
                self.df['Text'] = self.df['Text'].fillna('')
                cleaning_report['operations'].append('missing_text_filled')
                cleaning_report['missing_text'] = missing_text
                
            # Basic text cleaning
            self.df['clean_text'] = self.df['Text'].apply(self._clean_text)
            cleaning_report['operations'].append('text_cleaning')
            
        # Remove unnecessary columns
        keep_columns = ['ID', 'Text', 'clean_text', 'Timestamp', 'datetime', 'date', 'time', 
                        'Username', 'First Name', 'Last Name']
        dropped_columns = [col for col in self.df.columns if col not in keep_columns]
        
        if dropped_columns:
            self.df = self.df[[col for col in self.df.columns if col in keep_columns]]
            cleaning_report['operations'].append('columns_dropped')
            cleaning_report['dropped_columns'] = dropped_columns
            
        # Generate final report
        cleaning_report['final_rows'] = len(self.df)
        cleaning_report['final_columns'] = len(self.df.columns)
        cleaning_report['text_quality'] = {
            'empty_clean_text': (self.df['clean_text'] == '').sum(),
            'avg_clean_text_length': self.df['clean_text'].str.len().mean(),
            'short_texts': (self.df['clean_text'].str.len() < self.config['min_text_length']).sum()
        }
        
        return self.df, cleaning_report
        
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def tokenize_text(self, remove_stopwords: bool = None) -> pd.DataFrame:
        """
        Tokenize the cleaned text.
        
        Args:
            remove_stopwords (bool, optional): Whether to remove stopwords
                
        Returns:
            pd.DataFrame: DataFrame with tokenized text
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if 'clean_text' not in self.df.columns:
            _, _ = self.clean_data()
            
        # Use parameter or fall back to config
        remove_stopwords = remove_stopwords if remove_stopwords is not None else self.config['remove_stopwords']
        
        # Tokenize text
        self.df = self.df.assign(
            tokens=self.df['clean_text'].apply(
                lambda x: word_tokenize(x.lower()) if isinstance(x, str) else []
            )
        )
        
        # Remove stopwords if requested
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            self.df = self.df.assign(
                tokens=self.df['tokens'].apply(
                    lambda tokens: [word for word in tokens if word not in stop_words]
                )
            )
            
        # Calculate token statistics
        token_counts = self.df['tokens'].apply(len)
        
        token_stats = {
            'total_tokens': token_counts.sum(),
            'avg_tokens_per_message': token_counts.mean(),
            'max_tokens': token_counts.max(),
            'min_tokens': token_counts.min(),
            'empty_token_lists': (token_counts == 0).sum()
        }
        
        # Vocabulary statistics
        all_tokens = [token for tokens in self.df['tokens'] for token in tokens]
        unique_tokens = set(all_tokens)
        
        token_stats['vocabulary_size'] = len(unique_tokens)
        token_stats['vocabulary_density'] = len(unique_tokens) / max(len(all_tokens), 1)
        
        print("\nToken Statistics:")
        for key, value in token_stats.items():
            print(f"- {key}: {value}")
            
        # Show sample
        if not self.df.empty:
            idx = self.df.index[0]
            print(f"\nSample tokenization for message {idx}:")
            print(f"Original: {self.df.loc[idx, 'Text'][:100]}{'...' if len(self.df.loc[idx, 'Text']) > 100 else ''}")
            print(f"Tokens: {self.df.loc[idx, 'tokens'][:10]}{'...' if len(self.df.loc[idx, 'tokens']) > 10 else ''}")
            
        return self.df
        
    def detect_spam(self, threshold: float = None) -> pd.DataFrame:
        """
        Detect potential spam messages.
        
        Args:
            threshold (float, optional): Spam score threshold
                
        Returns:
            pd.DataFrame: DataFrame with spam detection results
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        if 'tokens' not in self.df.columns:
            self.tokenize_text()
            
        # Use parameter or fall back to config
        threshold = threshold if threshold is not None else self.config['spam_score_threshold']
        
        # Calculate spam score based on various heuristics
        spam_indicators = {
            'all_caps_ratio': self.df['clean_text'].apply(
                lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1) if isinstance(x, str) else 0
            ),
            'exclamation_count': self.df['clean_text'].apply(
                lambda x: x.count('!') if isinstance(x, str) else 0
            ),
            'spam_words': self.df['tokens'].apply(
                lambda tokens: sum(1 for t in tokens if t.lower() in ['free', 'win', 'money', 'cash', 'prize', 'buy'])
            ),
            'token_length_ratio': self.df['tokens'].apply(
                lambda tokens: max([len(t) for t in tokens] or [0]) / 20
            ),
            'url_count': self.df['Text'].apply(
                lambda x: len(re.findall(r'https?://\S+|www\.\S+', x)) if isinstance(x, str) else 0
            ),
            'short_text': self.df['clean_text'].apply(
                lambda x: 1 if (isinstance(x, str) and len(x) < 10) else 0
            )
        }
        
        # Combine indicators with weights
        weights = {
            'all_caps_ratio': 0.2,
            'exclamation_count': 0.1,
            'spam_words': 0.3,
            'token_length_ratio': 0.15,
            'url_count': 0.15,
            'short_text': 0.1
        }
        
        # Calculate combined spam score
        self.df['spam_score'] = sum(
            spam_indicators[key] * weights[key] for key in weights
        )
        
        # Normalize to 0-1 range
        max_spam_score = self.df['spam_score'].max()
        if max_spam_score > 0:
            self.df['spam_score'] = self.df['spam_score'] / max_spam_score
            
        # Flag potential spam messages
        self.df['is_spam'] = (self.df['spam_score'] > threshold).astype(int)
        
        spam_stats = {
            'spam_messages': self.df['is_spam'].sum(),
            'spam_percentage': (self.df['is_spam'].sum() / len(self.df)) * 100,
            'avg_spam_score': self.df['spam_score'].mean(),
            'top_spam_scores': self.df.nlargest(5, 'spam_score')[['spam_score', 'clean_text']].values.tolist()
        }
        
        print("\nSpam Detection Results:")
        print(f"- Total spam messages: {spam_stats['spam_messages']} ({spam_stats['spam_percentage']:.2f}%)")
        print(f"- Average spam score: {spam_stats['avg_spam_score']:.4f}")
        
        if spam_stats['spam_messages'] > 0:
            print("\nTop potential spam messages:")
            for score, text in spam_stats['top_spam_scores']:
                print(f"- Score {score:.4f}: {text[:50]}{'...' if len(text) > 50 else ''}")
                
        return self.df
        
    def save_processed_data(self, output_path: Optional[str] = None) -> str:
        """
        Save processed data to CSV.
        
        Args:
            output_path (str, optional): Output file path
                
        Returns:
            str: Path to the saved file
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        if output_path is None:
            # Generate output path based on original file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"processed_messages_{timestamp}.csv"
            
        self.df.to_csv(output_path, index=False)
        print(f"\nProcessed data saved to: {output_path}")
        
        return output_path 