#!/usr/bin/env python3
"""
Command Line Interface for Conversation Analytics

This module provides a CLI to access the conversation analytics functionality,
making it easy to analyze conversation data from the command line.

Authors: Zyra V23 and Zyxel 7B
Date: 2025-03-21
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Optional, Any
import datetime
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Import API module
try:
    from .api.conversation_analyzer import ConversationAnalyzer
except ImportError:
    # For standalone usage
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from conversation_analytics.api.conversation_analyzer import ConversationAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Analyze conversations from message data'
    )
    
    subparsers = parser.add_subparsers(dest='command')
    
    # analyze_conversations command
    analyze_parser = subparsers.add_parser('analyze_conversations', help='Analyze conversations in a dataset')
    analyze_parser.add_argument(
        'input_file',
        help='Path to input CSV file containing messages'
    )
    analyze_parser.add_argument(
        '--output-dir',
        default='results',
        help='Directory to store analysis results'
    )
    analyze_parser.add_argument(
        '--config',
        help='Path to configuration YAML file'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for conversation analysis"""
    args = parse_args()
    
    if args.command == 'analyze_conversations':
        try:
            # Initialize analyzer with config if provided
            analyzer = ConversationAnalyzer(
                output_dir=args.output_dir,
                config_path=args.config
            )
            
            # Process the dataset
            logger.info(f"Processing dataset from {args.input_file}")
            results = analyzer.analyze_conversations(
                input_file=args.input_file,
                output_dir=args.output_dir
            )
            
            logger.info(f"Analysis complete. Results saved to {args.output_dir}")
            return results
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            raise
    else:
        logger.error("No command specified. Use --help for usage information.")
        sys.exit(1)

if __name__ == '__main__':
    main() 