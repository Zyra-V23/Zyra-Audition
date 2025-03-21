"""
Mistral 7B implementation using Ollama for conversation detection.
This serves as a starter example for implementing open-source models.

Requirements:
- Ollama installed (https://ollama.ai)
- Mistral 7B model pulled (`ollama pull mistral`)
"""

import os
import json
import csv
import logging
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
OLLAMA_DEFAULT_PORT = 11434
OLLAMA_BASE_URL = f"http://localhost:{os.environ.get('OLLAMA_PORT', OLLAMA_DEFAULT_PORT)}"
MODEL_NAME = "mistral"
MODEL_ID = "mistral7b"  # ID used in output filenames
BATCH_SIZE = 25  # Process messages in smaller batches due to context window limitations

def generate_output_filename(input_file: str) -> str:
    """Generate the output filename following the convention:
    labels_[YYYYMMDD_HHMMSS]_[model_id]_[group_name].csv
    """
    # Extract group name from input path
    input_path = Path(input_file)
    group_name = input_path.parent.name
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Construct filename
    filename = f"labels_{timestamp}_{MODEL_ID}_{group_name}.csv"
    
    # Construct full path in the same directory as input
    output_path = input_path.parent / filename
    
    return str(output_path)

class Message:
    def __init__(self, id: str, text: str, timestamp: str, user: Dict[str, str]):
        self.id = id
        self.text = text
        self.timestamp = timestamp
        self.user = user

class Label:
    def __init__(self, message_id: str, conversation_id: str, topic: str, timestamp: str, metadata: Dict[str, Any]):
        self.message_id = message_id
        self.conversation_id = conversation_id
        self.topic = topic
        self.timestamp = timestamp
        self.metadata = metadata

class MistralConversationDetector:
    """
    Conversation detector using Mistral 7B through Ollama for message analysis and topic detection.
    """
    
    def __init__(self, port=None):
        """Initialize the Mistral detector."""
        self.model = MODEL_NAME
        self.labeler_id = MODEL_ID
        self.port = port or os.environ.get('OLLAMA_PORT', OLLAMA_DEFAULT_PORT)
        self.base_url = f"http://localhost:{self.port}"
        
        # Verify Ollama is running
        try:
            # Check server status
            response = requests.post(f"{self.base_url}/api/chat", json={
                "model": self.model,
                "messages": [{"role": "system", "content": "test"}]
            })
            
            if response.status_code == 404:
                # Try older API format
                response = requests.post(f"{self.base_url}/api/generate", json={
                    "model": self.model,
                    "prompt": "test"
                })
            
            if response.status_code not in [200, 404]:
                raise ConnectionError(f"Ollama server is not responding properly on port {self.port}")
                
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Ollama server is not running on port {self.port}. Please start Ollama first.")

    def load_messages(self, input_file: str) -> List[Message]:
        """Load messages from CSV file."""
        messages = []
        with open(input_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                user = {
                    'username': row['username'],
                    'first_name': row.get('first_name', ''),
                    'last_name': row.get('last_name', '')
                }
                message = Message(
                    id=row['id'],
                    text=row['text'],
                    timestamp=row['timestamp'],
                    user=user
                )
                messages.append(message)
        return messages

    def save_labels(self, labels: List[Label], output_file: str):
        """Save labels to CSV file."""
        output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else '.'
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['message_id', 'conversation_id', 'topic', 'timestamp', 'labeler_id', 'confidence'])
            for label in labels:
                writer.writerow([
                    label.message_id,
                    label.conversation_id,
                    label.topic,
                    label.timestamp,
                    label.metadata['labeler_id'],
                    label.metadata['confidence']
                ])

    def detect(self, messages: List[Message]) -> List[Label]:
        """
        Detect conversations in a list of messages.
        
        Args:
            messages: List of messages to analyze
            
        Returns:
            List of labels assigning messages to conversations
        """
        message_dicts = [
            {
                'id': msg.id,
                'text': msg.text,
                'timestamp': msg.timestamp,
                'user': msg.user
            }
            for msg in messages
        ]
        
        all_labels = []
        for i in range(0, len(message_dicts), BATCH_SIZE):
            batch = message_dicts[i:i + BATCH_SIZE]
            logger.info(f"Processing batch {i//BATCH_SIZE + 1} of {(len(message_dicts) + BATCH_SIZE - 1)//BATCH_SIZE}")
            batch_labels = self._detect_conversations(batch)
            all_labels.extend(batch_labels)
        
        return all_labels

    def _detect_conversations(self, messages: List[Dict[str, Any]]) -> List[Label]:
        """Implementation of conversation detection using Mistral."""
        try:
            prompt = self._create_analysis_prompt(messages)
            
            # Try new chat API first
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system", 
                                "content": """You are a conversation detection system. Output ONLY CSV data in this exact format:
message_id,conversation_id,topic,timestamp,confidence

Each field should be:
- message_id: integer
- conversation_id: integer (0 for spam)
- topic: string without quotes
- timestamp: ISO format string
- confidence: float between 0 and 1

No headers, no quotes, no additional text."""
                            },
                            {"role": "user", "content": prompt}
                        ],
                        "stream": False,
                        "options": {
                            "temperature": 0.3
                        }
                    }
                )
            except:
                # Fallback to older generate API
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 1000,
                        }
                    }
                )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.text}")
            
            # Parse the response based on API version
            if "message" in response.json():
                # New chat API
                csv_response = response.json()["message"]["content"].strip()
            else:
                # Old generate API
                csv_response = response.json()["response"].strip()
            
            # Remove any markdown formatting
            if csv_response.startswith('```csv'):
                csv_response = csv_response[6:]
            if csv_response.endswith('```'):
                csv_response = csv_response[:-3]
            
            # Split into lines and remove empty lines
            lines = [line.strip() for line in csv_response.split('\n') if line.strip()]
            
            # Process lines into Label objects
            labels = []
            
            for line in lines:
                try:
                    # Skip header lines and explanatory text
                    if line.startswith('message_id,') or not ',' in line or line.startswith('Note') or line.startswith('Here'):
                        continue
                    
                    # Split and clean the fields
                    parts = [p.strip().strip('"').strip("'") for p in line.split(',')]
                    
                    # Handle both 5-field (no labeler_id) and 6-field formats
                    if len(parts) == 5:
                        msg_id, conv_id, topic, timestamp, confidence = parts
                        labeler_id = self.labeler_id
                    elif len(parts) == 6:
                        msg_id, conv_id, topic, timestamp, labeler_id, confidence = parts
                    else:
                        logger.warning(f"Skipping malformed line (wrong number of fields): {line}")
                        continue
                    
                    try:
                        # Ensure numeric fields are valid
                        msg_id = str(int(msg_id))  # Should be a valid integer
                        conv_id = str(int(conv_id))  # Should be a valid integer
                        confidence = float(confidence)
                        
                        # Ensure labeler_id is always mistral7b
                        if labeler_id.isdigit():
                            labeler_id = self.labeler_id
                            
                        # Ensure confidence is between 0 and 1
                        confidence = min(1.0, max(0.0, float(confidence)))
                    except ValueError:
                        logger.warning(f"Skipping line with invalid numeric fields: {line}")
                        continue
                    
                    label = Label(
                        message_id=msg_id,
                        conversation_id=conv_id,
                        topic=topic,
                        timestamp=timestamp,
                        metadata={
                            'labeler_id': labeler_id,
                            'confidence': confidence
                        }
                    )
                    labels.append(label)
                except Exception as e:
                    logger.warning(f"Error processing line: {line} - {str(e)}")
                    continue
            
            return labels
            
        except Exception as e:
            logger.error(f"Error in Mistral conversation detection: {str(e)}")
            raise

    def _create_analysis_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Create the analysis prompt for Mistral."""
        # Load the conversation detection prompt template
        prompt_path = os.path.join(os.path.dirname(__file__), 'prompts/conversation_detection_prompt.txt')
        with open(prompt_path, 'r') as f:
            prompt_template = f.read()
            
        # Format messages
        formatted_messages = "\n"
        for msg in messages:
            formatted_messages += f"Message ID: {msg['id']}\n"
            formatted_messages += f"Timestamp: {msg['timestamp']}\n"
            formatted_messages += f"User: {msg['user']['username']}\n"
            formatted_messages += f"Content: {msg['text']}\n\n"

        return prompt_template.replace("[MESSAGES]", formatted_messages)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Mistral conversation detection on a messages file')
    parser.add_argument('input_file', help='Path to input CSV file containing messages')
    parser.add_argument('--output', '-o', help='Path to output file (optional, will use standard naming convention if not provided)',
                      default=None)
    parser.add_argument('--port', '-p', type=int, help='Ollama server port (default: 11434)',
                      default=OLLAMA_DEFAULT_PORT)
    
    args = parser.parse_args()
    
    # Generate output path if not provided
    output_path = args.output or generate_output_filename(args.input_file)
    
    # Create detector instance with specified port
    detector = MistralConversationDetector(port=args.port)
    
    # Load messages
    messages = detector.load_messages(args.input_file)
    logger.info(f"Loaded {len(messages)} messages from {args.input_file}")
    
    # Detect conversations
    labels = detector.detect(messages)
    
    # Save results
    detector.save_labels(labels, output_path)
    logger.info(f"Results written to: {output_path}") 