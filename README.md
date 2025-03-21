Use this as a prompt to determine each time where we are at the general process timeline and align the work we doing with this:

# Conversation Analysis Tests with LLMs - Test Suite

## Table of Contents
- [Step 1: Import the evaluation data](#step-1-import-the-evaluation-data)
- [Step 2: Run the evaluation](#step-2-run-the-evaluation)
- [Step 3: Show the results](#step-3-show-the-results)
- [Step 4: Your First Challenge - Experiment with Different Models](#step-4-your-first-challenge-experiment-with-different-models)
- [Step 5: Start understanding the modifying the prompts, pre-grouping techniques, models](#step-5-start-understanding-the-modifying-the-prompts-pre-grouping-techniques-models)

# Step 1: Import the evaluation data

The evaluation data you'll be working with has already been pre-packaged for you so you do not need to worry about parsing or downloading the data.
If you want to look broadly at all the different communities and the files associated with them, you can do so by navigating to the `data/groups` folder. As you'll see, there's 10 different communities associated with different Telegram groups for you to choose from. For our case, we'll be focusing on the Cere Network community, associated with the `thisiscere` Telegram group.

In every community, the data is organized into the following file types:

<details>
<summary><strong>Community Group Chat Data containing the conversations - Inputs to the experimentation</strong></summary>

The input data contains complete conversation histories from each Telegram community group chat, including message content, timestamps, and user information. This serves as the primary source for all our analysis tasks.

  * File path: `data/groups/thisiscere/messages_thisiscere.csv`
  * Contains original conversation content, timestamps, and user information
  * Primary input for all evaluations

    | ID | Text | Timestamp | Username | First Name | Last Name |
    |----|------|-----------|----------|------------|-----------|
    | 36569 | "You create your own attack and burn yourself…it makes no sense when the supply is still 10% and there is no real use case for the $cere token." | 2025-01-14T01:22:56Z | goldgold888 | TT | |
    | 36570 | "That will be improved in the future. I think Burning the supply using tokens from the Treasury is a positive thing. The aim is to reduce inflation." | 2025-01-14T01:25:40Z | Richnd | Richnd | \| I will never DM you first |
    | 36587 | "there was an actual announcement scheduled for today right?" | 2025-01-14T09:40:36Z | jjpdijkstra | Hans | Dijkstra |
    | 36588 | "I for one dont want CERE to miss out on face melting alt season that is not a day longer than q1 of this year." | 2025-01-14T09:42:06Z | jjpdijkstra | Hans | Dijkstra |
    | 36582 | "Confirm Bull run 🎉" | 2025-01-14T09:21:02Z | karwanxoshnaw_marshall | KARWAN | 馬修 克斯 |
</details>

<details>
<summary><strong>Ground Truth Files - Human-verified, manually labeled data that serves as the benchmark for measuring model accuracy</strong></summary>

These files contain human-annotated labels for conversations and spam messages, serving as the gold standard against which we evaluate model performance. Each file represents a different aspect of the ground truth: conversation groupings and spam identification.

  * `data/groups/thisiscere/GT_conversations_thisiscere.csv`: Manual conversation grouping labels

    | Message ID | Conversation ID |
    |------------|----------------|
    | 36569 | 1 |
    | 36570 | 1 |
    | 36587 | 3 |
    | 36588 | 3 |
    | 36582 | 2 |

  * `data/groups/thisiscere/GT_spam_thisiscere.csv`: Manual spam classification labels

    | Message ID | Is Spam |
    |------------|---------|
    | 36569 | 0 |
    | 36570 | 0 |
    | 36587 | 0 |
    | 36588 | 0 |
    | 36582 | 0 |
</details>

For readability's sake, for every individual file, we've provided just a small sample of the data. You're welcome to look at the full data in the `data/groups/thisiscere` folder.

# Step 2: Run the evaluation

## How to run the evaluation - A practical guide

Before running any evaluations, make sure you have the prerequisites and environment set up:

<details>
<summary><strong>A: Environment Setup</strong></summary>

### Prerequisites
- Python 3.8+
- Ollama installed (for running Mistral 7B locally)

### Setup Instructions
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama
# macOS or Windows - Download from https://ollama.ai/download

# Linux
curl https://ollama.ai/install.sh | sh

# Pull the Mistral 7B model
ollama pull mistral
```
</details>

### Main Task: Conversation Clustering

<details>
<summary><strong>Running the Evaluation with Mistral 7B</strong></summary>

We'll use Mistral 7B, a powerful open-source model, to perform conversation clustering. This model provides a good balance between performance and resource requirements.

### Required Files
- Input data: `data/groups/thisiscere/messages_thisiscere.csv`
- Ground truth file: `data/groups/thisiscere/GT_conversations_thisiscere.csv`

### Running the Model
```bash
# Start Ollama server (if not already running)
ollama serve

# Pull the Mistral 7B model (if you haven't already)
ollama pull mistral

# Run the conversation detection
python open_source_examples/mistral.py data/groups/thisiscere/messages_thisiscere.csv
```
</details>

### Additional Tasks

<details>
<summary><strong>C: Running Spam Detection Evaluation (Optional ⭐)</strong></summary>

The spam detection evaluation compares how well different models identify spam messages in a community.

### Required Files
- Ground truth file: `data/groups/thisiscere/GT_spam_thisiscere.csv`
- Model prediction files:
  * GPT-4: `data/groups/thisiscere/labels_20250131_143535_gpt4o_thisiscere.csv`
  * Claude 3.5: `data/groups/thisiscere/labels_20250131_171944_claude35s_thisiscere.csv`
  * DeepSeek V3: `data/groups/thisiscere/labels_20250131_185300_deepseekv3_thisiscere.csv`

### Running the Evaluation
```bash
python spam_metrics.py data/groups/thisiscere
```

### Output
The script will generate:
- Accuracy, precision, recall, and F1 scores for each model
- Results saved as `data/groups/thisiscere/metrics_spam_detection_thisiscere.csv`

Example output:
```csv
model,label_file,accuracy,precision,recall,f1
143535,labels_20250131_143535_gpt4o_thisiscere.csv,1.0,1.0,1.0,1.0      # Perfect spam detection
185300,labels_20250131_185300_deepseekv3_thisiscere.csv,0.955,0.842,1.0,0.914  # High recall but some false positives
171944,labels_20250131_171944_claude35s_thisiscere.csv,0.939,0.800,1.0,0.889    # Good overall but more false positives
```

Example interpretation from Cere Network results:
```csv
model,label_file,accuracy,precision,recall,f1
143535,labels_20250131_143535_gpt4o_thisiscere.csv,1.0,1.0,1.0,1.0      # Perfect spam detection
185300,labels_20250131_185300_deepseekv3_thisiscere.csv,0.955,0.842,1.0,0.914  # High recall but some false positives
171944,labels_20250131_171944_claude35s_thisiscere.csv,0.939,0.800,1.0,0.889    # Good overall but more false positives
```

Which translates to this more readable table:

| Model | Accuracy | Precision | Recall | F1 Score | Notes |
|-------|----------|-----------|---------|-----------|-------|
| GPT-4 | 1.000 | 1.000 | 1.000 | 1.000 | Perfect spam detection |
| DeepSeek V3 | 0.955 | 0.842 | 1.000 | 0.914 | High recall but some false positives |
| Claude 3.5 | 0.939 | 0.800 | 1.000 | 0.889 | Good overall but more false positives |

This table shows that while all models achieve perfect recall (catching all spam), GPT-4 stands out with perfect precision, while DeepSeek V3 and Claude 3.5 occasionally flag legitimate messages as spam.
</details>

<details>
<summary><strong>D: Running Topic Labeling Evaluation (Optional ⭐)</strong></summary>

The topic labeling evaluation assesses the quality and informativeness of conversation topic labels assigned by each model.

### Required Files
- Model prediction files:
  * GPT-4: `data/groups/thisiscere/labels_20250131_143535_gpt4o_thisiscere.csv`
  * Claude 3.5: `data/groups/thisiscere/labels_20250131_171944_claude35s_thisiscere.csv`
  * DeepSeek V3: `data/groups/thisiscere/labels_20250131_185300_deepseekv3_thisiscere.csv`
- Original message content: `data/groups/thisiscere/messages_thisiscere.csv`

### Running the Evaluation
```bash
python evaluate_topics.py data/groups/thisiscere
```

### Output
The script will generate:
- Information density scores
- Redundancy metrics
- Contextual relevance scores
- Label efficiency ratings
- Results saved as `data/groups/thisiscere/metrics_topics_thisiscere.csv`

Example output:
```csv
model,label_file,info_density,redundancy,relevance,efficiency,overall_score
143535,labels_20250131_143535_gpt4o_thisiscere.csv,8.5,0.95,0.92,0.88,0.91      # Excellent topic labeling
185300,labels_20250131_185300_deepseekv3_thisiscere.csv,7.8,0.88,0.85,0.82,0.84  # Good topic labeling
171944,labels_20250131_171944_claude35s_thisiscere.csv,8.2,0.90,0.88,0.85,0.88    # Very good topic labeling
```

Example interpretation from Cere Network results:
```csv
model,label_file,info_density,redundancy,relevance,efficiency,overall_score
143535,labels_20250131_143535_gpt4o_thisiscere.csv,8.5,0.95,0.92,0.88,0.91      # Excellent topic labeling
185300,labels_20250131_185300_deepseekv3_thisiscere.csv,7.8,0.88,0.85,0.82,0.84  # Good topic labeling
171944,labels_20250131_171944_claude35s_thisiscere.csv,8.2,0.90,0.88,0.85,0.88    # Very good topic labeling
```

Which translates to this more readable table:

| Model | Info Density (1-10) | Redundancy (0-1) | Relevance (0-1) | Efficiency (0-1) | Overall Score | Notes |
|-------|-------------------|-----------------|----------------|-----------------|---------------|-------|
| GPT-4 | 8.5 | 0.95 | 0.92 | 0.88 | 0.91 | Excellent topic labeling |
| DeepSeek V3 | 7.8 | 0.88 | 0.85 | 0.82 | 0.84 | Good topic labeling |
| Claude 3.5 | 8.2 | 0.90 | 0.88 | 0.85 | 0.88 | Very good topic labeling |

This table shows that all models perform well at topic labeling, with GPT-4 achieving the highest scores across all metrics. GPT-4 particularly excels in information density and redundancy reduction, while Claude 3.5 maintains strong performance across all categories. DeepSeek V3 shows good results but has slightly lower scores in information density and efficiency.
</details>

<br/>

# Step 3: Show the results

## Main Task - Conversation Clustering Results

<details>
<summary><strong>📄 Model Output Format</strong></summary>

  <br/>

  Results will be stored as `metrics_conversations_[community_name].csv` in your community's folder.

  For example, for the Cere Network community: `data/groups/thisiscere/metrics_conversations_thisiscere.csv`

  The model generates a CSV file following the naming convention:
  `data/groups/<community_name>/labels_<YYYYMMDD>_<HHMMSS>_mistral7b_<community_name>.csv`

  For example, if you're evaluating the Cere Network community (thisiscere) and run the script on February 25, 2025 at 20:22:30, the output file will be:
  `data/groups/thisiscere/labels_20250225_202230_mistral7b_thisiscere.csv`

  The file will have the following format (and for readability, we've only included a sample of the data):
  ```csv
  message_id,conversation_id,topic,timestamp,labeler_id,confidence
  36598,1,Token Discussion,2021-07-14T14:26:50Z,mistral7b,0.85
  36635,2,Project Updates,2025-01-15T02:52:44Z,mistral7b,0.82
  36638,0,Spam Message,2025-01-15T04:31:48Z,mistral7b,0.95
  ```

  Which translates to this more readable table:

  | Message ID | Conversation ID | Topic | Timestamp | Labeler ID | Confidence |
  |------------|----------------|--------|-----------|------------|------------|
  | 36598 | 1 | Token Discussion | 2021-07-14T14:26:50Z | mistral7b | 0.85 |
  | 36635 | 2 | Project Updates | 2025-01-15T02:52:44Z | mistral7b | 0.82 |
  | 36638 | 0 | Spam Message | 2025-01-15T04:31:48Z | mistral7b | 0.95 |

  You might be wondering - how well did the model perform? Is it good? Is it bad? To answer this question, we need to look at the performance metrics.

</details>

<details>
<summary><strong>📊 Performance Metrics</strong></summary>

  <br/>

  If you want to see what the performance metrics are for the model predictions you've just generated, you can run the following command:

  ```bash
  python conversation_metrics.py data/groups/thisiscere
  ```

  This will generate a CSV file with the performance metrics for the model predictions. It will take into account several reference models that have been already evaluated on the same data (GPT-4o, Claude 3.5, DeepSeek V3), as well as the model you've just run (Mistral 7B). 
  
  To calculate the metrics, the script "looks at":
  * what labels all the models generated (in the directory associated with the community), for example:
    - `data/groups/thisiscere/labels_20250131_143535_gpt4o_thisiscere.csv`
    - `data/groups/thisiscere/labels_20250131_171944_claude35s_thisiscere.csv`
    - `data/groups/thisiscere/labels_20250131_185300_deepseekv3_thisiscere.csv`
    - `data/groups/thisiscere/labels_20250225_202230_mistral7b_thisiscere.csv`
  * what the labels are for the ground truth (manually labelled by a human):
    - `data/groups/thisiscere/GT_conversations_thisiscere.csv`
  
  and then calculates the metrics based on that.
  
  The big picture understanding you need to have is, the closer the model labels are to the ground truth, the better the model is. The perfect score is 1, and the worst score is -1.
  
  If you want to understand this evaluation on a deeper level, you can read more about the metrics in the [Knowledge Base](#useful-links).

  | Model | ARI Score (-1 to 1) | Messages Processed | Notes |
  |-------|-------------------|-------------------|-------|
  | GPT-4 | 0.583 | 49 | Moderate conversation grouping accuracy |
  | DeepSeek V3 | 0.865 | 67 | Best performance, processed most messages |
  | Claude 3.5 | 0.568 | 49 | Moderate conversation grouping accuracy |
  | Mistral 7B | 0.219 | 27 | Lower accuracy, processed fewer messages |


  **Now you can see how the open source, locally deployed model performs against the big players 🏆 (GPT-4o, Claude 3.5, DeepSeek V3). As you can see, there's quite a performance gap - the ARI score for the smaller model is 0.219, while the big players are around 0.865 (with 1 being the perfect score).**

  **Your focus and core task is to try to close this gap 🎯 by experimenting with different models, different parameters, different system prompts, etc.**
  
  **Now that you understand this context, you can start experimenting! Feel free to jump straight into step number 4, referenced here: [Step 4: Experiment with different models](#step-4-experiment-with-different-models)**

</details>

## Additional Tasks

<details>
<summary><strong>Spam Detection Results (Optional ⭐)</strong></summary>


Results will be stored as `metrics_spam_detection_[community_name].csv` in your community's folder.

For example, for the Cere Network community: `data/groups/thisiscere/metrics_spam_detection_thisiscere.csv`

This file contains:
- Accuracy: Overall correctness of spam classification
- Precision: Proportion of true spam among messages flagged as spam
- Recall: Proportion of actual spam messages that were caught
- F1 Score: Balanced measure between precision and recall

Example interpretation from Cere Network results:
```csv
model,label_file,accuracy,precision,recall,f1
143535,labels_20250131_143535_gpt4o_thisiscere.csv,1.0,1.0,1.0,1.0      # Perfect spam detection
185300,labels_20250131_185300_deepseekv3_thisiscere.csv,0.955,0.842,1.0,0.914  # High recall but some false positives
171944,labels_20250131_171944_claude35s_thisiscere.csv,0.939,0.800,1.0,0.889    # Good overall but more false positives
```

Which translates to this more readable table:

| Model | Accuracy | Precision | Recall | F1 Score | Notes |
|-------|----------|-----------|---------|-----------|-------|
| GPT-4 | 1.000 | 1.000 | 1.000 | 1.000 | Perfect spam detection |
| DeepSeek V3 | 0.955 | 0.842 | 1.000 | 0.914 | High recall but some false positives |
| Claude 3.5 | 0.939 | 0.800 | 1.000 | 0.889 | Good overall but more false positives |

This table shows that while all models achieve perfect recall (catching all spam), GPT-4 stands out with perfect precision, while DeepSeek V3 and Claude 3.5 occasionally flag legitimate messages as spam.
</details>

<details>
<summary><strong>Topic Labeling Results (Optional ⭐)</strong></summary>

Results will be stored as `metrics_topics_[community_name].csv` in your community's folder.

For example, for the Cere Network community: `data/groups/thisiscere/metrics_topics_thisiscere.csv`

This file contains:
- Information Density: How well topics capture essential information (1-10)
- Redundancy: Measure of information efficiency (0-1)
- Relevance: How well topics match conversation content (0-1)
- Efficiency: Optimal use of words in labels (0-1)
- Overall Score: Combined performance metric (0-1)

Example interpretation from Cere Network results:
```csv
model,label_file,info_density,redundancy,relevance,efficiency,overall_score
143535,labels_20250131_143535_gpt4o_thisiscere.csv,8.5,0.95,0.92,0.88,0.91      # Excellent topic labeling
185300,labels_20250131_185300_deepseekv3_thisiscere.csv,7.8,0.88,0.85,0.82,0.84  # Good topic labeling
171944,labels_20250131_171944_claude35s_thisiscere.csv,8.2,0.90,0.88,0.85,0.88    # Very good topic labeling
```

Which translates to this more readable table:

| Model | Info Density (1-10) | Redundancy (0-1) | Relevance (0-1) | Efficiency (0-1) | Overall Score | Notes |
|-------|-------------------|-----------------|----------------|-----------------|---------------|-------|
| GPT-4 | 8.5 | 0.95 | 0.92 | 0.88 | 0.91 | Excellent topic labeling |
| DeepSeek V3 | 7.8 | 0.88 | 0.85 | 0.82 | 0.84 | Good topic labeling |
| Claude 3.5 | 8.2 | 0.90 | 0.88 | 0.85 | 0.88 | Very good topic labeling |

This table shows that all models perform well at topic labeling, with GPT-4 achieving the highest scores across all metrics. GPT-4 particularly excels in information density and redundancy reduction, while Claude 3.5 maintains strong performance across all categories. DeepSeek V3 shows good results but has slightly lower scores in information density and efficiency.
</details>

## Useful links

<details>
<summary><strong>🖥️ Frontend Visualization - Interactive Results Explorer</strong></summary>
<br/>
For a more interactive experience with a graphical user interface, you can access the results through our web application:
- Frontend Application URL: https://conversation-detection.stage.cere.io/
</details>

<details>
<summary><strong>📚 Knowledge Base - Theory behind evaluation metrics</strong></summary>

### Main Task: Conversation Clustering
The primary focus of our evaluation framework is the accurate clustering of messages into coherent conversations. This is the core challenge that directly impacts the quality of community analytics.

<details>
<summary><strong>Conversation Clustering</strong></summary>

The quality of conversation clustering is evaluated using the Adjusted Rand Index (ARI), a standard metric for comparing clustering results:

### Adjusted Rand Index (ARI)
  * Measures the similarity between two clusterings by considering all pairs of messages and checking whether they are grouped together or separately in both clusterings
  * Ranges from -1 to 1, where:
    - 1 indicates perfect agreement with ground truth
    - 0 indicates random labeling
    - Negative values indicate less agreement than expected by chance
  * Advantages:
    - Accounts for chance groupings
    - Handles different numbers of conversations
    - Independent of conversation labels/IDs

### ARI Calculation Process
   * First, convert to pair-wise relationships:
     - Ground Truth pairs in same conversation:
       * (msg1,msg2), (msg1,msg4), (msg2,msg4)  # Group 1
       * (msg3,msg5)                            # Group 2
     
     - Model Output pairs in same conversation:
       * (msg1,msg2)                           # Group 100
       * (msg3,msg5)                           # Group 101
       * msg4 alone in Group 102

   * ARI Score = 0.4 (moderate agreement) because:
     - Correctly grouped: (msg1,msg2), (msg3,msg5)
     - Incorrectly separated: msg4 from (msg1,msg2)

This example demonstrates how even with different conversation IDs (1,2 vs 100,101,102), ARI effectively measures clustering agreement by comparing pair-wise relationships between messages.
</details>

### Additional Tasks
The following tasks complement the main conversation clustering evaluation, providing additional insights into model capabilities:

<details>
<summary><strong>Topic Labeling (Optional ⭐)</strong></summary>

The evaluation of topic labels focuses on how well they capture and convey the essential information from conversations. Using principles from information theory, each topic label is evaluated against the actual conversation content it represents.

### Evaluation Framework
Topic labels are assessed by an expert system using the following information-theoretic criteria:

1. **Information Density** (1-10 scale):
   * Balance between brevity and informativeness
   * Optimal compression of conversation meaning
   * Example: "BTC Price Analysis Q4 2023" (9/10) vs "Crypto Discussion" (3/10)

2. **Redundancy Elimination**:
   * Penalizes repetitive or unnecessary information
   * Measures information efficiency
   * Example: "Bitcoin BTC Crypto Price" (low score due to redundancy) vs "Bitcoin Price Trends" (high score)

3. **Contextual Relevance**:
   * How well the label captures key conversation elements
   * Alignment with actual message content
   * Example: For a technical discussion about blockchain architecture, "Ethereum Gas Optimization" (high relevance) vs "ETH Discussion" (low relevance)

4. **Label Efficiency**:
   * Ratio of useful information to label length
   * Optimal use of each word/term
   * Example: "DeFi Liquidity Pool Returns" (efficient) vs "Discussion About Various Aspects of Decentralized Finance Liquidity Pools" (inefficient)

### Scoring System
Labels are scored on a 1-10 scale where:
- **1-2**: Severely problematic
  * Too vague or incomprehensible
  * Example: "Crypto stuff"
- **3-4**: Poor information value
  * Too generic or extremely redundant
  * Example: "Bitcoin cryptocurrency digital currency discussion"
- **5-6**: Acceptable but suboptimal
  * Conveys basic meaning but lacks precision
  * Example: "Cryptocurrency trading"
- **7-8**: Good balance
  * Clear, informative, efficient
  * Example: "BTC-ETH Price Correlation Analysis"
- **9-10**: Excellent
  * Optimal information density
  * Highly descriptive yet concise
  * Example: "L2 Rollup Performance Benchmarks Q1 2024"

### Example Evaluation

1. **Sample Conversation**:
   ```   User1: "How's Arbitrum's TPS compared to other L2s?"
   User2: "Currently around 40-50k TPS"
   User3: "Optimism is showing similar numbers"
   User1: "What about transaction costs?"
   User2: "Arb slightly cheaper, around $0.1-0.3 per tx"
   ```

2. **Topic Label Evaluation**:
   ```
   Label: "L2 Scaling: Arbitrum vs Optimism Performance"
   Score: 9/10
   Reasoning:
   - Specifies the exact L2 solutions being compared
   - Indicates the comparison is about performance
   - Captures both TPS and cost aspects
   - Concise yet comprehensive
   ```

3. **Alternative Label Analysis**:
   ```
   "L2 Discussion" - Score: 3/10
   - Too vague, loses critical information
   - Fails to capture comparative aspect
   - Missing specific solutions discussed

   "Detailed Technical Analysis of Layer 2 Blockchain Solutions Including Arbitrum and Optimism Transaction Speed Comparisons" - Score: 4/10
   - Unnecessarily verbose
   - High redundancy
   - Poor information-to-length ratio
   ```
</details>

<details>
<summary><strong>Spam Detection (Optional ⭐)</strong></summary>

Spam classification is evaluated using standard binary classification metrics. In our framework, spam messages are identified by `conversation_id = 0` in model outputs.

### Evaluation Metrics
- **Precision**: Accuracy of spam identification (minimize false positives)
  * Formula: `true_positives / (true_positives + false_positives)`
  * Critical for avoiding misclassification of legitimate messages
  * Example: Precision of 0.95 means 95% of messages labeled as spam are actually spam

- **Recall**: Completeness of spam detection (minimize false negatives)
  * Formula: `true_positives / (true_positives + false_negatives)`
  * Important for catching all spam messages
  * Example: Recall of 0.90 means 90% of all actual spam messages were caught

- **F1 Score**: Balanced measure of precision and recall
  * Formula: `2 * (precision * recall) / (precision + recall)`
  * Single metric for overall spam detection performance
  * Helps balance the trade-off between precision and recall

### Example Evaluation

1. **Sample Messages and Ground Truth**:
   ```csv
   message_id,text,is_spam
   msg1,"Check out crypto profits now!",1
   msg2,"What's the BTC price?",0
   msg3,"FREE BITCOIN click here!!!",1
   msg4,"Around $48k right now",0
   msg5,"Make 1000% gains guaranteed!!",1
   ```

2. **Model Output**:
   ```csv
   message_id,conversation_id,confidence
   msg1,0,0.95        # Correctly identified spam
   msg2,1,0.88        # Correctly identified non-spam
   msg3,0,0.92        # Correctly identified spam
   msg4,1,0.85        # Correctly identified non-spam
   msg5,2,0.70        # Missed spam (false negative)
   ```

3. **Metric Calculation**:
   ```
   True Positives (TP) = 2  (msg1, msg3)
   False Positives (FP) = 0
   True Negatives (TN) = 2  (msg2, msg4)
   False Negatives (FN) = 1  (msg5)

   Precision = TP/(TP+FP) = 2/(2+0) = 1.00
   Recall = TP/(TP+FN) = 2/(2+1) = 0.67
   F1 Score = 2 * (1.00 * 0.67)/(1.00 + 0.67) = 0.80
   ```

4. **Confidence Analysis**:
   * High confidence (>0.90) for clear spam patterns
   * Lower confidence (0.70-0.85) for ambiguous cases
   * Threshold of 0.80 used for spam classification

#### Common Spam Patterns
Models are evaluated on their ability to detect:
- Promotional language and excessive punctuation
- Unrealistic promises and urgency
- Suspicious links and contact information
- Repetitive message patterns
- Cross-posting across conversations

The evaluation emphasizes high precision to avoid disrupting legitimate conversations while maintaining acceptable recall for effective spam control.
</details>
</details>

<br/>

# Step 4: Your First Challenge - Experiment with Different Models

In Step 2, we used a script specifically designed for Mistral 7B (`open_source_examples/mistral.py`). Now, we'll use a more flexible, general-purpose script (`model_playground.py`) that can work with any model available through Ollama. This script is designed to be configurable through a YAML file, allowing you to easily experiment with different models, parameters, and system prompts without changing the code.

<details>
<summary><strong>🔧 Practical Guide: How to Experiment with Different Models</strong></summary>

<br/>

1. First, check if Ollama is running and start it if needed:
```bash
# Check if Ollama is running
ollama serve

# If you see "address already in use", Ollama is already running
# If not running, the command above will start the server
```

2. Pull the models you want to experiment with:
```bash
# Try different models available in Ollama
ollama pull llama2
ollama pull codellama
ollama pull phi
```

3. Configure your experiment in `open_source_examples/model_config.yaml`. You can now specify multiple models and use wildcards for prompts:
```yaml
# Update the model configuration
model:
  names: ["phi:latest", "llama2:latest"]  # List of models to test
  temperature: 0.3  # Lower for more focused outputs
  max_tokens: 1000  # Response length limit
  top_p: 0.95  # Sampling parameter

prompt:
  path: "open_source_examples/prompts/*"  # Use * to test all prompts in the directory
  # Or specific prompts: "open_source_examples/prompts/*.txt"
```

4. Run your experiment. You have two options:

   **Option 1: Single Model/Prompt Run**
   ```bash
   # Regular run using first model and prompt from config
   python model_playground.py data/groups/thisiscere/messages_thisiscere.csv
   ```

   **Option 2: Automated Testing (New!)**
   ```bash
   # Use --auto flag to test all model+prompt combinations
   python model_playground.py data/groups/thisiscere/messages_thisiscere.csv --auto
   ```

   The `--auto` flag will:
   - Test every combination of models specified in `names`
   - Process all matching prompt files if using wildcards
   - Generate separate output files for each combination
   - Name files clearly with model and prompt identifiers

5. Compare results:
```bash
# Generate metrics for your experiments
python conversation_metrics.py data/groups/thisiscere
```

### Example Automation Output

If your config has:
```yaml
model:
  names: ["phi:latest", "llama2:latest"]
prompt:
  path: "open_source_examples/prompts/*.txt"
```

Running with `--auto` will generate files like:
```
labels_20240227_phi-CD_prompt_alejandro_1_community.csv
labels_20240227_phi-CD_prompt_alejandro_2_community.csv
labels_20240227_llama2-CD_prompt_alejandro_1_community.csv
labels_20240227_llama2-CD_prompt_alejandro_2_community.csv
```

This makes it easy to compare how different models perform with different prompts.

### Key Areas for Experimentation

1. **Model Selection**
   - Try different model sizes and architectures
   - Compare specialized models (e.g., CodeLlama) with general models
   - Test latest models from the [Ollama model library](https://ollama.ai/library)
   - Experiment with multiple models in one run using the `names` list

2. **Parameter Tuning**
   - Adjust temperature for creativity vs. precision
   - Modify max_tokens for different output lengths
   - Fine-tune top_p for different sampling strategies

3. **Prompt Engineering**
   - Test multiple prompts using wildcards
   - Compare different prompt styles and formats
   - Analyze which prompts work best with which models

4. **Processing Configuration**
   - Experiment with batch_size for different processing speeds
   - Adjust max_context_messages for different context windows
   - Try different confidence thresholds

### Tips for Systematic Experimentation

- Use `--auto` for broad exploration of model+prompt combinations
- Keep a log of your experiments and their results
- Change one parameter at a time to understand its impact
- Monitor both performance metrics and resource usage
- Consider trade-offs between accuracy and processing speed
- Document any interesting findings or patterns you discover

Remember: The goal is to improve upon the baseline Mistral 7B performance (ARI score of 0.219) by finding the right combination of model and parameters for optimal conversation clustering.
</details>

<br/>

# Step 5: Start understanding the modifying the prompts, pre-grouping techniques, models

## Models Configuration

The project uses a flexible model configuration system that allows you to easily experiment with different models and parameters. The configuration is managed through `open_source_examples/model_config.yaml`.

### Key Components:

1. **Model Settings**
   - Choose from various Ollama models (e.g., Mistral, Phi-2, LLaMA 2, CodeLLama)
   - Configure temperature, max tokens, and top_p parameters
   - Example configuration:
   ```yaml
   model:
     name: "phi"  # Model name
     temperature: 0.3  # Lower for more focused outputs
     max_tokens: 1000  # Response length limit
     top_p: 0.95  # Sampling parameter
   ```

2. **Prompt Management**
   - Prompts are stored in separate text files
   - Easy to modify and experiment with different prompting strategies
   - Configuration specifies the prompt file:
   ```yaml
   prompt:
     path: "open_source_examples/prompts/conversation_detection_prompt.txt"
   ```

3. **Output Configuration**
   - Flexible output formatting (CSV, JSON)
   - Configurable timestamp formats
   - Confidence score inclusion options
   ```yaml
   output:
     format: "csv"
     include_confidence: true
     timestamp_format: "%Y%m%d_%H%M%S"
     output_dir: "data/groups"
   ```

4. **Processing Parameters**
   - Batch size control for efficient processing
   - Context window size configuration
   - Confidence thresholds for filtering results
   ```yaml
   processing:
     batch_size: 8
     max_context_messages: 5
     min_confidence_threshold: 0.7
   ```

## Model Playground

The project includes a model playground (`open_source_examples/model_playground.py`) for experimenting with different models and configurations:

1. **Key Features**:
   - Interactive testing of different models
   - Batch processing of messages
   - Standardized output formatting
   - Robust error handling and logging
   - Easy integration with Ollama models

2. **Usage**:
   ```bash
   python3 open_source_examples/model_playground.py data/groups/thisiscere/messages_thisiscere.csv --config model_config.yaml
   ```

3. **Customization**:
   - Modify `model_config.yaml` to experiment with different models
   - Adjust processing parameters for your use case
   - Test different prompt strategies by modifying the prompt file

## Modifying Prompts

To experiment with different prompting strategies:

1. **Locate the prompt file** specified in `model_config.yaml`
2. **Modify the prompt** while maintaining the `[MESSAGES]` placeholder
3. **Test the changes** using the model playground
4. **Evaluate results** using the metrics scripts

### Best Practices for Prompt Engineering:

1. **Clear Instructions**: Ensure prompts are clear and specific
2. **Consistent Format**: Maintain the expected output format (CSV structure)
3. **Context Window**: Consider the model's context window limitations
4. **Required Fields**: Always include message_id, conversation_id, topic, and timestamp
5. **Error Handling**: Include guidance for edge cases and error conditions

## Pre-grouping Techniques

[This section will be updated in the future with information about pre-grouping techniques]
