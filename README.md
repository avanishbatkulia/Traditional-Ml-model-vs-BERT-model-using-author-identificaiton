# Author Style Analysis: Traditional NLP vs Modern Transformers

## Overview

This project implements and compares traditional NLP techniques with modern transformer-based approaches for author identification. The analysis focuses on identifying authorship based on writing style patterns rather than content-specific markers across works by:
- Charles Dickens
- Jane Austen
- William Shakespeare

## Dataset Preparation

### Data Sources
- Charles Dickens: Gutenberg Corpus
- Jane Austen: Gutenberg Corpus
- William Shakespeare: Text file compilation

### Processing Pipeline

1. **Initial Cleanup**
   - Removed prefaces and introductions
   - Started extraction from Chapter 1
   - Split texts into training/testing sets per author

2. **Entity Removal**
   - Implemented Named Entity Recognition (NER) to remove:
     - Location names
     - Person names
   - Rationale: Prevent model from relying on character/location names specific to certain works

3. **Sequence Generation**
   - Split text into 25-word sequences for traditional and 15-word sequence for modern method
   - Created labeled datasets per author
   - Removed sequences containing identified entities
   - Rationale: 25-word and 15-word sequences provide sufficient context while maintaining manageable complexity for respective models

4. **Preprocessing Steps**
   - Text Normalization:
     - Lowercase conversion
     - Special character removal
     - Whitespace normalization
   - Pattern Removal:
     - Numerical digits
     - URLs and special markers
     - Non-alphabetic characters
   - Text Transformation:
     - Lemmatization using NLTK's WordNetLemmatizer
     - Stopword removal
     - Tokenization
   - Feature Engineering:
     - Created Bag of Words (BoW) representation
     - Generated train_bow.csv and test_bow.csv

5. **Final Dataset**
   - Combined and shuffled processed sequences
   - Created train_df.csv and test_df.csv
   - Dataset sizes:
     - Training: 10,174 sequences
     - Testing: 5,709 sequences

## Model Implementations

### Traditional NLP Approach

#### 1. Naive Bayes Multinomial
- **Performance**: 85.16% accuracy consistent across random states
- **Class-wise Performance**:
  - Class 0: Precision: 0.83, Recall: 0.83, F1: 0.83
  - Class 1: Precision: 0.85, Recall: 0.84, F1: 0.84
  - Class 2: Precision: 0.88, Recall: 0.89, F1: 0.88
- **Cross-Validation**:
  - Scores: [0.938, 0.940, 0.935, 0.933, 0.932]
  - Average: 0.94
  - Standard deviation: ≈0.003

#### 2. SVM vs Logistic Regression
- **SVM Advantages**:
  - Better non-linear pattern handling
  - Robust to overfitting
  - RBF kernel captures complex writing patterns
- **Logistic Regression**:
  - Accuracy: 81%
  - Best Parameters: {'solver': 'saga', 'penalty': 'l2', 'C': 1}

### Modern Approach: Fine-tuned DistilBERT

#### Implementation Architecture

1. **Tokenization**
```python
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
```
- Uses DistilBERT's built-in tokenizer
- Maintains contextual information
- Handles out-of-vocabulary words

2. **Model Architecture**
```python
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=3
)
```
- Base: DistilBERT (40% smaller, 60% faster than BERT-base)
- Classification head with 3 output labels
- Softmax activation with dropout regularization

3. **Training Configuration**
```python
training_args = TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=16,
    weight_decay=0.01
)
```

#### Performance Results
- Training Loss: 0.0503
- Test Accuracy: 92.61%
- Processing Speed: 114.414 samples/second
- Training Runtime: 2491.16s

## Comparative Analysis

### Traditional vs Modern Approaches

|Aspect|Traditional (Naive Bayes)|Modern (DistilBERT)|
|------|------------------------|-------------------|
|Accuracy|85%|92.61%|
|Training Time|Minutes|~42 minutes|
|Resource Needs|CPU sufficient|GPU recommended|
|Deployment|Simple|Complex|

### Use Case Recommendations

Choose Traditional When:
- Limited computational resources
- Need for model interpretability
- Quick prototyping required
- Small dataset (<1000 samples)

Choose Modern When:
- GPU resources available
- Higher accuracy needed
- Complex language patterns
- Larger datasets (>1000 samples)

## Future Improvements

1. **Dataset Enhancement**
   - Include more authors
   - Experiment with sequence lengths
   - Implement advanced data augmentation

2. **Model Optimization**
   - Explore ensemble methods
   - Implement feature selection
   - Test additional algorithms

3. **Analysis Extension**
   - Add chronological analysis
   - Implement style transfer
   - Explore cross-genre performance

