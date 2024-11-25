# Author Style Analysis: Traditional NLP vs Modern Transformers

## Overview
This project explores the effectiveness of **traditional NLP techniques** and **modern transformer-based approaches** in identifying authorship of literary texts. By analyzing works from **Charles Dickens, Jane Austen, and William Shakespeare**, the project investigates how stylistic features can reveal an author's unique writing signature.

The pipeline preprocesses raw texts by cleaning, tokenizing, and removing named entities to focus solely on writing style. Two contrasting methodologies are implemented:

1. **Traditional NLP Methods:**  
   Leveraging models like Naive Bayes, Support Vector Machines (SVM), and Logistic Regression using Bag-of-Words (BoW) representations.
   
2. **Modern Transformers:**  
   Fine-tuning **DistilBERT**, a lightweight BERT-based transformer, for sequence classification to identify authorship with contextual embeddings.

## Key Features
- **Dataset Preparation:**  
  - Texts are split into 25-word sequences for balanced training and testing.  
  - Named entities (characters, locations) are removed to avoid content-specific bias.  
  - Extensive preprocessing includes normalization, lemmatization, and stopword removal.

- **Traditional NLP Approaches:**  
  - Feature engineering with BoW.  
  - Implemented Naive Bayes, SVM, and Logistic Regression.  
  - Achieved **85.16% accuracy** with Naive Bayes.  

- **Transformer-Based Approach:**  
  - Fine-tuned **DistilBERT** with custom classification layers.  
  - Achieved **92.61% test accuracy** after 16 epochs of training.  

- **Performance Comparison:**  
  - Modern transformers significantly outperformed traditional methods (+7.45% accuracy).  
  - Traditional models were faster and more interpretable but lacked the nuance of context learning.  

- **Error Analysis:**  
  - Traditional methods struggled with stylistically similar authors.  
  - Transformers excelled at separating subtle patterns in sentence structure and vocabulary.

## Key Takeaways
- **Traditional NLP Models:**  
  - Efficient for lightweight applications where interpretability and speed are critical.  
  - Struggle to capture deep contextual nuances of writing.  

- **Modern Transformers:**  
  - Ideal for achieving high accuracy in complex text classification tasks.  
  - Require significant computational resources and careful fine-tuning to avoid overfitting.  

## Future Scope
- Enhance traditional models with advanced feature engineering and ensemble methods.  
- Optimize transformer training with data augmentation and hyperparameter tuning.  
- Explore domain adaptation for applying models to other genres or author sets.

This project underscores the trade-off between **simplicity and interpretability** in traditional methods versus the **complexity and accuracy** of modern NLP techniques.
