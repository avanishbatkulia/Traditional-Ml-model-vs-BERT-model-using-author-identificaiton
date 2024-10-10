# Author-Identification-NLTK-vs-BERT
This project compares traditional NLP methods using NLTK with modern deep learning models, specifically Hugging Face's BERT, for the task of author identification. The dataset includes text samples from three classic authors: Charles Dickens, Jane Austen, and William Shakespeare.
We explore how well each method performs in identifying the authorship of text passages, highlighting the strengths and weaknesses of traditional vs. modern approaches in natural language processing.

Key Features:

Data: Text excerpts from Dickens, Austen, and Shakespeare.
Models:
Traditional approach using NLTK for feature extraction and classification.
Modern approach using pretrained BERT models from Hugging Face.
Comparison: Accuracy, efficiency, and interpretability between traditional and deep learning models.

Data and Preprocessing
For this project, we use text data from three classic authors—Charles Dickens, Jane Austen, and William Shakespeare—to perform author identification. The data preprocessing steps were carefully designed to eliminate biases that could make the task too easy, focusing the model on learning the authors' unique writing styles.

Data Sources:
Charles Dickens & Jane Austen: Texts were sourced from the Gutenberg Corpus.
William Shakespeare: Texts were extracted from a provided .txt file.
Preprocessing Steps:

1. Text Extraction:
For each author, text was extracted starting from Chapter 1, skipping any prefaces or introductions that were not relevant for the task.

2. Dataset Preparation:
Separate training and test datasets were created for each author and saved into .txt files.

3. Named Entity Recognition (NER):
Named entities, such as locations and human names, were identified and removed using NER techniques. This step was crucial in preventing the model from leveraging easy-to-guess entities (like character names) and instead focusing on the author's writing style.

4. Sequence Creation:
The text for each author was split into sequences of 25 words for nltk model and 15 words for bert model. Due to limited data, I tried to make more trianing and testing instances for bert model by splitting text into a shorter sequence length, as BERT was easily getting overtrained. Any sequence containing named entities was excluded to further reduce bias.

5. Data Cleaning:
Additional preprocessing steps included:
 - Removing unwanted characters and patterns.
 - Lemmatization to normalize words.

6. Data Labeling and Structuring:
The preprocessed sequences were labeled according to the respective author. The training and test sets for all authors were then combined and shuffled.

7. Final Datasets:
The preprocessed data was saved into two files:
 - train_df.csv: Training dataset with labeled sequences.
 - test_df.csv: Test dataset with labeled sequences.
