# Text Classification with Neural Networks  

This repository contains the code and notebooks for the **Text Classification with Neural Networks** project. The project explores various neural network architectures for text classification, as well as traditional machine learning algorithms such as Naive Bayes and Logistic Regression.  

## Project Objectives  
- Utilize pretrained word embeddings (e.g., Spacy embeddings).  
- Train and evaluate different models for text classification:
  - Naive Bayes and Logistic Regression (using bag-of-words representation).  
  - Multilayer Perceptrons (MLPs).  
  - Recurrent Neural Networks (LSTMs).  
  - Transformer-based models (using the HuggingFace library).  
- Experiment with text classification on a dataset of incident descriptions and compare performance across models.  

## Dataset  
This project uses datasets containing textual descriptions of incidents categorized by type. The datasets include:  
- **Training**: `data/incidents_train.json`, `data/t1_train.json`.  
- **Validation**: `data/incidents_dev.json`.  
- **Test**: `data/incidents_test.json`, `data/t1_test.json`.  

Each file contains textual descriptions of incidents, where labels correspond to integer-encoded incident types.  

## Tasks Overview  

### Task 1: Naive Bayes and Logistic Regression  
- **Objective**: Classify text descriptions using traditional machine learning models with bag-of-words representation.  
- **Key Features**:  
  - Compare Naive Bayes and Logistic Regression classifiers.  
  - Evaluate performance using accuracy and confusion matrices.  
  - Analyze the impact of normalization techniques (lemmatization vs. no normalization).  
  - Propose labels for incident types based on model weights.
- **Notebook**: [t1_classifier_incidents.ipynb](t1_classifier_incidents.ipynb)
  
### Task 2: Feedforward Network (MLP)
- **Objective**: Classify text descriptions using an MLP with word embeddings.
- **Key Features**:  
  - Tokenization and embeddings from Spacy.  
  - Pooling strategies (max, average, min).  
  - Analysis of results and hyperparameter tuning.  
- **Notebook**: [mlp.ipynb](mlp.ipynb)

### Task 3: Recurrent Neural Networks (LSTM)

- **Objective**: Classify text descriptions using an LSTM.
- **Key Features**:
  - Compare unidirectional and bidirectional LSTMs.
  - Tokenization and embeddings from Spacy.
  - Analyze model architecture and performance.

#### Notebooks:
  - **[rnn_uni.ipynb](rnn_uni.ipynb)**: This notebook focuses on the unidirectional LSTM model, which processes the input sequence in one direction.
  - **[rnn_bi.ipynb](rnn_bi.ipynb)**: This notebook explores the bidirectional LSTM model, which processes the input sequence in both directions to capture context from both past and future tokens.

Both models are trained and evaluated for performance comparison.

### Task 4: Transformer-Based Models
- **Objective**: Use transformer models (BERT and another model) for text classification.  
- **Key Features**:  
  - Tokenization and embeddings from HuggingFace.  
  - Compare results between BERT and a significantly different transformer.  
  - Analyze performance against previous tasks.
- **Notebook**: [transformer.ipynb](transformer.ipynb)

## Results and Analysis  
### Model Performance Summary  
| Model            | Accuracy |  
|------------------|----------|  
| Logistic Reg.    | 0.72     |  
| Naive Bayes      | 0.71     |  
| MLP (AVG pool)   | 0.77     |  
| RNN (unidirectional) | 0.67 |  
| RNN (bidirectional)  | 0.64 |  
| BERT             | 0.76     |  
| GPT-2            | 0.74     |  

### Observations  
The results indicate a variety of performances among the models. Simpler models like Logistic Regression and Naive Bayes perform well, while the MLP with average pooling achieves the best accuracy. RNNs, despite being designed for sequential data, yield weaker results, especially for the bidirectional version. Surprisingly, advanced language models like BERT and GPT-2 do not significantly outperform simpler methods, suggesting that for this task, increased complexity does not necessarily lead to better performance. 
