# Hybrid CNN–GRU Architectures with Contextual Embeddings for Cyberbullying Detection

## Description

This repository contains the complete code and datasets for the paper:

> **"Hybrid CNN–GRU Architectures with Contextual Embeddings for Cyberbullying Detection"**
> Syafariani, F., Lola, M. S., et al. *PeerJ Computer Science*, 2025.

This project implements and evaluates **six hybrid deep learning configurations** combining CNN and GRU for cyberbullying detection on X (formerly Twitter):

| Model | Description |
|-------|-------------|
| CNN-Softmax | CNN with Softmax classifier (baseline) |
| GRU-Softmax | GRU with Softmax classifier (baseline) |
| CNN-GRU-Softmax | CNN then GRU with Softmax |
| CNN-GRU-SVM | CNN then GRU with SVM |
| GRU-CNN-Softmax | GRU then CNN with Softmax |
| GRU-CNN-SVM | GRU then CNN with SVM |

Text representations: TF-IDF, Word2Vec, BERT.

---

## Dataset Information

Three datasets collected from X via Twitter API using Python, Google Colaboratory, and Tweepy:

| # | File | Rows | Labels | Class Balance |
|---|------|------|--------|---------------|
| 1 | dataset1 - twitter_racism_parsed_dataset.csv | 13,471 | racism / none | 14.6% positive (severe imbalance) |
| 2 | dataset2 - cyberbullying_tweets.csv | 47,692 | 6 categories to binary | 83.5% positive (inverse imbalance) |
| 3 | dataset3.csv | 8,452 | Bullying / Not-Bullying | 57.2% positive (balanced) |

**Dataset 1** — Racism/hate speech tweets. Avg. 14.9 words/tweet. Ref: Elsafoury et al. (2021), DOI: 10.1109/ACCESS.2021.3098979

**Dataset 2** — Multi-category cyberbullying (religion, age, gender, ethnicity, other, not_cyberbullying). Avg. 23.7 words/tweet. Ref: Wang et al. (2020), DOI: 10.1109/BigData50022.2020.9378065

**Dataset 3** — Multi-type bullying (Sexual, Troll, Political, Vocational, Religion, Threats, Ethnicity). Avg. 9.5 words/sample.

---

## Repository Structure

```
.
├── Datasets/
│   ├── dataset1 - twitter_racism_parsed_dataset.csv
│   ├── dataset2 - cyberbullying_tweets.csv
│   └── dataset3.csv
├── Generate Datasets/
│   └── batch2.json
├── run.py
├── utils_hybrid_dl.py
├── utils_hybrid_ml.py
├── generate_dataset.py
├── test_implementation.py
├── test_individual_models.py
├── test_ml_models.py
├── requirements.txt
└── README.md
```

---

## Requirements

Python 3.11.5. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python run.py <model_type> <data_name>
```

model_type: dl (Deep Learning) or ml (Machine Learning)
data_name: dataset1, dataset2, dataset3, simdataset

Examples:
```bash
python run.py dl dataset1
python run.py dl dataset2
python run.py dl dataset3
python run.py ml dataset1
```

Results saved as: Results_<model_type>_<dataset>.xlsx

---

## Preprocessing Pipeline

1. Lowercase normalization
2. Noise removal (URLs, mentions, hashtags, numbers, special chars)
3. Stop word removal (NLTK)
4. Tokenization (NLTK word_tokenize)
5. Lemmatization (NLTK WordNetLemmatizer)
6. Detokenization (TreebankWordDetokenizer)

## Training Configuration

- Stratified 5-fold cross-validation
- Epochs: 10, Batch size: 32
- Adam optimizer (lr=0.001), Binary cross-entropy loss
- Dropout: 0.5, Random seed: 42
- SVM: RBF kernel, C=1.0, gamma=scale

---

## Citation

Syafariani, F., Lola, M. S., et al. (2025). Hybrid CNN-GRU Architectures with Contextual Embeddings for Cyberbullying Detection. PeerJ Computer Science.

Related: Syafariani, F., et al. (2025). Leveraging a hybrid machine learning model for enhanced cyberbullying detection. ATT, 7(2). https://doi.org/10.34306/att.v7i2.536

---

## Contact

Corresponding author: Muhamad Safiih Lola — safiihmd@umt.edu.my
