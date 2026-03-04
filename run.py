# import packages Python 3.13.1
import re
import sys
import os
import logging
import glob

# Suppress TensorFlow Logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import word_tokenize
from utils_hybrid_dl import run_dl_experiments, split_dataset
from utils_hybrid_ml import run_ml_experiments 


# normalize text to lower case
def lower_case(text):
    return text.lower()

# remove unicode characters, special characters, links, numbers, mentions, and hashtags
def remove_unic(text):
    text = re.sub(r"(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|([0-9]+)|(\w+:\/\/\S+)|^rt|http.+?", "", text)
    return text

# remove stop words
def remove_stop(text):
    stop = stopwords.words('english')
    text = " ".join([word for word in text.split() if word not in (stop)])
    return text

# tokenize text
def tokenize(text):
    text = word_tokenize(text)
    return text

# lemmatization
def Lemmatize(text):
    word_lem = WordNetLemmatizer()
    text = [word_lem.lemmatize(token) for token in text]
    return text

# detokenize text
def sen_tokenize(text):
    text = TreebankWordDetokenizer().detokenize(text)
    return text

# complete preprocessing function
def preprocess_text(text):
    """
    Preprocess text data by converting to lower case, removing unicode characters, 
    special characters, links, numbers, mentions, and hashtags, removing stop words, 
    tokenizing, lemmatizing, and detokenizing.
    
    Parameters:
    text (str): Input text data
    
    Returns:
    str: Preprocessed text data
    """
    text = lower_case(text)
    text = remove_unic(text)
    text = remove_stop(text)
    text = tokenize(text)
    text = Lemmatize(text)
    text = sen_tokenize(text)
    return text

# run all experiments
def run(model_type: str, dataset: str):

    if dataset == 'dataset1':
        # load datasets
        df = pd.read_csv('Datasets/dataset1 - twitter_racism_parsed_dataset.csv') # dataset 1
        # change column names to standard format
        df.rename(columns={'Text': 'tweet_text', 'Label': 'labels'}, inplace=True)
        # remove empty "cleantext" rows
        df["cleantext"] = df['tweet_text'].astype(str).apply(preprocess_text) # dataset 1
        df = df[df['cleantext'].str.strip().astype(bool)]

    elif dataset == 'dataset2':
        # load datasets
        df = pd.read_csv('Datasets/dataset2 - cyberbullying_tweets.csv') # dataset 2
        # change column names to standard format
        df['cyberbullying_type'] = df['cyberbullying_type'].apply(lambda x: 0 if x == 'not_cyberbullying' else 1)
        # change column names to standard format
        df.rename(columns={'tweet_text': 'tweet_text', 'cyberbullying_type': 'labels'}, inplace=True)
        # remove empty "cleantext" rows
        df["cleantext"] = df['tweet_text'].astype(str).apply(preprocess_text) # dataset 2
        df = df[df['cleantext'].str.strip().astype(bool)]
    
    elif dataset == 'dataset3':
        # load datasets
        df = pd.read_csv('Datasets/dataset3.csv') # dataset 3
        # change labels to standard format 0 = not cyberbullying, 1 = cyberbullying
        df['Label'] = df['Label'].apply(lambda x: 0 if x == 'Not-Bullying' else 1)
        # change column names to standard format
        df.rename(columns={'Text': 'tweet_text', 'Label': 'labels'}, inplace=True)
        # remove empty "cleantext" rows
        df["cleantext"] = df['tweet_text'].astype(str).apply(preprocess_text) # dataset 3
        df = df[df['cleantext'].str.strip().astype(bool)]

    elif dataset == 'simdataset':
        # load data from "Generate Datasets"
        files = glob.glob('Generate Datasets/*.json')
        df = pd.concat([pd.read_json(f) for f in files], ignore_index=True)
        # change labels to standard format 0 = not cyberbullying, 1 = cyberbullying
        df['label'] = df['label'].apply(lambda x: 0 if x == 'not CYBERBULLYING' else 1)
        # change column names to standard format
        df.rename(columns={'text': 'tweet_text', 'label': 'labels'}, inplace=True)
        # remove empty "cleantext" rows
        df["cleantext"] = df['tweet_text'].astype(str).apply(preprocess_text) # simdataset
        df = df[df['cleantext'].str.strip().astype(bool)]
    else:
        raise ValueError("Invalid dataset name, i.q. dataset1, dataset2, dataset3, simdataset")

    # create review_len
    # df1['review_len'] = df1['cleantext'].astype(str).apply(len) # dataset 1
    # df2['review_len'] = df2['cleantext'].astype(str).apply(len) # dataset 2
    # df3['review_len'] = df3['cleantext'].astype(str).apply(len) # dataset 3

    # create word_count
    # df1['word_count'] = df1['cleantext'].apply(lambda x: len(str(x).split()))
    # df2['word_count'] = df2['cleantext'].apply(lambda x: len(str(x).split()))
    # df3['word_count'] = df3['cleantext'].apply(lambda x: len(str(x).split()))

    # if dataset == 'dataset1':
    #     df = df1
    # elif dataset == 'dataset2':
    #     df = df2
    # elif dataset == 'dataset3':
    #     df = df3

    # split datasets
    train_texts, test_texts, train_labels, test_labels = split_dataset(df["cleantext"], df["labels"], 
                                                                    test_size=0.2, random_state=42)

    if model_type == "ml":
        model_types = ["SVM", "LDA", "SVM-LDA"]
        embedding_types = ["TF-IDF", "Word2Vec", "BoW", "GloVe", "BERT", "ELMo"]
        results, results_df = run_ml_experiments(train_texts, test_texts, 
                                                    train_labels, test_labels, 
                                                    model_types, embedding_types, regression=True)
    

    elif model_type == "dl":
        model_types = ["CNN-Softmax", "GRU-Softmax", "CNN-GRU-Softmax", "CNN-GRU-SVM", "GRU-CNN-Softmax", "GRU-CNN-SVM"]
        embedding_types = ["TF-IDF", "Word2Vec", "BoW", "GloVe", "BERT", "ELMo"]
        results, results_df = run_dl_experiments(train_texts, test_texts, 
                                                    train_labels, test_labels, 
                                                    model_types, embedding_types, regression=True)
    else:
        raise ValueError("Invalid model type. Must be 'ml' or 'dl'.")

    # save results to excel
    results_df.to_excel(f'Results_{model_type}_{dataset}.xlsx', index=False)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SyntaxError("Usage: python3 run.py model_type data_name")
    run(sys.argv[1], sys.argv[2])