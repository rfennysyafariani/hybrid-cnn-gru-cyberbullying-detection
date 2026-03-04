
import pandas as pd
import numpy as np
import tensorflow as tf
from utils_hybrid_dl import HybridDeepLearningModels

# Dummy data
train_texts = ["this is positive", "this is negative", "another positive text", "terrible experience"]
train_labels = ["positive", "negative", "positive", "negative"]
test_texts = ["positive result", "negative result"]
test_labels = ["positive", "negative"]

print("Initializing model...")
hdlm = HybridDeepLearningModels(max_len=10)

embeddings = ["TF-IDF", "Word2Vec", "BERT", "BoW", "GloVe", "ELMo"]
models = ["CNN-Softmax", "GRU-Softmax"]

for emb in embeddings:
    for model_name in models:
        print(f"\nTesting {model_name} with {emb}...")
        try:
            hdlm.train_and_evaluate(
                train_texts, train_labels, test_texts, test_labels,
                model_type=model_name, embedding_type=emb, epochs=1, batch_size=2
            )
            print(f"PASS: {model_name} with {emb}")
        except Exception as e:
            print(f"FAIL: {model_name} with {emb}")
            print(e)
            import traceback
            traceback.print_exc()
