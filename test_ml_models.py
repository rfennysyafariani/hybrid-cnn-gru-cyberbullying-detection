import unittest
import numpy as np
import pandas as pd
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from utils_hybrid_ml import HybridMachineLearningModels

class TestHybridMLModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\nSetting up ML Model tests...")
        # Reduce vocab size for speed
        cls.models = HybridMachineLearningModels(max_len=10, vocab_size=100)
        
        cls.texts = [
            "this is a positive text about good things",
            "this is a negative text about bad things",
            "another good positive example fitting the class",
            "terrible bad negative example not fitting well",
            "neutral text just sitting here"
        ] * 20
        cls.labels = ["pos", "neg", "pos", "neg", "neu"] * 20
        
    def test_svm_tfidf(self):
        print("\nTesting SVM with TF-IDF...")
        train_texts, test_texts = self.texts[:80], self.texts[80:]
        train_labels, test_labels = self.labels[:80], self.labels[80:]
        # Signature: train_texts, train_labels, test_texts, test_labels, model_type, embedding_type
        results = self.models.train_and_evaluate(train_texts, train_labels, test_texts, test_labels, 'SVM', 'TF-IDF')
        print(f"SVM-TFIDF Results: {results['metrics']}")
        self.assertIn('accuracy', results['metrics'])

    def test_lda_bow(self):
        print("\nTesting LDA with BoW...")
        train_texts, test_texts = self.texts[:80], self.texts[80:]
        train_labels, test_labels = self.labels[:80], self.labels[80:]
        results = self.models.train_and_evaluate(train_texts, train_labels, test_texts, test_labels, 'LDA', 'BoW')
        print(f"LDA-BoW Results: {results['metrics']}")
        self.assertIn('accuracy', results['metrics'])

    def test_svm_lda_word2vec(self):
        print("\nTesting SVM-LDA Hybrid with Word2Vec...")
        train_texts, test_texts = self.texts[:80], self.texts[80:]
        train_labels, test_labels = self.labels[:80], self.labels[80:]
        results = self.models.train_and_evaluate(train_texts, train_labels, test_texts, test_labels, 'SVM-LDA', 'Word2Vec')
        print(f"SVM-LDA-W2V Results: {results['metrics']}")
        self.assertIn('accuracy', results['metrics'])

    def test_bert_embedding(self):
        print("\nTesting BERT embedding (this might take a moment)...")
        # Just run one model to verify connection
        train_texts, test_texts = self.texts[:80], self.texts[80:]
        train_labels, test_labels = self.labels[:80], self.labels[80:]
        results = self.models.train_and_evaluate(train_texts, train_labels, test_texts, test_labels, 'SVM', 'BERT')
        print(f"BERT-SVM Results: {results['metrics']}")
        self.assertIn('accuracy', results['metrics'])

    def test_elmo_embedding(self):
        print("\nTesting ELMo embedding...")
        if self.models.elmo_model is None:
            print("ELMo model not loaded (expected in this env), but code path should handle it safely.")
        train_texts, test_texts = self.texts[:80], self.texts[80:]
        train_labels, test_labels = self.labels[:80], self.labels[80:]
        results = self.models.train_and_evaluate(train_texts, train_labels, test_texts, test_labels, 'SVM', 'ELMo')
        print(f"ELMo-SVM Results: {results['metrics']}")
        self.assertIn('accuracy', results['metrics'])

    def test_all_combinations_smoke(self):
        """Quick smoke test for all combinations to ensure no shape errors"""
        embeddings = ['TF-IDF', 'BoW', 'Word2Vec', 'GloVe'] # Skip heavy BERT/ELMo for loop
        models = ['SVM', 'LDA', 'SVM-LDA']
        
        train_texts, test_texts = self.texts[:80], self.texts[80:]
        train_labels, test_labels = self.labels[:80], self.labels[80:]
        
        for emb in embeddings:
            for mod in models:
                with self.subTest(embedding=emb, model=mod):
                    print(f"Testing {mod} with {emb}...")
                    try:
                        # Signature: ..., model, embedding
                        res = self.models.train_and_evaluate(train_texts, train_labels, test_texts, test_labels, mod, emb)
                        self.assertIsNotNone(res)
                    except Exception as e:
                        self.fail(f"Failed {mod} with {emb}: {e}")

if __name__ == '__main__':
    unittest.main()
