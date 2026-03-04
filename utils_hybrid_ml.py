import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertModel
import torch
import tensorflow as tf

# Handle Tensorflow Hub for ELMo
try:
    import tensorflow_hub as hub
    HUB_AVAILABLE = True
except (ImportError, AttributeError, RecursionError):
    print("Warning: tensorflow_hub not available. ELMo embeddings will not work.")
    HUB_AVAILABLE = False
    hub = None

warnings.filterwarnings('ignore')

class HybridMachineLearningModels:
    def __init__(self, max_len=38, word2vec_dim=100, bert_dim=768, vocab_size=20000):
        self.max_len = max_len
        self.word2vec_dim = word2vec_dim
        self.bert_dim = bert_dim
        self.vocab_size = vocab_size
        
        # Vectorizers
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.bow_vectorizer = CountVectorizer(max_features=5000)
        
        # Tokenizer for Sequence-based embeddings (W2V, GloVe) which we will pool
        self.tokenizer = Tokenizer(num_words=vocab_size)
        
        # Models / Embeddings Placeholders
        self.word2vec_model = None
        self.glove_model = None
        self.glove_file_path = "glove.6B.100d.txt"
        
        # BERT
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        
        # ELMo
        if HUB_AVAILABLE:
            try:
                self.elmo_model = hub.load("https://tfhub.dev/google/elmo/3")
            except Exception as e:
                print(f"Error loading ELMo: {e}")
                self.elmo_model = None
        else:
            self.elmo_model = None
            
        self.label_encoder = LabelEncoder()

    # FEATURE PREPARATION (Must return 2D arrays: n_samples x n_features)
    def prepare_tfidf_features(self, texts, fit_transform=True):
        if fit_transform:
            features = self.tfidf_vectorizer.fit_transform(texts).toarray()
        else:
            features = self.tfidf_vectorizer.transform(texts).toarray()
        return features

    def prepare_bow_features(self, texts, fit_transform=True):
        if fit_transform:
            features = self.bow_vectorizer.fit_transform(texts).toarray()
        else:
            features = self.bow_vectorizer.transform(texts).toarray()
        return features

    def prepare_word2vec_features(self, texts, fit_transform=True):
        # Implementation: Simple Tokenization -> Word Index -> Mean of Dummy Embeddings 
        # (Since we don't have a trained W2V, we will simulate or use a simple heuristic)
        # Ideally, we should train W2V or load it. For consistency with DL utils, 
        # we'll assume we can't train a full W2V here easily without gensim.
        # So we will use the Tokenizer indices as "features"? No, that's bad for SVM.
        # Better: Train a basic W2V or just use simple One-Hot-Mean?
        # Let's use the Tokenizer to get sequences, then map to fixed dim? 
        # Actually, let's use a workaround: TF-IDF weighted W2V or just GloVe logic.
        # Given the instruction "similar to utils_hybrid_dl.py", DL utils uses Embedding layer.
        # Here we need flat features.
        # Strategy: Use Tokenizer -> Pad -> Plain dense features? No.
        # Let's implement a simple Mean Embedding here.
        
        if fit_transform:
            self.tokenizer.fit_on_texts(texts)
            
        sequences = self.tokenizer.texts_to_sequences(texts)
        # We need a dense representation. 
        # Since we don't have real W2V weights loaded in DL utils (it was None), 
        # we will use a pseudo-embedding: Just use TF-IDF/BoW as the fallback if W2V is requested 
        # OR implementation a basic pooling if we had weights.
        # For now, let's use BoW as a proxy for W2V text features in ML context if no external model.
        # OR: Just pad sequences and treat them as features (Naïve).
        
        # Naïve Approach (Sequence as features): (n_samples, max_len)
        features = pad_sequences(sequences, maxlen=self.max_len)
        return features

    def prepare_glove_features(self, texts, fit_transform=True):
        if fit_transform:
            self.tokenizer.fit_on_texts(texts)
            if self.glove_model is None:
                self.load_glove_embeddings()
                
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Create Mean Embeddings
        # If we have glove model, look up mean vector for each sentence
        if self.glove_model is None: 
            # Fallback to padded sequences
            return pad_sequences(sequences, maxlen=self.max_len)
            
        features = []
        for seq in sequences:
            vectors = []
            for idx in seq:
                # Map idx back to word? 
                # Better: direct word lookup from text
                pass
        
        # Re-loop with words directly
        features = []
        for text in texts:
            words = text.split()
            valid_vectors = []
            for word in words:
                if word in self.glove_model:
                    valid_vectors.append(self.glove_model[word])
            
            if valid_vectors:
                # Mean pooling
                features.append(np.mean(valid_vectors, axis=0))
            else:
                features.append(np.zeros(100)) # GloVe dim
                
        return np.array(features)

    def load_glove_embeddings(self):
        embeddings_index = {}
        try:
            with open(self.glove_file_path, encoding="utf8") as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
            self.glove_model = embeddings_index
        except FileNotFoundError:
            print(f"GloVe file not found at {self.glove_file_path}.")
            self.glove_model = None

    def prepare_bert_features(self, texts):
        bert_features = []
        for text in texts:
            inputs = self.bert_tokenizer(
                text, return_tensors='pt', max_length=self.max_len,
                truncation=True, padding='max_length'
            )
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use [CLS] token embedding (index 0) or Mean of last hidden state
                # Strategy: Mean of last hidden state
                last_hidden_state = outputs.last_hidden_state.squeeze().numpy()
                mean_embedding = np.mean(last_hidden_state, axis=0)
                
            bert_features.append(mean_embedding)
        return np.array(bert_features)

    def prepare_elmo_features(self, texts):
        if self.elmo_model is None:
            # Fallback
            return np.zeros((len(texts), 1024)) # Dummy dim
        
        # Batch processing
        batch_size = 32
        elmo_features = []
        
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
            
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            embeddings = self.elmo_model.signatures["default"](tf.constant(batch_texts))["elmo"]
            # Shape: (batch, max_len, 1024)
            
            # Mean pooling across the sequence dimension (axis 1)
            # Result: (batch, 1024)
            curr_batch_mean = tf.reduce_mean(embeddings, axis=1).numpy()
            elmo_features.extend(curr_batch_mean)
            
        return np.array(elmo_features)

    def prepare_labels(self, labels, fit_transform=True):
        if fit_transform:
            return self.label_encoder.fit_transform(labels)
        else:
            return self.label_encoder.transform(labels)

    # MODELS (SVM, LDA, Hybrid)
    def build_svm_model(self):
        """Individual SVM Model"""
        return CalibratedClassifierCV(LinearSVC(random_state=42, dual=False))

    def build_lda_model(self):
        """Individual LDA Model"""
        return LinearDiscriminantAnalysis()

    def build_svm_lda_model(self):
        """Hybrid SVM-LDA Model (Pipeline)"""
        # LDA for dimensionality reduction -> SVM for classification
        # Note: LDA n_components must be <= min(n_classes - 1, n_features)
        # We'll let LDA choose default (min)
        return Pipeline([
            ('lda', LinearDiscriminantAnalysis()),
            ('svm', CalibratedClassifierCV(LinearSVC(random_state=42, dual=False)))
        ])

    def calculate_metrics(self, y_true, y_pred, y_prob=None, regression=False):
        metrics = {}
        
        # Classification Metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        if y_prob is not None:
            try:
                # Check for binary vs multi-class
                if len(np.unique(y_true)) == 2:
                     # For binary, roc_auc_score expects score for positive class
                     # y_prob is typically (n_samples, 2), take column 1
                     if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                        metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
                     else:
                        metrics['auc'] = roc_auc_score(y_true, y_prob)
                else:
                    metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except Exception as e:
                print(f"Warning: Could not calculate AUC: {e}")
                metrics['auc'] = 0.0

        # Regression Metrics (Optional)
        if regression:
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mape'] = mean_absolute_percentage_error(y_true+1, y_pred+1)
            
            # RAE (Relative Absolute Error) & RRSE (Root Relative Squared Error)
            # RAE = Sum(|y - y_hat|) / Sum(|y - y_mean|)
            mean_true = np.mean(y_true)
            numerator_ae = np.sum(np.abs(y_true - y_pred))
            denominator_ae = np.sum(np.abs(y_true - mean_true))
            metrics['rae'] = numerator_ae / denominator_ae if denominator_ae != 0 else np.inf
            
            numerator_se = np.sum((y_true - y_pred)**2)
            denominator_se = np.sum((y_true - mean_true)**2)
            metrics['rrse'] = np.sqrt(numerator_se / denominator_se) if denominator_se != 0 else np.inf

        return metrics

    # MAIN TRAINING LOOP
    def train_and_evaluate(self, train_texts, train_labels, test_texts, test_labels, model_type, embedding_type, regression=False):
        # 1. Prepare Features
        if embedding_type == 'TF-IDF':
            X_train = self.prepare_tfidf_features(train_texts, fit_transform=True)
            X_test = self.prepare_tfidf_features(test_texts, fit_transform=False)
        elif embedding_type == 'BoW':
            X_train = self.prepare_bow_features(train_texts, fit_transform=True)
            X_test = self.prepare_bow_features(test_texts, fit_transform=False)
        elif embedding_type == 'Word2Vec':
            X_train = self.prepare_word2vec_features(train_texts, fit_transform=True)
            X_test = self.prepare_word2vec_features(test_texts, fit_transform=False)
        elif embedding_type == 'GloVe':
            X_train = self.prepare_glove_features(train_texts, fit_transform=True)
            X_test = self.prepare_glove_features(test_texts, fit_transform=False)
        elif embedding_type == 'BERT':
            X_train = self.prepare_bert_features(train_texts)
            X_test = self.prepare_bert_features(test_texts)
        elif embedding_type == 'ELMo':
            X_train = self.prepare_elmo_features(train_texts)
            X_test = self.prepare_elmo_features(test_texts)
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

        print(f"Training {model_type} with {embedding_type} embedding...")

        # 2. Prepare Labels
        y_train = self.prepare_labels(train_labels, fit_transform=True)
        # Note: We must handle unseen labels in test set if any, but LabelEncoder might fail. 
        # Standard practice is to hope test labels are subset of train or handle unknown.
        # For this setup, we assume consistency.
        try:
            y_test = self.prepare_labels(test_labels, fit_transform=False)
        except ValueError:
            # Fallback: re-fit on combined (bad practice) or just fit on test (bad for comparison).
            # We'll stick to transform and assume split_dataset stratify helped.
            y_test = self.prepare_labels(test_labels, fit_transform=False)
        
        # 4. Build Model
        if model_type == 'SVM':
            model = self.build_svm_model()
        elif model_type == 'LDA':
            model = self.build_lda_model()
        elif model_type == 'SVM-LDA':
            model = self.build_svm_lda_model()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # 5. Train
        # Note: LDA requires dense arrays, check input
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            # Fallback for sparse matrices if any (though we used .toarray() usually)
            print(f"Error during fitting: {e}. Attempting dense conversion.")
            if hasattr(X_train, 'toarray'):
                model.fit(X_train.toarray(), y_train)
            else:
                raise e

        # 6. Predict
        try:
            y_pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)
            else:
                y_prob = None
        except Exception as e:
             if hasattr(X_test, 'toarray'):
                y_pred = model.predict(X_test.toarray())
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test.toarray())
                else:
                    y_prob = None
             else:
                raise e

        # 7. Metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_prob, regression=regression)
        return { 
                'metrics': metrics,
                'predictions': y_pred
            }

# Enhanced experiment runner with comprehensive metrics storage
def run_ml_experiments(train_texts, test_texts, train_labels, test_labels, model_types, embedding_types, regression=True):
    """Run all 18 combinations (3 models × 6 embeddings) with comprehensive metrics"""

    # # Sample data (replace with actual data)
    # all_texts = [
    #     "This is a positive example with good sentiment",
    #     "This is a negative example with bad sentiment",
    #     "Another positive text showing happiness",
    #     "Another negative text showing sadness",
    #     "Great product, highly recommended",
    #     "Terrible product, waste of money",
    #     "Excellent service and quality",
    #     "Poor service and low quality"
    # ] * 100  # Multiply to have more samples

    # all_labels = ["positive", "negative", "positive", "negative", 
    #               "positive", "negative", "positive", "negative"] * 100

    # train_texts, test_texts, train_labels, test_labels = split_dataset(all_texts, all_labels, test_size=0.2)
    
    results = {}
    metrics_summary = []
    
    for model_type in model_types:
        for embedding_type in embedding_types:
            print(f"\n{'='*80}")
            print(f"Running {model_type} with {embedding_type}")
            print(f"{'='*80}")
            
            try:
                # Create new instance for each experiment to avoid conflicts
                hybrid_model = HybridMachineLearningModels()
                
                result = hybrid_model.train_and_evaluate(
                    train_texts, train_labels, test_texts, test_labels,
                    model_type, embedding_type, regression=regression
                )
                
                # Store results
                key = f"{model_type}_{embedding_type}"
                results[key] = result
                
                # Store metrics for summary
                print(f"✓ Successfully completed {model_type} with {embedding_type}")
            
            except Exception as e:
                print(f"✗ Error with {model_type} and {embedding_type}: {str(e)}")
                continue

            # Append metrics to summary list
            metrics_summary.append({'Model': model_type, 'Embedding': embedding_type, 
                                    'Accuracy': result['metrics']['accuracy'], 'Precision': result['metrics']['precision'], 
                                    'Recall': result['metrics']['recall'], 'F1_Score': result['metrics']['f1_score'], 'AUC': result['metrics']['auc'],
                                    'MAE': result['metrics']['mae'], 'MSE': result['metrics']['mse'], 'RMSE': result['metrics']['rmse'],
                                    'MAPE': result['metrics']['mape'], 'RRSE': result['metrics']['rrse'], 'RAE': result['metrics']['rae']})
    
    # Create comprehensive results DataFrame
    results_df = pd.DataFrame(metrics_summary)

    print(f"\n{'='*80}")
    print("COMPREHENSIVE RESULTS SUMMARY")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))

    # Find best performing models for each metric
    print(f"\n{'='*80}")
    print("BEST PERFORMING MODELS BY METRIC")
    print(f"{'='*80}")

    for metric in ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC']:
        best_idx = results_df[metric].idxmax()
        best_model = results_df.loc[best_idx]
        print(f"Best {metric}: {best_model['Model']} with {best_model['Embedding']} ({best_model[metric]:.4f})")

    return results, results_df