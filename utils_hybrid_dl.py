import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, GRU, LSTM, Flatten, Dropout, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertModel
import torch
try:
    import tensorflow_hub as hub
    HUB_AVAILABLE = True
except (ImportError, AttributeError, RecursionError):
    print("Warning: tensorflow_hub not available. ELMo embeddings will not work.")
    HUB_AVAILABLE = False
    hub = None
    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
# import gensim
import warnings

warnings.filterwarnings('ignore')

class HybridDeepLearningModels:
    def __init__(self, max_len=38, word2vec_dim=100, bert_dim=768, vocab_size=20000):
        self.max_len = max_len
        self.word2vec_dim = word2vec_dim
        self.bert_dim = bert_dim
        self.vocab_size = vocab_size
        self.input_shape = (max_len,)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        self.tokenizer = Tokenizer(num_words=vocab_size)
        self.word2vec_model = None
        self.glove_model = None # Placeholder for GloVe dictionary
        if HUB_AVAILABLE:
            try:
                self.elmo_model = hub.load("https://tfhub.dev/google/elmo/3")
            except Exception as e:
                print(f"Error loading ELMo: {e}")
                self.elmo_model = None
        else:
            self.elmo_model = None
            
        self.bow_vectorizer = CountVectorizer(max_features=5000)
        self.glove_file_path = "glove.6B.100d.txt" # Placeholder path
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.label_encoder = LabelEncoder()
        
    def prepare_tfidf_features(self, texts, fit_transform=True):
        """Prepare TF-IDF features - reshape to sequence format"""
        if fit_transform:
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts).toarray()
        else:
            tfidf_features = self.tfidf_vectorizer.transform(texts).toarray()
        
        # Reshape TF-IDF to sequence format for consistency
        # Take top features and reshape to (samples, max_len, features_per_timestep)
        # features_per_timestep = tfidf_features.shape[1] // self.max_len
        features_per_timestep = self.tfidf_vectorizer.max_features // self.max_len 
        # if features_per_timestep == 0:
        #     features_per_timestep = 1
        
        # Pad or truncate to make it divisible
        target_features = self.max_len * features_per_timestep
        if tfidf_features.shape[1] > target_features:
            tfidf_features = tfidf_features[:, :target_features]
        else:
            padding = target_features - tfidf_features.shape[1]
            tfidf_features = np.pad(tfidf_features, ((0, 0), (0, padding)), mode='constant')
        
        # Reshape to sequence format
        tfidf_features = tfidf_features.reshape(-1, self.max_len, features_per_timestep)
        return tfidf_features
    
    def prepare_bow_features(self, texts, fit_transform=True):
        """Prepare Bag of Words features"""
        if fit_transform:
            bow_features = self.bow_vectorizer.fit_transform(texts).toarray()
        else:
            bow_features = self.bow_vectorizer.transform(texts).toarray()
        
        # Reshape BoW to sequence format (similar to TF-IDF logic for CNN input)
        features_per_timestep = self.bow_vectorizer.max_features // self.max_len
        if self.bow_vectorizer.max_features % self.max_len != 0:
             # Ensure correct sizing if not perfectly divisible, though max_len=38, 5000 features...
             # Let's align with TF-IDF method logic
             features_per_timestep = self.bow_vectorizer.max_features // self.max_len

        target_features = self.max_len * features_per_timestep
        if bow_features.shape[1] > target_features:
            bow_features = bow_features[:, :target_features]
        else:
            padding = target_features - bow_features.shape[1]
            bow_features = np.pad(bow_features, ((0, 0), (0, padding)), mode='constant')

        bow_features = bow_features.reshape(-1, self.max_len, features_per_timestep)
        return bow_features

    def prepare_word2vec_features(self, texts, fit_transform=True):
        """Prepare Word2Vec features"""

        if fit_transform:

            # Tokenize texts
            self.tokenizer.fit_on_texts(texts)  

            # # Train Word2Vec model
            # tokenized_texts = [text.split() for text in texts]
            # self.word2vec_model = gensim.models.Word2Vec(
            #     tokenized_texts, vector_size=self.word2vec_dim, 
            #     window=5, min_count=1, workers=4
            # )
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        word2vec_features = pad_sequences(sequences, maxlen=self.max_len)
        
        return word2vec_features
    
    def prepare_glove_features(self, texts, fit_transform=True):
        """Prepare GloVe features - Prepare sequences for embedding layer"""
        if fit_transform:
            self.tokenizer.fit_on_texts(texts)
            
        sequences = self.tokenizer.texts_to_sequences(texts)
        glove_features = pad_sequences(sequences, maxlen=self.max_len)
        return glove_features

    def load_glove_embeddings(self):
        """Load GloVe embeddings from file"""
        embeddings_index = {}
        try:
            with open(self.glove_file_path, encoding="utf8") as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
            print("Found %s word vectors." % len(embeddings_index))
        except FileNotFoundError:
            print(f"GloVe file not found at {self.glove_file_path}. Using random initialization.")
            return None
        
        self.glove_model = embeddings_index
        return embeddings_index

    def prepare_bert_features(self, texts):
        """Prepare BERT features"""
        bert_features = []
        for text in texts:
            inputs = self.bert_tokenizer(
                text, return_tensors='pt', max_length=self.max_len,
                truncation=True, padding='max_length'
            )
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use last hidden state
                bert_embedding = outputs.last_hidden_state.squeeze().numpy()
            
            bert_features.append(bert_embedding)
        
        return np.array(bert_features)
    
    def prepare_elmo_features(self, texts):
        """Prepare ELMo features"""
        if self.elmo_model is None:
             print("ELMo model not loaded. Returning zeros.")
             return np.zeros((len(texts), self.max_len, 1024))
             
        # Batch processing to avoid memory issues
        batch_size = 32
        elmo_features = []
        
        # Convert texts to list of strings if they are pandas Series
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
            
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            embeddings = self.elmo_model.signatures["default"](tf.constant(batch_texts))["elmo"]
            # embeddings shape: (batch_size, max_len, 1024)
            
            # If input texts are longer than max_len supported (unlikely with ELMo per sentence),
            # but we need fixed max_len 38. 
            # TF-Hub ELMo usually outputs variable length based on longest in batch if simplified?
            # Actually standard elmo hub "elmo" output is (batch, max_len_in_batch, 1024).
            
            curr_batch_features = embeddings.numpy()
            
            # Resize/Pad/Truncate to self.max_len
            processed_batch = []
            for item in curr_batch_features:
                if item.shape[0] >= self.max_len:
                    processed_batch.append(item[:self.max_len, :])
                else:
                    padding = np.zeros((self.max_len - item.shape[0], item.shape[1]))
                    processed_batch.append(np.vstack([item, padding]))
            
            elmo_features.extend(processed_batch)
            
        return np.array(elmo_features)  
    
    def prepare_labels(self, labels, fit_transform=True):
        """Prepare labels"""
        if fit_transform:
            encoded_labels = self.label_encoder.fit_transform(labels)
        else:
            encoded_labels = self.label_encoder.transform(labels)
        return encoded_labels

    def create_tfidf_embedding_layer(self, input_dim):
        """Create embedding layer for TF-IDF"""
        return Dense(100, activation='relu', name='tfidf_embedding')
    
    def create_word2vec_embedding_layer(self):
        """Create embedding layer for Word2Vec"""
        # embedding_matrix = self.create_embedding_matrix()
        return Embedding(self.vocab_size, self.word2vec_dim, # weights=[embedding_matrix], 
                        input_length=self.max_len, 
                        trainable=False, name='word2vec_embedding')
    
    def create_bow_embedding_layer(self, input_dim):
        """Create embedding layer for BoW (Dense)"""
        return Dense(100, activation='relu', name='bow_embedding')

    def create_glove_embedding_layer(self):
        """Create embedding layer for GloVe"""
        if self.glove_model is None:
            self.load_glove_embeddings()

        embedding_matrix = np.zeros((self.vocab_size, 100)) # 100d
        if self.glove_model is not None:
            for word, i in self.tokenizer.word_index.items():
                if i < self.vocab_size:
                    embedding_vector = self.glove_model.get(word)
                    if embedding_vector is not None:
                        embedding_matrix[i] = embedding_vector

        return Embedding(self.vocab_size, 100, weights=[embedding_matrix], 
                        input_length=self.max_len, 
                        trainable=False, name='glove_embedding')

    def create_elmo_embedding_layer(self):
        """Create embedding layer for ELMo (Dense or just Input projection)"""
        # ELMo gives 1024d vectors. We can project them or just use them.
        return Dense(256, activation='relu', name='elmo_projection')

    def create_bert_embedding_layer(self):
        """Create embedding layer for BERT (identity layer since BERT is pre-embedded)"""
        return Dense(self.bert_dim, activation='linear', name='bert_embedding')

    # Model 1: CNN-GRU -> Softmax
    def cnn_gru_model_tfidf(self, num_classes=2):
        """CNN-GRU model with TF-IDF embedding"""
        inputs = Input(shape=(self.max_len, self.tfidf_vectorizer.max_features // self.max_len))
        embedding_layer = self.create_tfidf_embedding_layer(inputs.shape[-1])
        
        x = embedding_layer(inputs)
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = GRU(500, return_sequences=False)(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
        else:
            outputs = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model
    
    def cnn_gru_model_word2vec(self, num_classes=2):
        """CNN-GRU model with Word2Vec embedding"""
        inputs = Input(shape=self.input_shape)
        embedding_layer = self.create_word2vec_embedding_layer()
        
        x = embedding_layer(inputs)
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = GRU(500, return_sequences=False)(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
        else:
            outputs = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model
    
    def cnn_gru_model_bert(self, num_classes=2):
        """CNN-GRU model with BERT embedding"""
        inputs = Input(shape=(self.max_len, self.bert_dim))
        embedding_layer = self.create_bert_embedding_layer()
        
        x = embedding_layer(inputs)
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = GRU(500, return_sequences=False)(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
        else:
            outputs = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model

    def cnn_gru_model_bow(self, num_classes=2):
        """CNN-GRU model with BoW embedding"""
        # Input shape similar to TF-IDF reshape
        input_dim = self.bow_vectorizer.max_features // self.max_len
        inputs = Input(shape=(self.max_len, input_dim))
        
        embedding_layer = self.create_bow_embedding_layer(input_dim)
        
        x = embedding_layer(inputs)
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = GRU(500, return_sequences=False)(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
        else:
            outputs = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model

    def cnn_gru_model_glove(self, num_classes=2):
        """CNN-GRU model with GloVe embedding"""
        inputs = Input(shape=self.input_shape)
        embedding_layer = self.create_glove_embedding_layer()
        
        x = embedding_layer(inputs)
        # GloVe 100d -> Conv1D
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = GRU(500, return_sequences=False)(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
        else:
            outputs = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model

    def cnn_gru_model_elmo(self, num_classes=2):
        """CNN-GRU model with ELMo embedding"""
        inputs = Input(shape=(self.max_len, 1024)) # ELMo is 1024d
        embedding_layer = self.create_elmo_embedding_layer()
        
        x = embedding_layer(inputs)
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = GRU(500, return_sequences=False)(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
        else:
            outputs = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model

    # Model 2: CNN-GRU -> SVM
    def cnn_gru_svm_tfidf(self, num_classes=2):
        """CNN-GRU feature extractor with TF-IDF"""
        inputs = Input(shape=(self.max_len, self.tfidf_vectorizer.max_features // self.max_len))
        embedding_layer = self.create_tfidf_embedding_layer(inputs.shape[-1])
        
        x = embedding_layer(inputs)
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = GRU(500, return_sequences=False)(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='linear')(x)
            loss = 'categorical_hinge'
        else:
            outputs = Dense(1, activation='linear')(x)
            loss = 'hinge'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model
    
    def cnn_gru_svm_word2vec(self, num_classes=2):
        """CNN-GRU feature extractor with Word2Vec"""
        inputs = Input(shape=self.input_shape)
        embedding_layer = self.create_word2vec_embedding_layer()
        
        x = embedding_layer(inputs)
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = GRU(500, return_sequences=False)(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='linear')(x)
            loss = 'categorical_hinge'
        else:
            outputs = Dense(1, activation='linear')(x)
            loss = 'hinge'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model
    
    def cnn_gru_svm_bert(self, num_classes=2):
        """CNN-GRU feature extractor with BERT"""
        inputs = Input(shape=(self.max_len, self.bert_dim))
        embedding_layer = self.create_bert_embedding_layer()
        
        x = embedding_layer(inputs)
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = GRU(500, return_sequences=False)(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='linear')(x)
            loss = 'categorical_hinge'
        else:
            outputs = Dense(1, activation='linear')(x)
            loss = 'hinge'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model

    def cnn_gru_svm_bow(self, num_classes=2):
        """CNN-GRU SVM with BoW embedding"""
        input_dim = self.bow_vectorizer.max_features // self.max_len
        inputs = Input(shape=(self.max_len, input_dim))
        embedding_layer = self.create_bow_embedding_layer(input_dim)
        
        x = embedding_layer(inputs)
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = GRU(500, return_sequences=False)(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='linear')(x)
            loss = 'categorical_hinge'
        else:
            outputs = Dense(1, activation='linear')(x)
            loss = 'hinge'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model

    def cnn_gru_svm_glove(self, num_classes=2):
        """CNN-GRU SVM with GloVe embedding"""
        inputs = Input(shape=self.input_shape)
        embedding_layer = self.create_glove_embedding_layer()
        
        x = embedding_layer(inputs)
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = GRU(500, return_sequences=False)(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='linear')(x)
            loss = 'categorical_hinge'
        else:
            outputs = Dense(1, activation='linear')(x)
            loss = 'hinge'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model

    def cnn_gru_svm_elmo(self, num_classes=2):
        """CNN-GRU SVM with ELMo embedding"""
        inputs = Input(shape=(self.max_len, 1024))
        embedding_layer = self.create_elmo_embedding_layer()
        
        x = embedding_layer(inputs)
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = GRU(500, return_sequences=False)(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='linear')(x)
            loss = 'categorical_hinge'
        else:
            outputs = Dense(1, activation='linear')(x)
            loss = 'hinge'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model

    # Model 3: GRU-CNN -> Softmax
    def gru_cnn_model_tfidf(self, num_classes=2):
        """GRU-CNN model with TF-IDF embedding"""
        inputs = Input(shape=(self.max_len, self.tfidf_vectorizer.max_features // self.max_len))
        embedding_layer = self.create_tfidf_embedding_layer(inputs.shape[-1])
        
        x = embedding_layer(inputs)
        x = GRU(500, return_sequences=True)(x)
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
        else:
            outputs = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model
    
    def gru_cnn_model_word2vec(self, num_classes=2):
        """GRU-CNN model with Word2Vec embedding"""
        inputs = Input(shape=self.input_shape)
        embedding_layer = self.create_word2vec_embedding_layer()
        
        x = embedding_layer(inputs)
        x = GRU(500, return_sequences=True)(x)
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
        else:
            outputs = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model
    
    def gru_cnn_model_bert(self, num_classes=2):
        """GRU-CNN model with BERT embedding"""
        inputs = Input(shape=(self.max_len, self.bert_dim))
        embedding_layer = self.create_bert_embedding_layer()
        
        x = embedding_layer(inputs)
        x = GRU(500, return_sequences=True)(x)
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
        else:
            outputs = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model

    def gru_cnn_model_bow(self, num_classes=2):
        """GRU-CNN model with BoW embedding"""
        input_dim = self.bow_vectorizer.max_features // self.max_len
        inputs = Input(shape=(self.max_len, input_dim))
        embedding_layer = self.create_bow_embedding_layer(input_dim)
        
        x = embedding_layer(inputs)
        x = GRU(500, return_sequences=True)(x)
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
        else:
            outputs = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model

    def gru_cnn_model_glove(self, num_classes=2):
        """GRU-CNN model with GloVe embedding"""
        inputs = Input(shape=self.input_shape)
        embedding_layer = self.create_glove_embedding_layer()
        
        x = embedding_layer(inputs)
        x = GRU(500, return_sequences=True)(x)
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
        else:
            outputs = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model

    def gru_cnn_model_elmo(self, num_classes=2):
        """GRU-CNN model with ELMo embedding"""
        inputs = Input(shape=(self.max_len, 1024))
        embedding_layer = self.create_elmo_embedding_layer()
        
        x = embedding_layer(inputs)
        x = GRU(500, return_sequences=True)(x)
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
        else:
            outputs = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model

    # Model 4: GRU-CNN -> SVM
    def gru_cnn_svm_tfidf(self, num_classes=2):
        """GRU-CNN feature extractor with TF-IDF"""
        inputs = Input(shape=(self.max_len, self.tfidf_vectorizer.max_features // self.max_len))
        embedding_layer = self.create_tfidf_embedding_layer(inputs.shape[-1])
        
        x = embedding_layer(inputs)
        x = GRU(500, return_sequences=True)(x)
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='linear')(x)
            loss = 'categorical_hinge'
        else:
            outputs = Dense(1, activation='linear')(x)
            loss = 'hinge'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model
    
    def gru_cnn_svm_word2vec(self, num_classes=2):
        """GRU-CNN feature extractor with Word2Vec"""
        inputs = Input(shape=self.input_shape)
        embedding_layer = self.create_word2vec_embedding_layer()
        
        x = embedding_layer(inputs)
        x = GRU(500, return_sequences=True)(x)
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='linear')(x)
            loss = 'categorical_hinge'
        else:
            outputs = Dense(1, activation='linear')(x)
            loss = 'hinge'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model
    
    def gru_cnn_svm_bert(self, num_classes=2):
        """GRU-CNN feature extractor with BERT"""
        inputs = Input(shape=(self.max_len, self.bert_dim))
        embedding_layer = self.create_bert_embedding_layer()
        
        x = embedding_layer(inputs)
        x = GRU(500, return_sequences=True)(x)
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='linear')(x)
            loss = 'categorical_hinge'
        else:
            outputs = Dense(1, activation='linear')(x)
            loss = 'hinge'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model
    
    def gru_cnn_svm_bow(self, num_classes=2):
        """GRU-CNN SVM with BoW embedding"""
        input_dim = self.bow_vectorizer.max_features // self.max_len
        inputs = Input(shape=(self.max_len, input_dim))
        embedding_layer = self.create_bow_embedding_layer(input_dim)
        
        x = embedding_layer(inputs)
        x = GRU(500, return_sequences=True)(x)
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='linear')(x)
            loss = 'categorical_hinge'
        else:
            outputs = Dense(1, activation='linear')(x)
            loss = 'hinge'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model

    def gru_cnn_svm_glove(self, num_classes=2):
        """GRU-CNN SVM with GloVe embedding"""
        inputs = Input(shape=self.input_shape)
        embedding_layer = self.create_glove_embedding_layer()
        
        x = embedding_layer(inputs)
        x = GRU(500, return_sequences=True)(x)
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='linear')(x)
            loss = 'categorical_hinge'
        else:
            outputs = Dense(1, activation='linear')(x)
            loss = 'hinge'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model

    def gru_cnn_svm_elmo(self, num_classes=2):
        """GRU-CNN SVM with ELMo embedding"""
        inputs = Input(shape=(self.max_len, 1024))
        embedding_layer = self.create_elmo_embedding_layer()
        
        x = embedding_layer(inputs)
        x = GRU(500, return_sequences=True)(x)
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='linear')(x)
            loss = 'categorical_hinge'
        else:
            outputs = Dense(1, activation='linear')(x)
            loss = 'hinge'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model

    # Model 5: CNN only
    def cnn_model_tfidf(self, num_classes=2):
        """CNN model with TF-IDF embedding"""
        inputs = Input(shape=(self.max_len, self.tfidf_vectorizer.max_features // self.max_len))
        embedding_layer = self.create_tfidf_embedding_layer(inputs.shape[-1])
        return self._build_cnn_model(inputs, embedding_layer, num_classes)

    def cnn_model_word2vec(self, num_classes=2):
        """CNN model with Word2Vec embedding"""
        inputs = Input(shape=self.input_shape)
        embedding_layer = self.create_word2vec_embedding_layer()
        return self._build_cnn_model(inputs, embedding_layer, num_classes)

    def cnn_model_bert(self, num_classes=2):
        """CNN model with BERT embedding"""
        inputs = Input(shape=(self.max_len, self.bert_dim))
        embedding_layer = self.create_bert_embedding_layer()
        return self._build_cnn_model(inputs, embedding_layer, num_classes)

    def cnn_model_bow(self, num_classes=2):
        """CNN model with BoW embedding"""
        input_dim = self.bow_vectorizer.max_features // self.max_len
        inputs = Input(shape=(self.max_len, input_dim))
        embedding_layer = self.create_bow_embedding_layer(input_dim)
        return self._build_cnn_model(inputs, embedding_layer, num_classes)

    def cnn_model_glove(self, num_classes=2):
        """CNN model with GloVe embedding"""
        inputs = Input(shape=self.input_shape)
        embedding_layer = self.create_glove_embedding_layer()
        return self._build_cnn_model(inputs, embedding_layer, num_classes)

    def cnn_model_elmo(self, num_classes=2):
        """CNN model with ELMo embedding"""
        inputs = Input(shape=(self.max_len, 1024))
        embedding_layer = self.create_elmo_embedding_layer()
        return self._build_cnn_model(inputs, embedding_layer, num_classes)

    def _build_cnn_model(self, inputs, embedding_layer, num_classes):
        x = embedding_layer(inputs)
        x = Conv1D(512, kernel_size=3, activation='relu')(x)
        x = Conv1D(256, kernel_size=3, activation='relu')(x)
        x = Conv1D(128, kernel_size=3, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
        else:
            outputs = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model

    # Model 6: GRU only
    def gru_model_tfidf(self, num_classes=2):
        """GRU model with TF-IDF embedding"""
        inputs = Input(shape=(self.max_len, self.tfidf_vectorizer.max_features // self.max_len))
        embedding_layer = self.create_tfidf_embedding_layer(inputs.shape[-1])
        return self._build_gru_model(inputs, embedding_layer, num_classes)

    def gru_model_word2vec(self, num_classes=2):
        """GRU model with Word2Vec embedding"""
        inputs = Input(shape=self.input_shape)
        embedding_layer = self.create_word2vec_embedding_layer()
        return self._build_gru_model(inputs, embedding_layer, num_classes)

    def gru_model_bert(self, num_classes=2):
        """GRU model with BERT embedding"""
        inputs = Input(shape=(self.max_len, self.bert_dim))
        embedding_layer = self.create_bert_embedding_layer()
        return self._build_gru_model(inputs, embedding_layer, num_classes)

    def gru_model_bow(self, num_classes=2):
        """GRU model with BoW embedding"""
        input_dim = self.bow_vectorizer.max_features // self.max_len
        inputs = Input(shape=(self.max_len, input_dim))
        embedding_layer = self.create_bow_embedding_layer(input_dim)
        return self._build_gru_model(inputs, embedding_layer, num_classes)

    def gru_model_glove(self, num_classes=2):
        """GRU model with GloVe embedding"""
        inputs = Input(shape=self.input_shape)
        embedding_layer = self.create_glove_embedding_layer()
        return self._build_gru_model(inputs, embedding_layer, num_classes)

    def gru_model_elmo(self, num_classes=2):
        """GRU model with ELMo embedding"""
        inputs = Input(shape=(self.max_len, 1024))
        embedding_layer = self.create_elmo_embedding_layer()
        return self._build_gru_model(inputs, embedding_layer, num_classes)

    def _build_gru_model(self, inputs, embedding_layer, num_classes):
        x = embedding_layer(inputs)
        x = GRU(500, return_sequences=False)(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
        if num_classes > 2:
            outputs = Dense(num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
        else:
            outputs = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=['accuracy'])
        return model

    def calculate_metrics(self, y_true, y_pred, average='binary', regression=True):
        """Calculate comprehensive evaluation metrics"""
        # Classification metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
        }
        
        try:
            # For AUC, y_pred should ideally be probabilities but here it's labels
            # If using labels for AUC it's not ideal but better than nothing if probs not available
            metrics['auc'] = roc_auc_score(y_true, y_pred)
        except:
            metrics['auc'] = 0.0

        if regression:
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mape'] = mean_absolute_percentage_error(y_true+1, y_pred+1)
            
            # Calculate RAE and RRSE (manual implementation or generic formulas)
            # RAE = sum(|y - y_hat|) / sum(|y - y_bar|)
            # RRSE = sqrt(sum((y - y_hat)^2) / sum((y - y_bar)^2))
            
            y_bar = np.mean(y_true)
            numerator_rae = np.sum(np.abs(y_true - y_pred))
            denominator_rae = np.sum(np.abs(y_true - y_bar))
            metrics['rae'] = numerator_rae / denominator_rae if denominator_rae != 0 else 0
            
            numerator_rrse = np.sum((y_true - y_pred)**2)
            denominator_rrse = np.sum((y_true - y_bar)**2)
            metrics['rrse'] = np.sqrt(numerator_rrse / denominator_rrse) if denominator_rrse != 0 else 0
            
            # MAPE (Mean Absolute Percentage Error)
            # Avoid division by zero
            #  non_zero = y_true != 0
            #  if np.any(non_zero):
            #      metrics['mape'] = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero]))
            #  else:
            #      metrics['mape'] = 0.0

        return metrics
    
    def train_and_evaluate(self, train_texts, train_labels, test_texts, test_labels, 
                          model_type, embedding_type, epochs=10, batch_size=32, regression=True):
        """Train and evaluate the specified model with specified embedding"""
        
        print(f"Training {model_type} with {embedding_type} embedding...")
        
        # Prepare labels
        y_train = self.prepare_labels(train_labels, fit_transform=True)
        y_test = self.prepare_labels(test_labels, fit_transform=False)
        num_classes = len(np.unique(y_train))
        
        # Prepare features based on embedding type
        if embedding_type == "TF-IDF":
            X_train = self.prepare_tfidf_features(train_texts, fit_transform=True)
            X_test = self.prepare_tfidf_features(test_texts, fit_transform=False)
        elif embedding_type == "Word2Vec":
            X_train = self.prepare_word2vec_features(train_texts, fit_transform=True)
            X_test = self.prepare_word2vec_features(test_texts, fit_transform=False)
        elif embedding_type == "BERT":
            X_train = self.prepare_bert_features(train_texts)
            X_test = self.prepare_bert_features(test_texts)
        elif embedding_type == "BoW":
            X_train = self.prepare_bow_features(train_texts, fit_transform=True)
            X_test = self.prepare_bow_features(test_texts, fit_transform=False)
        elif embedding_type == "GloVe":
            X_train = self.prepare_glove_features(train_texts, fit_transform=True)
            X_test = self.prepare_glove_features(test_texts, fit_transform=False)
        elif embedding_type == "ELMo":
            X_train = self.prepare_elmo_features(train_texts)
            X_test = self.prepare_elmo_features(test_texts)
        else:
            raise ValueError("embedding_type must be 'TF-IDF', 'Word2Vec', 'BERT', 'BoW', 'GloVe', or 'ELMo'")
        
        # Select and create model
        if model_type == "CNN-GRU-Softmax":
            if embedding_type == "TF-IDF":
                model = self.cnn_gru_model_tfidf(num_classes)
            elif embedding_type == "Word2Vec":
                model = self.cnn_gru_model_word2vec(num_classes)
            elif embedding_type == "BERT":
                model = self.cnn_gru_model_bert(num_classes)
            elif embedding_type == "BoW":
                model = self.cnn_gru_model_bow(num_classes)
            elif embedding_type == "GloVe":
                model = self.cnn_gru_model_glove(num_classes)
            else: # ELMo
                model = self.cnn_gru_model_elmo(num_classes)
            
            # Prepare labels for training
            if num_classes > 2:
                y_train_cat = to_categorical(y_train)
                y_test_cat = to_categorical(y_test)
                average_type = 'macro'
            else:
                y_train_cat = y_train
                y_test_cat = y_test
                average_type = 'binary'
            
            # Train model
            history = model.fit(X_train, y_train_cat, 
                              validation_data=(X_test, y_test_cat),
                              epochs=epochs, batch_size=batch_size, verbose=1)
            
            # Evaluate
            test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
            predictions = model.predict(X_test)
            
            if num_classes > 2:
                y_pred = np.argmax(predictions, axis=1)
            else:
                y_pred = (predictions > 0.5).astype(int).flatten()
            
            # Calculate comprehensive metrics
            metrics = self.calculate_metrics(y_test, y_pred, average=average_type, regression=regression)
            
            print(f"Test Accuracy: {metrics['accuracy']:.4f}")
            print(f"Test Precision: {metrics['precision']:.4f}")
            print(f"Test Recall: {metrics['recall']:.4f}")
            print(f"Test F1-Score: {metrics['f1_score']:.4f}")
            print(f"Test AUC: {metrics['auc']:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            return {
                # 'model': model, 
                # 'history': history, 
                'metrics': metrics,
                'predictions': y_pred
            }
        
        elif model_type == "CNN-GRU-SVM":

            # condition for embedding type
            if embedding_type == "TF-IDF":
                model = self.cnn_gru_svm_tfidf(num_classes)
            elif embedding_type == "Word2Vec":
                model = self.cnn_gru_svm_word2vec(num_classes)
            elif embedding_type == "BERT":
                model = self.cnn_gru_svm_bert(num_classes)
            elif embedding_type == "BoW":
                model = self.cnn_gru_svm_bow(num_classes)
            elif embedding_type == "GloVe":
                model = self.cnn_gru_svm_glove(num_classes)
            else: # ELMo
                model = self.cnn_gru_svm_elmo(num_classes)
            
            # Prepare labels for training
            if num_classes > 2:
                y_train_cat = to_categorical(y_train)
                y_test_cat = to_categorical(y_test)
                average_type = 'macro'
            else:
                y_train_cat = y_train
                y_test_cat = y_test
                average_type = 'binary'
            
            # Train model
            history = model.fit(X_train, y_train_cat, 
                              validation_data=(X_test, y_test_cat),
                              epochs=epochs, batch_size=batch_size, verbose=1)
            
            # Evaluate
            test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
            predictions = model.predict(X_test)
            
            if num_classes > 2:
                y_pred = np.argmax(predictions, axis=1)
            else:
                y_pred = (predictions > 0.0).astype(int).flatten()
            
            # Calculate comprehensive metrics
            metrics = self.calculate_metrics(y_test, y_pred, average=average_type, regression=regression)
            
            print(f"Test Accuracy: {metrics['accuracy']:.4f}")
            print(f"Test Precision: {metrics['precision']:.4f}")
            print(f"Test Recall: {metrics['recall']:.4f}")
            print(f"Test F1-Score: {metrics['f1_score']:.4f}")
            print(f"Test AUC: {metrics['auc']:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            return {
                # 'model': model, 
                # 'history': history, 
                'metrics': metrics,
                'predictions': y_pred
            }
        
        elif model_type == "GRU-CNN-Softmax":
            if embedding_type == "TF-IDF":
                model = self.gru_cnn_model_tfidf(num_classes)
            elif embedding_type == "Word2Vec":
                model = self.gru_cnn_model_word2vec(num_classes)
            elif embedding_type == "BERT":
                model = self.gru_cnn_model_bert(num_classes)
            elif embedding_type == "BoW":
                model = self.gru_cnn_model_bow(num_classes)
            elif embedding_type == "GloVe":
                model = self.gru_cnn_model_glove(num_classes)
            else: # ELMo
                model = self.gru_cnn_model_elmo(num_classes)
            
            # Prepare labels for training
            if num_classes > 2:
                y_train_cat = to_categorical(y_train)
                y_test_cat = to_categorical(y_test)
                average_type = 'macro'
            else:
                y_train_cat = y_train
                y_test_cat = y_test
                average_type = 'binary'
            
            # Train model
            history = model.fit(X_train, y_train_cat, 
                              validation_data=(X_test, y_test_cat),
                              epochs=epochs, batch_size=batch_size, verbose=1)
            
            # Evaluate
            test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
            predictions = model.predict(X_test)
            
            if num_classes > 2:
                y_pred = np.argmax(predictions, axis=1)
            else:
                y_pred = (predictions > 0.5).astype(int).flatten()
            
            # Calculate comprehensive metrics
            metrics = self.calculate_metrics(y_test, y_pred, average=average_type, regression=regression)
            
            print(f"Test Accuracy: {metrics['accuracy']:.4f}")
            print(f"Test Precision: {metrics['precision']:.4f}")
            print(f"Test Recall: {metrics['recall']:.4f}")
            print(f"Test F1-Score: {metrics['f1_score']:.4f}")
            print(f"Test AUC: {metrics['auc']:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            return {
                # 'model': model, 
                # 'history': history, 
                'metrics': metrics,
                'predictions': y_pred
            }
        
        elif model_type == "GRU-CNN-SVM":
            
            # condition for embedding type
            if embedding_type == "TF-IDF":
                model = self.gru_cnn_svm_tfidf(num_classes)
            elif embedding_type == "Word2Vec":
                model = self.gru_cnn_svm_word2vec(num_classes)
            elif embedding_type == "BERT":
                model = self.gru_cnn_svm_bert(num_classes)
            elif embedding_type == "BoW":
                model = self.gru_cnn_svm_bow(num_classes)
            elif embedding_type == "GloVe":
                model = self.gru_cnn_svm_glove(num_classes)
            else: # ELMo
                model = self.gru_cnn_svm_elmo(num_classes)
            
            # Prepare labels for training
            if num_classes > 2:
                y_train_cat = to_categorical(y_train)
                y_test_cat = to_categorical(y_test)
                average_type = 'macro'
            else:
                y_train_cat = y_train
                y_test_cat = y_test
                average_type = 'binary'
            
            # Train model
            history = model.fit(X_train, y_train_cat, 
                              validation_data=(X_test, y_test_cat),
                              epochs=epochs, batch_size=batch_size, verbose=1)
            
            # Evaluate
            test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
            predictions = model.predict(X_test)
            
            if num_classes > 2:
                y_pred = np.argmax(predictions, axis=1)
            else:
                y_pred = (predictions > 0.0).astype(int).flatten()
            
            # Calculate comprehensive metrics
            # average_type = 'macro' if num_classes > 2 else 'binary'
            metrics = self.calculate_metrics(y_test, y_pred, average=average_type, regression=regression)
            
            print(f"Test Accuracy: {metrics['accuracy']:.4f}")
            print(f"Test Precision: {metrics['precision']:.4f}")
            print(f"Test Recall: {metrics['recall']:.4f}")
            print(f"Test F1-Score: {metrics['f1_score']:.4f}")
            print(f"Test AUC: {metrics['auc']:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            return {
                # 'model': model, 
                # 'history': history, 
                'metrics': metrics,
                'predictions': y_pred
            }
        
        elif model_type == "CNN-Softmax":
            if embedding_type == "TF-IDF":
                model = self.cnn_model_tfidf(num_classes)
            elif embedding_type == "Word2Vec":
                model = self.cnn_model_word2vec(num_classes)
            elif embedding_type == "BERT":
                model = self.cnn_model_bert(num_classes)
            elif embedding_type == "BoW":
                model = self.cnn_model_bow(num_classes)
            elif embedding_type == "GloVe":
                model = self.cnn_model_glove(num_classes)
            else: # ELMo
                model = self.cnn_model_elmo(num_classes)

            # Common training logic reused
            return self._train_common(model, X_train, y_train, X_test, y_test, num_classes, epochs, batch_size, regression)

        elif model_type == "GRU-Softmax":
            if embedding_type == "TF-IDF":
                model = self.gru_model_tfidf(num_classes)
            elif embedding_type == "Word2Vec":
                model = self.gru_model_word2vec(num_classes)
            elif embedding_type == "BERT":
                model = self.gru_model_bert(num_classes)
            elif embedding_type == "BoW":
                model = self.gru_model_bow(num_classes)
            elif embedding_type == "GloVe":
                model = self.gru_model_glove(num_classes)
            else: # ELMo
                model = self.gru_model_elmo(num_classes)

            # Common training logic reused
            return self._train_common(model, X_train, y_train, X_test, y_test, num_classes, epochs, batch_size, regression)
            
        else:
            raise ValueError("model_type must be 'CNN-GRU-Softmax', 'CNN-GRU-SVM', 'GRU-CNN-Softmax', 'GRU-CNN-SVM', 'CNN-Softmax', or 'GRU-Softmax'")

    def _train_common(self, model, X_train, y_train, X_test, y_test, num_classes, epochs, batch_size, regression):
        # Prepare labels for training
        if num_classes > 2:
            y_train_cat = to_categorical(y_train)
            y_test_cat = to_categorical(y_test)
            average_type = 'macro'
        else:
            y_train_cat = y_train
            y_test_cat = y_test
            average_type = 'binary'
        
        # Train model
        history = model.fit(X_train, y_train_cat, 
                            validation_data=(X_test, y_test_cat),
                            epochs=epochs, batch_size=batch_size, verbose=1)
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
        predictions = model.predict(X_test)
        
        if num_classes > 2:
            y_pred = np.argmax(predictions, axis=1)
        else:
            y_pred = (predictions > 0.5).astype(int).flatten()
        
        # Calculate comprehensive metrics
        metrics = self.calculate_metrics(y_test, y_pred, average=average_type, regression=regression)
        
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test Precision: {metrics['precision']:.4f}")
        print(f"Test Recall: {metrics['recall']:.4f}")
        print(f"Test F1-Score: {metrics['f1_score']:.4f}")
        print(f"Test AUC: {metrics['auc']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            # 'model': model, 
            # 'history': history, 
            'metrics': metrics,
            'predictions': y_pred
        }


# Utility function to split dataset
def split_dataset(texts, labels, test_size=0.2, random_state=42):
    """Split texts and labels into train and test sets"""
    return train_test_split(texts, labels, test_size=test_size, random_state=random_state, stratify=labels)

# Enhanced experiment runner with comprehensive metrics storage
def run_dl_experiments(train_texts, test_texts, train_labels, test_labels, model_types, embedding_types, regression=True):
    """Run all 24 combinations (4 models × 6 embeddings) with comprehensive metrics"""

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
                hybrid_model = HybridDeepLearningModels()
                
                result = hybrid_model.train_and_evaluate(
                    train_texts, train_labels, test_texts, test_labels,
                    model_type, embedding_type, epochs=10, batch_size=32, regression=regression
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