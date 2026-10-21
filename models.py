import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import config

tf.random.set_seed(config.RANDOM_STATE)
np.random.seed(config.RANDOM_STATE)



class BaseHybridModel:
    name = "BaseHybrid"

    def __init__(self):
        self.train_time_ = None
        self.predict_time_ = None

    def fit(self, X_train, y_train):
        raise NotImplementedError

    def predict(self, X_test):
        raise NotImplementedError

    def _time_fit(self, fit_callable):
        start = time.time()
        fit_callable()
        self.train_time_ = time.time() - start

    def _time_predict(self, predict_callable):
        start = time.time()
        result = predict_callable()
        self.predict_time_ = time.time() - start
        return result



class H1_AutoEncoder_RF(BaseHybridModel):
    name = "H1 (Auto-Encoder + RF)"

    def __init__(self, encoding_dim_ratio=0.5):
        super().__init__()
        self.encoding_dim_ratio = encoding_dim_ratio
        self.autoencoder = None
        self.rf = RandomForestClassifier(
            n_estimators=200, random_state=config.RANDOM_STATE, n_jobs=-1
        )

    def _build_autoencoder(self, input_dim):
        encoding_dim = max(2, int(input_dim * self.encoding_dim_ratio))
        inputs = keras.Input(shape=(input_dim,))
        x = layers.Dense(max(encoding_dim * 2, 8), activation="relu")(inputs)
        bottleneck = layers.Dense(encoding_dim, activation="relu")(x)
        x = layers.Dense(max(encoding_dim * 2, 8), activation="relu")(bottleneck)
        outputs = layers.Dense(input_dim, activation="linear")(x)
        autoencoder = keras.Model(inputs, outputs)
        autoencoder.compile(optimizer="adam", loss="mse")
        return autoencoder

    def _reconstruction_error(self, X):
        recon = self.autoencoder.predict(X, verbose=0)
        return np.mean(np.square(X - recon), axis=1, keepdims=True)

    def fit(self, X_train, y_train):
        def _fit():
            self.autoencoder = self._build_autoencoder(X_train.shape[1])
            # L'Auto-Encoder est entraîné UNIQUEMENT sur le trafic normal (label==0)
            X_normal = X_train[y_train == 0]
            self.autoencoder.fit(
                X_normal, X_normal,
                epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                validation_split=config.VALIDATION_SPLIT, verbose=0,
            )
            err_train = self._reconstruction_error(X_train)
            X_train_aug = np.hstack([X_train, err_train])
            self.rf.fit(X_train_aug, y_train)

        self._time_fit(_fit)
        return self

    def predict(self, X_test):
        def _predict():
            err_test = self._reconstruction_error(X_test)
            X_test_aug = np.hstack([X_test, err_test])
            return self.rf.predict(X_test_aug)
        return self._time_predict(_predict)

    def predict_proba(self, X_test):
        """Probabilité de la classe 1 (anomalie), utilisée pour la courbe ROC."""
        err_test = self._reconstruction_error(X_test)
        X_test_aug = np.hstack([X_test, err_test])
        return self.rf.predict_proba(X_test_aug)[:, 1]



class H2_LSTM_SVM(BaseHybridModel):
    name = "H2 (LSTM + SVM)"

    def __init__(self, embedding_dim=32):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.lstm_model = None
        self.feature_extractor = None
        self.svm = SVC(kernel="rbf", probability=True, random_state=config.RANDOM_STATE)

    def _build_lstm(self, input_dim):
        # chaque feature du vecteur est traitée comme un pas de temps (timestep=1 dim)
        inputs = keras.Input(shape=(input_dim, 1))
        x = layers.LSTM(64, return_sequences=False)(inputs)
        embedding = layers.Dense(self.embedding_dim, activation="relu", name="embedding")(x)
        outputs = layers.Dense(1, activation="sigmoid")(embedding)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def fit(self, X_train, y_train):
        def _fit():
            X_train_seq = X_train[..., np.newaxis]  # (n, n_features, 1)
            self.lstm_model = self._build_lstm(X_train.shape[1])
            self.lstm_model.fit(
                X_train_seq, y_train,
                epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                validation_split=config.VALIDATION_SPLIT, verbose=0,
            )
            self.feature_extractor = keras.Model(
                inputs=self.lstm_model.input,
                outputs=self.lstm_model.get_layer("embedding").output,
            )
            train_features = self.feature_extractor.predict(X_train_seq, verbose=0)
            self.svm.fit(train_features, y_train)

        self._time_fit(_fit)
        return self

    def predict(self, X_test):
        def _predict():
            X_test_seq = X_test[..., np.newaxis]
            test_features = self.feature_extractor.predict(X_test_seq, verbose=0)
            return self.svm.predict(test_features)
        return self._time_predict(_predict)

    def predict_proba(self, X_test):
        X_test_seq = X_test[..., np.newaxis]
        test_features = self.feature_extractor.predict(X_test_seq, verbose=0)
        return self.svm.predict_proba(test_features)[:, 1]



class H3_MLP_XGBoost(BaseHybridModel):
    name = "H3 (MLP + XGBoost)"

    def __init__(self, embedding_dim=32):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp_model = None
        self.feature_extractor = None
        self.xgb = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=config.RANDOM_STATE, eval_metric="logloss",
            n_jobs=-1,
        )

    def _build_mlp(self, input_dim):
        inputs = keras.Input(shape=(input_dim,))
        x = layers.Dense(128, activation="relu")(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation="relu")(x)
        embedding = layers.Dense(self.embedding_dim, activation="relu", name="embedding")(x)
        outputs = layers.Dense(1, activation="sigmoid")(embedding)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def fit(self, X_train, y_train):
        def _fit():
            self.mlp_model = self._build_mlp(X_train.shape[1])
            self.mlp_model.fit(
                X_train, y_train,
                epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                validation_split=config.VALIDATION_SPLIT, verbose=0,
            )
            self.feature_extractor = keras.Model(
                inputs=self.mlp_model.input,
                outputs=self.mlp_model.get_layer("embedding").output,
            )
            train_features = self.feature_extractor.predict(X_train, verbose=0)
            # on combine les features profondes ET les features originales
            X_train_aug = np.hstack([X_train, train_features])
            self.xgb.fit(X_train_aug, y_train)

        self._time_fit(_fit)
        return self

    def predict(self, X_test):
        def _predict():
            test_features = self.feature_extractor.predict(X_test, verbose=0)
            X_test_aug = np.hstack([X_test, test_features])
            return self.xgb.predict(X_test_aug)
        return self._time_predict(_predict)

    def predict_proba(self, X_test):
        test_features = self.feature_extractor.predict(X_test, verbose=0)
        X_test_aug = np.hstack([X_test, test_features])
        return self.xgb.predict_proba(X_test_aug)[:, 1]



class H4_CNN1D_GB(BaseHybridModel):
    name = "H4 (CNN1D + Gradient Boosting)"

    def __init__(self, embedding_dim=32):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.cnn_model = None
        self.feature_extractor = None
        self.gb = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            random_state=config.RANDOM_STATE,
        )

    def _build_cnn(self, input_dim):
        inputs = keras.Input(shape=(input_dim, 1))
        x = layers.Conv1D(32, kernel_size=3, activation="relu", padding="same")(inputs)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(x)
        x = layers.GlobalAveragePooling1D()(x)
        embedding = layers.Dense(self.embedding_dim, activation="relu", name="embedding")(x)
        outputs = layers.Dense(1, activation="sigmoid")(embedding)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def fit(self, X_train, y_train):
        def _fit():
            X_train_seq = X_train[..., np.newaxis]
            self.cnn_model = self._build_cnn(X_train.shape[1])
            self.cnn_model.fit(
                X_train_seq, y_train,
                epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
                validation_split=config.VALIDATION_SPLIT, verbose=0,
            )
            self.feature_extractor = keras.Model(
                inputs=self.cnn_model.input,
                outputs=self.cnn_model.get_layer("embedding").output,
            )
            train_features = self.feature_extractor.predict(X_train_seq, verbose=0)
            self.gb.fit(train_features, y_train)

        self._time_fit(_fit)
        return self

    def predict(self, X_test):
        def _predict():
            X_test_seq = X_test[..., np.newaxis]
            test_features = self.feature_extractor.predict(X_test_seq, verbose=0)
            return self.gb.predict(test_features)
        return self._time_predict(_predict)

    def predict_proba(self, X_test):
        X_test_seq = X_test[..., np.newaxis]
        test_features = self.feature_extractor.predict(X_test_seq, verbose=0)
        return self.gb.predict_proba(test_features)[:, 1]


def get_all_models():
    return [
        H1_AutoEncoder_RF(),
        H2_LSTM_SVM(),
        H3_MLP_XGBoost(),
        H4_CNN1D_GB(),
    ]
