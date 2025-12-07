import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

logger = logging.getLogger(__name__)


class LSTMMod]= config or {}
        self.model = None
        self.history = None
        self.sequence_length = self.config.get('models', {}).get('lstm', {}).get('sequence_length', 60)
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray = None,
                        sequence_length: int = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        X_seq = []
        y_seq = [] if y is not None else None
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            if y is not None:
                y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        
        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        
        return X_seq, None
    ig = self.config.get('models', {}).get('lstm', {})
        
        logger.info(f"Building LSTM model with input shape: {input_shape}")
        
        model = models.Sequential()
        
        lstm_layers = lstm_config.get('layers', [
            {'units': 128, 'dropout': 0.2, 'return_sequences': True},
            {'units': 64, 'dropout': 0.2, 'return_sequences': False}
        ])
        
        for i, layer_config in enumerate(lstm_layers):
            if i == 0:
                model.add(layers.LSTM(
                    units=layer_config['units'],
                    return_sequences=layer_config.get('return_sequences', True),
                    input_shape=input_shape
                ))
            else:
                model.add(layers.LSTM(
                    units=layer_config['units'],
                    return_sequences=layer_config.get('return_sequences', False)
                ))
            
            if 'dropout' in layer_config:
                model.add(layers.Dropout(layer_config['dropout']))
        g in dense_layers:
            model.add(layers.Dense(
                units=layer_config['units'],
                activation=layer_config.get('activation', 'relu')
            ))
            
            if 'dropout' in layer_config:
                model.add(layers.Dropout(layer_config['dropout']))
        
        output_activation = lstm_config.get('output_activation', 'sigmoid')
        model.add(layers.Dense(1, activation=output_activation))
        
        learning_rate = lstm_config.get('learning_rate', 0.001)
        optimizer = Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        logger.info("Model architecture:")
        model.summary(print_fn=logger.info)
        fig: Dict = None) -> keras.callbacks.History:
        if lstm_config is None:
            lstm_config = self.config.get('models', {}).get('lstm', {})
        
        logger.info("Training LSTM model...")
        
        if self.model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.build_model(input_shape, lstm_config)
        
        callback_list = []
        
        early_stopping_patience = lstm_config.get('early_stopping_patience', 10)
        callback_list.append(callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ))
        
        callback_list.append(callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ))
        
        , y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        self.history = history
        logger.info("Training complete")
        
        return history
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")
        
        y_pred_proba = self.model.predict(X)
        y_pr
        return self.model.predict(X).flatten()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        logger.info("Evaluating LSTM model...")
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        loss, accuracy, model_auc = self.model.evaluate(X_test, y_test, verbose=0)
        
        results = {
            'model': 'LSTM',
            'loss': loss,
            'accuracy': accuracy,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'roc_auc': auc,
        e_path / f"{model_name}.h5"
        
        self.model.save(model_file)
        logger.info(f"Saved LSTM model to {model_file}")
    
    def load_model(self, load_path: Path, model_name: str = 'lstm_model') -> None:
        model_file = load_path / f"{model_name}.h5"
        
    

    X_seq, y_seq = lstm.create_sequences(X, y, sequence_length=60)
    
    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    
    lstm.train(X_train, y_train, X_test, y_test)
    
    results = lstm.evaluate(X_test, y_test)
    print("\nResults:")
    for key, value in results.items():
        if key not in ['confusion_matrix', 'classification_report']:
            print(f"{key}: {value}")
