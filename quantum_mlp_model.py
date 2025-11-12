import tensorflow as tf
from tensorflow.keras import layers, models

def build_quantum_inspired_mlp(input_dim=784, num_classes=10):
    """
    Simple MLP classifier trained on amplitude-encoded features.
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model
