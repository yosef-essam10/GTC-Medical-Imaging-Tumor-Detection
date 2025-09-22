import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import Callback


def create_model(input_shape):
    model = models.Sequential()

    model.add(layers.Conv2D(64, (5,5), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(3,3)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (5,5), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(3,3)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, (4,4), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (4,4), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.Dropout(0.6))
    model.add(layers.Dense(4, activation="softmax"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.85, beta_2=0.9925)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return model


class DualMetricCheckpoint(Callback):
    def __init__(self, filepath):
        super(DualMetricCheckpoint, self).__init__()
        self.filepath = filepath
        self.best_acc = -float('inf')
        self.best_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get("val_accuracy")
        val_loss = logs.get("val_loss")

        if (val_acc > self.best_acc) or \
           (np.isclose(val_acc, self.best_acc, atol=1e-4) and val_loss < self.best_loss):

            print(f"\nEpoch {epoch+1}: val_accuracy={val_acc:.4f}, val_loss={val_loss:.4f} "
                  f"--> saving best model")

            self.best_acc = val_acc
            self.best_loss = val_loss
            self.model.save(self.filepath)
