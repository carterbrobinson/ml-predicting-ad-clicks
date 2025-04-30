#!/usr/bin/env python3

import pandas as pd
import tensorflow as tf
import keras
from keras.metrics import Precision, Recall
import matplotlib.pyplot as plt

label = "clicked"
input_filename = "balanced-preprocessed-train.csv"
model_filename = "clicked-model.keras"
train_ratio = 0.80
learning_curve_filename = "clicked-learning-curve.png"

# Load the training dataframe, separate into X/y
dataframe = pd.read_csv(input_filename, index_col=0)
X = dataframe.drop(label, axis=1)
y = dataframe[label]

# Prepare a tensorflow dataset from the dataframe
dataset = tf.data.Dataset.from_tensor_slices((X, y))

# Find the shape of the inputs and outputs
for features, labels in dataset.take(1):
    input_shape = features.shape
    output_shape = labels.shape

# Split the dataset into train and validation sets
dataset_size = dataset.cardinality().numpy()
train_size = int(train_ratio * dataset_size)
validate_size = dataset_size - train_size
train_dataset = dataset.take(train_size)
validation_dataset = dataset.skip(train_size)

# Shuffle and batch the datasets
BATCH_SIZE = 128  # Increased batch size for more stable gradients
train_dataset = train_dataset.shuffle(buffer_size=train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.shuffle(buffer_size=validate_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Build the model
tf.random.set_seed(42)

# Define the architecture with strong regularization
layers = [
    keras.layers.Input(shape=input_shape),
    keras.layers.BatchNormalization(),
    
    # First hidden layer with regularization
    keras.layers.Dense(64, activation="relu",
                      kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),  # Increased dropout

    # Second hidden layer
    keras.layers.Dense(32, activation="relu",
                      kernel_regularizer=keras.regularizers.l1_l2(l1=0.01, l2=0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),  # Increased dropout
    
    # Output layer
    keras.layers.Dense(1, activation="sigmoid")
]

# Create the model
model = keras.Sequential(layers)

# Compile the model with adjusted learning rate
initial_learning_rate = 0.0001  # Reduced learning rate
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=initial_learning_rate),
    metrics=["accuracy", Precision(), Recall(), "AUC"]
)

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,  # Increased patience
        restore_best_weights=True,
        min_delta=0.001
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    ),
    keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        save_best_only=True,
        monitor="val_loss"
    )
]

# Train the model
epoch_count = 100
history = model.fit(
    x=train_dataset,
    epochs=epoch_count,
    validation_data=validation_dataset,
    callbacks=callbacks
)

# Display the learning curves
pd.DataFrame(history.history).plot(
    figsize=(8, 5),
    xlim=[0, len(history.epoch)-1],
    ylim=[0, 1],
    grid=True,
    xlabel="Epoch"
)
plt.savefig(learning_curve_filename)
plt.clf()

# Save the model
model.save(model_filename)