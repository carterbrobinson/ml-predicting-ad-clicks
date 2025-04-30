#!/usr/bin/env python3

import pandas as pd
import tensorflow as tf
import keras
label = "Depression"
input_filename = "depression-preprocessed-test.csv"
model_filename = "depression-model.keras"
proba_filename = "predictions_proba.csv"
predictions_filename = "predictions.csv"

#
# Load the test data, removing the label column, if it exists
#
dataframe = pd.read_csv(input_filename, index_col=0)
if label in dataframe.columns:
    X = dataframe.drop(label, axis=1)
else:
    X = dataframe

#print(X)

#
# Convert to tensorflow dataset
#
dataset = tf.data.Dataset.from_tensors(X)

#print(dataset)
#print(dataset.element_spec)


#
# Prep the dataset for efficient processing
#
BATCH_SIZE = 32
dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


#
# Load the trained model
#
model = keras.saving.load_model(model_filename)
print(model.summary())
#print(model.layers[1].get_weights())

#
# Predict the labels.
# The model's output should be the correct shape for our purposes, if constructed with correct assumptions.
#
y_hat = model.predict(X)
y_hat_binary = (y_hat > 0.5).astype(int)
# print(type(y_hat))
# print(y_hat.shape)

#
# Construct a dataframe with the ids and predicted values for Kaggle submission
# proba
#
merged = dataframe.index.to_frame()
merged[label] = y_hat[:,0]
merged.to_csv(proba_filename, index=False)

#
# Construct a dataframe with the ids and predicted values for Kaggle submission
# Binary 
#
merged = dataframe.index.to_frame()
merged[label] = y_hat_binary[:,0]
merged.to_csv(predictions_filename, index=False)

