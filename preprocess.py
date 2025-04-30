#!/usr/bin/env python3

from pipeline_elements import *
import sklearn.impute
import sklearn.preprocessing
import sklearn.pipeline
import pandas as pd
import numpy as np
import joblib
import os

def make_numerical_feature_pipeline():
    items = []
    items.append(("numerical-features-only", DataFrameSelector(do_predictors=True, do_numerical=True)))
    items.append(("missing-data", sklearn.impute.SimpleImputer(strategy="mean")))
    items.append(("scaler", sklearn.preprocessing.StandardScaler()))
    numerical_pipeline = sklearn.pipeline.Pipeline(items)
    return numerical_pipeline


def make_categorical_feature_pipeline():
    items = []
    items.append(("categorical-features-only", DataFrameSelector(do_predictors=True, do_numerical=False)))
    items.append(("missing-data", sklearn.impute.SimpleImputer(strategy="constant", fill_value="NULL")))
    items.append(("encode-category-bits", sklearn.preprocessing.OneHotEncoder(categories='auto', handle_unknown='ignore')))
    categorical_pipeline = sklearn.pipeline.Pipeline(items)
    return categorical_pipeline

def make_feature_pipeline():
    items = []
    items.append(("numerical", make_numerical_feature_pipeline()))
    items.append(("categorical", make_categorical_feature_pipeline()))
    pipeline = sklearn.pipeline.FeatureUnion(transformer_list=items)
    return pipeline

def preprocess_dataframe(pipeline, dataframe, label):
    """
    Preprocess a dataframe with the given pipeline.
    Assumes the pipeline has been fit.
    Assumes dataframe has an index column, and preserves it.
    If dataframe has a series identified by label, it is preserved.
    Assumes all other columns are features, and transforms them.
    """
    
    # list of features to transform
    feature_names = list(dataframe.columns)
    have_label = label in feature_names
    if have_label:
        feature_names.remove(label)
        
    # separate features and label
    X = dataframe[feature_names]
    if have_label:
        y = dataframe[label]
    
    # transform features
    X_transformed = pipeline.transform(X)
    
    # if the transform became sparse, densify it.
    # this usually happens because of one-hot-encoding
    if not isinstance(X_transformed, np.ndarray):
        X_transformed = X_transformed.todense()

    # reconstruct a dataframe, we've lost the labels of features. Too bad.
    df1 = pd.DataFrame(X_transformed)
    
    if have_label:
        # add labels
        df1[label] = y.to_numpy()

    # replace indexes
    df1.index = dataframe.index

    return df1

def fit_pipeline_to_dataframe(dataframe):
    pipeline = make_feature_pipeline()
    pipeline.fit(dataframe)
    return pipeline

def save_pipeline(pipeline, filename):
    joblib.dump(pipeline, filename)
    return

def load_pipeline(filename):
    pipeline = joblib.load(filename)
    return pipeline

def preprocess_file(input_filename, output_filename, pipeline_filename, label):
    dataframe = pd.read_csv(input_filename, index_col=0)
    # Replace '\N' with NaN
    dataframe = dataframe.replace('\\N', np.nan)
    
    # Drop rows with any missing values
    dataframe = dataframe.dropna()
    
    if not os.path.exists(pipeline_filename):
        pipeline = fit_pipeline_to_dataframe(dataframe)
        save_pipeline(pipeline, pipeline_filename)
    else:
        pipeline = load_pipeline(pipeline_filename)

    processed_dataframe = preprocess_dataframe(pipeline, dataframe, label)
    processed_dataframe.to_csv(output_filename, index=True)
    
    return

def main_train():
    data_filename = "balanced_train.csv"
    out_filename = "balanced-preprocessed-train.csv"
    pipeline_filename = "balanced-preprocessor.joblib"
    label = "clicked"
    preprocess_file(data_filename, out_filename, pipeline_filename, label)
    return

def main_test():
    data_filename = "balanced_test.csv"
    out_filename = "balanced-preprocessed-test.csv"
    pipeline_filename = "balanced-preprocessor.joblib"
    label = "clicked"
    preprocess_file(data_filename, out_filename, pipeline_filename, label)
    return

def main():
    main_train()
    main_test()

if __name__ == "__main__":
    main()
