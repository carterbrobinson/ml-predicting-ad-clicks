#!/usr/bin/env python3

import sys
import argparse
import logging
import os.path
import pandas as pd
import numpy as np
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.base
import sklearn.metrics
import sklearn.impute
import sklearn.svm
import sklearn.ensemble
import joblib
import pprint
import matplotlib.pyplot as plt
from pipeline_elements import *
from preprocess import make_feature_pipeline
import xgboost as xgb

def load_data(args, filename):
    """
    Load and prepare data from CSV file
    """
    if not os.path.exists(filename):
        raise Exception(f"Data file: {filename} does not exist.")
    
    df = pd.read_csv(filename, index_col=0)
    # The last column is the target, all others are features
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def make_fit_pipeline(args):
    """
    Create a pipeline based on the model type and arguments
    """
    # Model selection
    if args.model_type == "linear":
        model = sklearn.linear_model.LogisticRegression(random_state=args.random_seed)
    elif args.model_type == "SVM":
        model = sklearn.svm.SVC(random_state=args.random_seed, probability=True)
    elif args.model_type == "forest":
        model = sklearn.ensemble.RandomForestClassifier(random_state=args.random_seed)
    elif args.model_type == "boost":
        model = sklearn.ensemble.GradientBoostingClassifier(
            learning_rate=0.01,
            max_depth=5,
            min_samples_leaf=2,
            min_samples_split=2,
            n_estimators=300,
            random_state=args.random_seed
        )
    elif args.model_type == "xgb":
        model = xgb.XGBClassifier(
            learning_rate=0.01,
            max_depth=5,
            min_child_weight=2,      # equivalent to min_samples_leaf
            subsample=1.0,
            colsample_bytree=1.0,
            n_estimators=300,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=args.random_seed
        )

    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    return model

def evaluate_model(model, X, y, X_val=None, y_val=None):
    """
    Evaluate model performance and return metrics
    """
    metrics = {}
    
    # Training metrics
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics['train'] = {
        'accuracy': sklearn.metrics.accuracy_score(y, y_pred),
        'precision': sklearn.metrics.precision_score(y, y_pred),
        'recall': sklearn.metrics.recall_score(y, y_pred),
        'f1': sklearn.metrics.f1_score(y, y_pred),
        'auc': sklearn.metrics.roc_auc_score(y, y_pred_proba) if y_pred_proba is not None else None,
        'loss': sklearn.metrics.log_loss(y, y_pred_proba) if y_pred_proba is not None else None
    }
    
    # Validation metrics if validation data is provided
    if X_val is not None and y_val is not None:
        y_val_pred = model.predict(X_val)
        y_val_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics['val'] = {
            'accuracy': sklearn.metrics.accuracy_score(y_val, y_val_pred),
            'precision': sklearn.metrics.precision_score(y_val, y_val_pred),
            'recall': sklearn.metrics.recall_score(y_val, y_val_pred),
            'f1': sklearn.metrics.f1_score(y_val, y_val_pred),
            'auc': sklearn.metrics.roc_auc_score(y_val, y_val_pred_proba) if y_val_pred_proba is not None else None,
            'loss': sklearn.metrics.log_loss(y_val, y_val_pred_proba) if y_val_pred_proba is not None else None
        }
    
    return metrics

def print_metrics(metrics):
    """
    Print metrics in a formatted way
    """
    print("\nModel Performance Metrics:")
    print("-" * 50)
    print(f"{'Metric':<15} {'Training':<10} {'Validation':<10}")
    print("-" * 50)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'loss']:
        train_val = metrics['train'][metric]
        val_val = metrics['val'][metric] if 'val' in metrics else None
        
        train_str = f"{train_val:.4f}" if train_val is not None else "N/A"
        val_str = f"{val_val:.4f}" if val_val is not None else "N/A"
        
        print(f"{metric:<15} {train_str:<10} {val_str:<10}")
    print("-" * 50)

def do_fit(args):
    """
    Fit pipeline to training data and evaluate
    """
    # Load and split data
    X, y = load_data(args, args.train_file)
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=args.random_seed
    )
    
    # Fit model
    model = make_fit_pipeline(args)
    model.fit(X_train, y_train)
    
    # Evaluate
    metrics = evaluate_model(model, X_train, y_train, X_val, y_val)
    print_metrics(metrics)
    
    # Save model
    model_file = args.model_file or f"{os.path.splitext(args.train_file)[0]}_model.joblib"
    joblib.dump(model, model_file)
    return

def do_cross_validation(args):
    """
    Perform cross-validation on training data
    """
    X, y = load_data(args, args.train_file)
    pipeline = make_fit_pipeline(args)
    
    cv_results = sklearn.model_selection.cross_validate(
        pipeline, X, y, 
        cv=args.cv_count,
        n_jobs=-1,
        verbose=3,
        scoring=('accuracy', 'precision', 'recall', 'f1')
    )
    
    print("\nCross-validation results:")
    for metric in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']:
        print(f"{metric}: {cv_results[metric].mean():.3f} ± {cv_results[metric].std():.3f}")
    
    return

def do_predict(args):
    """
    Make predictions on test data and evaluate if labels exist
    """
    X_test, y_test = load_data(args, args.test_file)
    model_file = args.model_file or f"{os.path.splitext(args.train_file)[0]}_model.joblib"
    
    if not os.path.exists(model_file):
        raise Exception(f"Model file: {model_file} does not exist.")
    
    model = joblib.load(model_file)
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Save predictions
    output_file = f"{os.path.splitext(args.test_file)[0]}_predictions.csv"
    pd.DataFrame({
        'predicted_click': y_pred,
        'actual_click': y_test
    }, index=X_test.index).to_csv(output_file)
    
    # Evaluate if labels are available
    if y_test is not None:
        print("\nTest Set Evaluation:")
        print("-" * 50)
        print(f"Accuracy : {sklearn.metrics.accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {sklearn.metrics.precision_score(y_test, y_pred):.4f}")
        print(f"Recall   : {sklearn.metrics.recall_score(y_test, y_pred):.4f}")
        print(f"F1 Score : {sklearn.metrics.f1_score(y_test, y_pred):.4f}")
        print(f"AUC      : {sklearn.metrics.roc_auc_score(y_test, y_pred_proba):.4f}")
        print(f"Log Loss : {sklearn.metrics.log_loss(y_test, y_pred_proba):.4f}")
        print("-" * 50)


def plot_learning_curves(metrics_history, output_file='model_learning_curves.png'):
    """
    Plot learning curves for all metrics
    """
    plt.figure(figsize=(12, 8))
    plt.grid(True)
    
    # Plot each metric
    colors = {
        'AUC': 'blue',
        'accuracy': 'orange',
        'loss': 'green',
        'precision': 'red',
        'recall': 'purple',
        'val_AUC': 'brown',
        'val_accuracy': 'pink',
        'val_loss': 'gray',
        'val_precision': 'yellow',
        'val_recall': 'cyan'
    }
    
    for metric, values in metrics_history.items():
        plt.plot(values, label=metric, color=colors.get(metric, None))
    
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('Model Learning Curves')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def train_with_partial_fits(model, X_train, y_train, X_val, y_val, n_estimators, metrics_history):
    """
    Train model incrementally and collect metrics
    """
    # Create a new instance of the same model type
    base_model = model.__class__(**model.get_params())
    
    # For models that support warm start
    if hasattr(base_model, 'warm_start'):
        base_model.warm_start = True
        base_model.n_estimators = 1  # Start with 1 estimator
    
    step_size = max(1, n_estimators // 20)  # Collect ~20 points for the curve
    
    for n in range(step_size, n_estimators + 1, step_size):
        if hasattr(base_model, 'n_estimators'):
            base_model.n_estimators = n
            base_model.fit(X_train, y_train)
        else:
            # For models that don't support warm start, just use the original model
            base_model = model
        
        # Collect metrics
        metrics = evaluate_model(base_model, X_train, y_train, X_val, y_val)
        
        # Update metrics history
        for metric in ['auc', 'accuracy', 'loss', 'precision', 'recall']:
            metrics_history[metric.upper() if metric == 'auc' else metric].append(metrics['train'][metric])
            metrics_history[f'val_{metric.upper() if metric == "auc" else metric}'].append(metrics['val'][metric])
    
    return model  # Return the original fully trained model

def do_grid_search(args):
    """
    Perform random search for hyperparameter tuning and evaluate with learning curves
    """
    # Load and split data
    X, y = load_data(args, args.train_file)
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=args.random_seed
    )
    
    model = make_fit_pipeline(args)
    
    # Define parameter distributions for random search
    if args.model_type == "forest":
        param_dist = {
            'n_estimators': [300],  # Fixed for learning curve
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        n_estimators = 300
    elif args.model_type == "boost":
        param_dist = {
            'n_estimators': [300],  # Fixed for learning curve
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        n_estimators = 300
    elif args.model_type == "xgb":
        param_dist = {
            'n_estimators': [300],  # For learning curves
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 2, 4],  # equivalent to min_samples_leaf
            'gamma': [0, 0.1, 0.2],         # equivalent to min_split_loss
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        n_estimators = 300

    else:
        param_dist = {}
        n_estimators = 100
    
    # Initialize metrics history
    metrics_history = {
        'AUC': [], 'accuracy': [], 'loss': [], 'precision': [], 'recall': [],
        'val_AUC': [], 'val_accuracy': [], 'val_loss': [], 'val_precision': [], 'val_recall': []
    }
    
    search = sklearn.model_selection.RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=20,
        cv=args.cv_count,
        n_jobs=-1,
        verbose=3,
        scoring='f1',
        random_state=args.random_seed
    )
    
    search.fit(X_train, y_train)
    
    print("\nBest parameters:")
    pprint.pprint(search.best_params_)
    
    # Train best model with partial fits to get learning curves
    best_model = search.best_estimator_
    best_model = train_with_partial_fits(
        best_model, X_train, y_train, X_val, y_val, 
        n_estimators, metrics_history
    )
    
    # Evaluate final model
    metrics = evaluate_model(best_model, X_train, y_train, X_val, y_val)
    print_metrics(metrics)
    
    # Plot learning curves
    plot_learning_curves(metrics_history, f"{os.path.splitext(args.train_file)[0]}_learning_curves.png")
    
    # Save best model
    model_file = args.model_file or f"{os.path.splitext(args.train_file)[0]}_best_model.joblib"
    joblib.dump(best_model, model_file)
    return

def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog=argv[0],
        description='Click Prediction Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('action',
        choices=['fit', 'cross', 'predict', 'grid-search'],
        help="desired action"
    )
    parser.add_argument('--model-type', '-M',
        default="forest",
        choices=["linear", "SVM", "boost", "forest", "xgb"],
        help="Model type"
    )
    parser.add_argument('--train-file', '-t',
        default="balanced-preprocessed-train.csv",
        help="name of file with training data"
    )
    parser.add_argument('--test-file', '-T',
        default="balanced-preprocessed-test.csv",
        help="name of file with test data"
    )
    parser.add_argument('--model-file', '-m',
        default="",
        help="name of file for the model"
    )
    parser.add_argument('--random-seed', '-R',
        default=42,
        type=int,
        help="random number seed"
    )
    parser.add_argument('--label',
        default="clicked",
        help="column name for label"
    )
    parser.add_argument('--cv-count',
        default=5,
        type=int,
        help="number of folds for cross-validation"
    )
    
    return parser.parse_args(argv[1:])

def main(argv):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    
    if args.action == 'fit':
        do_fit(args)
    elif args.action == 'cross':
        do_cross_validation(args)
    elif args.action == 'predict':
        do_predict(args)
    elif args.action == 'grid-search':
        do_grid_search(args)
    else:
        raise Exception(f"Unknown action: {args.action}")
    
    return

if __name__ == "__main__":
    main(sys.argv) 