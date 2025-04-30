#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def create_visualizations(data, output_dir='visualizations', max_features=10):
    """
    Create various visualizations for the preprocessed data
    
    Args:
        data: DataFrame containing the preprocessed data
        output_dir: Directory to save the visualizations
        max_features: Maximum number of features to visualize
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set the style for all plots
    sns.set_style("whitegrid")
    
    # 1. Distribution of the target variable (clicked)
    plt.figure(figsize=(8, 6))
    sns.countplot(x='clicked', data=data)
    plt.title('Distribution of Clicks')
    plt.savefig(f'{output_dir}/click_distribution.png')
    plt.close()
    
    # Separate numerical and categorical features
    numerical_features = data.select_dtypes(include=[np.number]).columns
    categorical_features = data.select_dtypes(include=['object', 'category']).columns
    
    # Select top numerical features based on correlation with target
    if len(numerical_features) > max_features:
        correlation_with_target = data[numerical_features].corrwith(data['clicked']).abs()
        top_numerical = correlation_with_target.nlargest(max_features).index
    else:
        top_numerical = numerical_features
    
    # Select top categorical features based on number of unique values
    if len(categorical_features) > max_features:
        unique_counts = {col: data[col].nunique() for col in categorical_features}
        top_categorical = sorted(unique_counts.items(), key=lambda x: x[1])[:max_features]
        top_categorical = [col for col, _ in top_categorical]
    else:
        top_categorical = categorical_features
    
    # 2. Correlation heatmap (only for numerical features)
    if len(top_numerical) > 1:
        plt.figure(figsize=(12, 10))
        correlation_matrix = data[top_numerical].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation Heatmap (Top Numerical Features)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_heatmap.png')
        plt.close()
    
    # 3. Distribution plots for numerical features
    print("Creating distribution plots for numerical features...")
    for feature in tqdm(top_numerical):
        if feature != 'clicked':
            plt.figure(figsize=(10, 6))
            try:
                sns.histplot(data=data, x=feature, hue='clicked', kde=False)
                plt.title(f'Distribution of {feature} by Click Status')
                plt.savefig(f'{output_dir}/{feature}_distribution.png')
            except Exception as e:
                print(f"Error creating distribution plot for {feature}: {str(e)}")
            finally:
                plt.close()
    
    # 4. Count plots for categorical features
    print("Creating count plots for categorical features...")
    for feature in tqdm(top_categorical):
        if feature != 'clicked':
            plt.figure(figsize=(12, 6))
            try:
                sns.countplot(data=data, x=feature, hue='clicked')
                plt.xticks(rotation=45, ha='right')
                plt.title(f'Distribution of {feature} by Click Status')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/{feature}_count.png')
            except Exception as e:
                print(f"Error creating count plot for {feature}: {str(e)}")
            finally:
                plt.close()
    
    # 5. Box plots for numerical features
    print("Creating box plots...")
    for feature in tqdm(top_numerical):
        if feature != 'clicked':
            plt.figure(figsize=(10, 6))
            try:
                sns.boxplot(x='clicked', y=feature, data=data)
                plt.title(f'Box Plot of {feature} by Click Status')
                plt.savefig(f'{output_dir}/{feature}_boxplot.png')
            except Exception as e:
                print(f"Error creating box plot for {feature}: {str(e)}")
            finally:
                plt.close()
    
    # 6. Pair plot for top 5 numerical features
    try:
        top_5_numerical = list(top_numerical)[:5]
        if len(top_5_numerical) > 1:  # Only create pair plot if we have at least 2 features
            print("Creating pair plot...")
            plt.figure(figsize=(15, 15))
            sns.pairplot(data=data, vars=top_5_numerical, hue='clicked', 
                        diag_kind='hist', plot_kws={'alpha': 0.5})
            plt.savefig(f'{output_dir}/pairplot.png')
    except Exception as e:
        print(f"Error creating pair plot: {str(e)}")
    finally:
        plt.close()

def main():
    # Load the preprocessed data
    print("Loading data...")
    train_data = pd.read_csv('balanced-preprocessed-train.csv')
    test_data = pd.read_csv('balanced-preprocessed-test.csv')
    
    # Sample 5000 rows from each dataset
    sample_size = 5000
    train_sample = train_data.sample(n=min(sample_size, len(train_data)), random_state=42)
    test_sample = test_data.sample(n=min(sample_size, len(test_data)), random_state=42)
    
    print(f"Sampled {len(train_sample)} rows from training data")
    print(f"Sampled {len(test_sample)} rows from test data")
    
    # Create visualizations for training data
    print("Creating visualizations for training data...")
    create_visualizations(train_sample, 'visualizations/train', max_features=10)
    
    # Create visualizations for test data
    print("Creating visualizations for test data...")
    create_visualizations(test_sample, 'visualizations/test', max_features=10)
    
    print("Visualizations created successfully!")

if __name__ == "__main__":
    main()