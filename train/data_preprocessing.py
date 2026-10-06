import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(csv_path):
    """
    Loads raw dataset, cleans missing values, splits data, 
    and applies SMOTE to balance the training set.
    """
    df = pd.read_csv(csv_path)
    print("Initial class distribution:")
    print(df['anomaly'].value_counts())

    # Drop missing values
    df = df.dropna()

    # Drop non-feature columns
    drop_cols = ['anomaly', 'timestamp', 'Routers', 'Planned route', 
                 'Network measure', 'Network target', 'Video target']
    
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df['anomaly']

    # Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print('\nBefore SMOTE:', Counter(y_train))

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print('After SMOTE:', Counter(y_res))

    return X_train, X_test, y_train, y_test, X_res, y_res

def create_and_save_processed_dataset(X_res, y_res, output_path='./dataset/processed_dataset.csv'):
    """
    Saves the SMOTE-balanced training set to a CSV file.
    """
    processed_df = pd.concat([X_res, y_res], axis=1)
    processed_df.to_csv(output_path, index=False)
    print(f"Processed SMOTE dataset saved successfully to {output_path}")

def plot_smote_distribution(y_before, y_after):
    """
    Plots the class distribution before and after applying SMOTE.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.countplot(x=y_before, ax=axes[0])
    axes[0].set_title('Before SMOTE')

    sns.countplot(x=y_after, ax=axes[1])
    axes[1].set_title('After SMOTE')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    DATA_PATH = './datasets/network_dataset_labeled.csv'
    X_train, X_test, y_train, y_test, X_res, y_res = load_and_preprocess_data(DATA_PATH)
    
    # Save the processed dataset
    create_and_save_processed_dataset(X_res, y_res)
    
    # Plot distribution
    plot_smote_distribution(y_train, y_res)