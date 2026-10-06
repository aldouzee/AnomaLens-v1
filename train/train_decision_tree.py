import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from data_preprocessing import load_and_preprocess_data
from utils import evaluate

def train_decision_tree(X_train, y_train, X_test, y_test):
    param_grid = {
        "DT__max_depth": [3, 5, 10, 15, None],
        "DT__min_samples_split": [2, 5, 10],
        "DT__criterion": ["gini", "entropy"]
    }

    estimator = ImbPipeline([
        ("SMOTE", SMOTE(random_state=42)),
        ("DT", DecisionTreeClassifier(random_state=42)),
    ])

    print("Running GridSearchCV for Decision Tree...")
    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")

    # Evaluate model
    evaluate("Decision Tree", best_model, X_test, y_test)

    # Save model for web app deployment
    joblib.dump(best_model, './models/decision_tree_model.pkl')
    print("Decision Tree model saved as 'decision_tree_model.pkl'")
    
    return best_model

if __name__ == "__main__":
    DATA_PATH = './datasets/processed_dataset.csv'
    X_train, X_test, y_train, y_test, _, _ = load_and_preprocess_data(DATA_PATH)
    train_decision_tree(X_train, y_train, X_test, y_test)