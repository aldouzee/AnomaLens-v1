import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from data_preprocessing import load_and_preprocess_data
from utils import evaluate

def train_random_forest(X_train, y_train, X_test, y_test):
    param_grid = {
        "RandF__n_estimators": [100, 200, 300, 400],
        "RandF__max_depth": [5, 10, 15, None],
        "RandF__min_samples_split": [2, 5, 10, 15],
    }

    estimator = ImbPipeline([
        ("SMOTE", SMOTE(random_state=42)),
        ("RandF", RandomForestClassifier(random_state=42)),
    ])

    print("Running GridSearchCV for Random Forest...")
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
    evaluate("Random Forest", best_model, X_test, y_test)

    # Save model for web app deployment
    joblib.dump(best_model, './models/random_forest_model.pkl')
    print("Random Forest model saved as 'random_forest_model.pkl'")
    
    return best_model

if __name__ == "__main__":
    DATA_PATH = './datasets/processed_dataset.csv'
    X_train, X_test, y_train, y_test, _, _ = load_and_preprocess_data(DATA_PATH)
    train_random_forest(X_train, y_train, X_test, y_test)