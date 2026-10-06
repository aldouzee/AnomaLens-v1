import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from data_preprocessing import load_and_preprocess_data
from utils import evaluate

def train_bagging(X_train, y_train, X_test, y_test):
    param_grid = {
        "Bagging__n_estimators": [10, 50, 100],
        "Bagging__max_samples": [0.5, 0.8, 1.0],
    }

    estimator = ImbPipeline([
        ("SMOTE", SMOTE(random_state=42)),
        ("Bagging", BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=42), 
            random_state=42
        )),
    ])

    print("Running GridSearchCV for Bagging Classifier...")
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
    evaluate("Bagging Classifier", best_model, X_test, y_test)

    # Save model for web app deployment
    joblib.dump(best_model, './models/bagging_model.pkl')
    print("Bagging model saved as 'bagging_model.pkl'")
    
    return best_model

if __name__ == "__main__":
    DATA_PATH = './datasets/processed_dataset.csv'
    X_train, X_test, y_train, y_test, _, _ = load_and_preprocess_data(DATA_PATH)
    train_bagging(X_train, y_train, X_test, y_test)