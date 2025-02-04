import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def train_and_evaluate_model(df):
    """
    Train and evaluate a machine learning model for predicting player positions.

    Args:
        df (pd.DataFrame): Preprocessed dataset
    """
    print("Starting model training and evaluation...")

    # Define target and features
    target = 'position'  # Replace with actual target column name
    features = [col for col in df.columns if col != target]

    X = df[features]
    y = df[target]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    print("Model training and evaluation completed.")
