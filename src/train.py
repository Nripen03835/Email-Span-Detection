from .model import SpamClassifier
from .data_preprocessing import get_train_test_data
from .config import TEST_SIZE, RANDOM_STATE, MODEL_FILE

def train_model(model_type='nb'):
    """Train and save the spam classifier"""
    # Get data
    X_train, X_test, y_train, y_test, _ = get_train_test_data(
        test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Initialize and train model
    classifier = SpamClassifier(model_type=model_type)
    classifier.train(X_train, y_train)
    
    # Evaluate
    metrics = classifier.evaluate(X_test, y_test)
    print("Model Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Save model
    classifier.save()
    print(f"Model saved to {MODEL_FILE}")
    
    return classifier

if __name__ == "__main__":
    # Train with Naive Bayes by default
    train_model(model_type='nb')