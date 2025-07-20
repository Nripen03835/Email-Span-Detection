from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

from .config import MODEL_FILE

class SpamClassifier:
    def __init__(self, model_type='nb'):
        """Initialize the classifier
        
        Args:
            model_type (str): Type of model to use ('nb' for Naive Bayes or 'svm' for SVM)
        """
        if model_type == 'nb':
            self.model = MultinomialNB()
        elif model_type == 'svm':
            self.model = SVC(kernel='linear', probability=True)
        else:
            raise ValueError("Invalid model_type. Use 'nb' or 'svm'")
    
    def train(self, X_train, y_train):
        """Train the model"""
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        return metrics
    
    def save(self):
        """Save the trained model"""
        joblib.dump(self.model, MODEL_FILE)
    
    @classmethod
    def load(cls):
        """Load a trained model"""
        model = joblib.load(MODEL_FILE)
        classifier = cls()
        classifier.model = model
        return classifier