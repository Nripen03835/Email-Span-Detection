import joblib
import pandas as pd
from .model import SpamClassifier
from .data_preprocessing import clean_text
from .config import PROCESSED_DATA

class SpamPredictor:
    def __init__(self):
        """Initialize the predictor with model and vectorizer"""
        # Load model
        self.classifier = SpamClassifier.load()
        
        # Load vectorizer from processed data
        processed_data = joblib.load(PROCESSED_DATA)
        self.vectorizer = processed_data['vectorizer']
    
    def predict(self, text):
        """Predict if text is spam (1) or ham (0)"""
        # Clean text
        cleaned_text = clean_text(text)
        
        # Vectorize
        X = self.vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = self.classifier.predict(X)
        probability = self.classifier.predict_proba(X)
        
        return {
            'prediction': prediction[0],
            'probability': probability[0][1],  # Probability of being spam
            'is_spam': bool(prediction[0])
        }
    
    def predict_batch(self, texts):
        """Predict for multiple texts"""
        cleaned_texts = [clean_text(text) for text in texts]
        X = self.vectorizer.transform(cleaned_texts)
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)[:, 1]
        
        results = []
        for text, pred, prob in zip(texts, predictions, probabilities):
            results.append({
                'text': text,
                'prediction': pred,
                'probability': prob,
                'is_spam': bool(pred)
            })
        
        return results

if __name__ == "__main__":
    # Example usage
    predictor = SpamPredictor()
    
    # Test with some example messages
    test_messages = [
        "Free entry in 2 a wkly comp to win FA Cup final tkts",
        "Hey, how are you doing today?",
        "Congratulations! You've won a $1000 prize! Call now to claim.",
        "Let's meet for lunch tomorrow"
    ]
    
    print("Spam Detection Results:")
    for msg in test_messages:
        result = predictor.predict(msg)
        print(f"\nMessage: {msg}")
        print(f"Prediction: {'SPAM' if result['is_spam'] else 'HAM'}")
        print(f"Probability: {result['probability']:.4f}")