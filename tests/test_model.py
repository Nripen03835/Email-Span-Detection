import unittest
import numpy as np
from src.model import SpamClassifier
from src.data_preprocessing import get_train_test_data
from src.config import TEST_SIZE, RANDOM_STATE

class TestSpamClassifier(unittest.TestCase):
    def setUp(self):
        # Get data for testing
        self.X_train, self.X_test, self.y_train, self.y_test, _ = get_train_test_data(
            test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # Initialize classifier
        self.classifier = SpamClassifier(model_type='nb')
        self.classifier.train(self.X_train, self.y_train)
    
    def test_predict_shape(self):
        predictions = self.classifier.predict(self.X_test)
        self.assertEqual(predictions.shape, self.y_test.shape)
    
    def test_predict_proba_shape(self):
        probabilities = self.classifier.predict_proba(self.X_test)
        self.assertEqual(probabilities.shape[0], self.y_test.shape[0])
        self.assertEqual(probabilities.shape[1], 2)  # Two classes
    
    def test_evaluate_metrics(self):
        metrics = self.classifier.evaluate(self.X_test, self.y_test)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        
        # Check metrics are between 0 and 1
        for metric in metrics.values():
            self.assertTrue(0 <= metric <= 1)
    
    def test_save_load(self):
        # Test saving and loading
        self.classifier.save()
        loaded_classifier = SpamClassifier.load()
        
        # Compare predictions
        original_pred = self.classifier.predict(self.X_test)
        loaded_pred = loaded_classifier.predict(self.X_test)
        
        np.testing.assert_array_equal(original_pred, loaded_pred)

if __name__ == "__main__":
    unittest.main()