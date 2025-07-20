import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

from .config import RAW_DATA, PROCESSED_DATA

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def clean_text(text):
    """Clean and preprocess text data"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords and stem
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    # Load data
    df = pd.read_csv(RAW_DATA, encoding='latin-1')
    df = df[['Category', 'Message']]  # Keep only relevant columns
    df.columns = ['label', 'text']
    
    # Convert labels to binary (0 for ham, 1 for spam)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Clean text
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['label'].values
    
    # Save processed data and vectorizer
    processed_data = {
        'X': X,
        'y': y,
        'vectorizer': vectorizer
    }
    joblib.dump(processed_data, PROCESSED_DATA)
    
    return X, y, vectorizer

def get_train_test_data(test_size=0.2, random_state=42):
    """Get train and test data splits"""
    try:
        # Try to load processed data
        processed_data = joblib.load(PROCESSED_DATA)
        X = processed_data['X']
        y = processed_data['y']
        vectorizer = processed_data['vectorizer']
    except:
        # Process data if not found
        X, y, vectorizer = load_and_preprocess_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, vectorizer