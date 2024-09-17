import pandas as pd
from collections import defaultdict
import numpy as np

def remove_plurals(word):
    if word.endswith('s'):
        return word[:-1] 
    return word

def preprocess(text):
    # Remove HTML tags
    processed_text = ''
    inside_tag = False
    for char in text:
        if char == '<':
            inside_tag = True
        elif char == '>':
            inside_tag = False
        elif not inside_tag:
            processed_text += char
    
    # Remove @, #, http, and www
    processed_text = processed_text.replace('@', '').replace('#', '')
    if 'http' in processed_text:
        processed_text = processed_text.split('http')[0]
    if 'www.' in processed_text:
        processed_text = processed_text.split('www.')[0]

    # Convert to lowercase
    processed_text = processed_text.lower()

    # Remove punctuation and non-alphabetic characters (emojis) - keep only letters and spaces
    processed_text = ''.join([char if char.isalpha() or char == ' ' else '' for char in processed_text])

    # Split into individual words
    tokens = processed_text.split()

    # Lemmatization: Convert plurals to singular
    #tokens = [remove_plurals(token) for token in tokens]

    # Join into a cleaned string
    return ' '.join(tokens)

# Load the datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
evaluation_data = pd.read_csv('evaluation.csv')

# Function to train Naive Bayes classifier
def train_naive_bayes(train_data, score):
    # Separate texts into positive and negative classes
    positive_texts = train_data[score == 1]
    negative_texts = train_data[score == 0]

    # Create frequency counts for each class
    positive_counts = defaultdict(int)
    negative_counts = defaultdict(int)
    
    total_positive_words = 0
    total_negative_words = 0
    vocabulary = set()

    for text in positive_texts:
        for word in text.split():
            positive_counts[word] += 1
            total_positive_words += 1
            vocabulary.add(word)
    
    for text in negative_texts:
        for word in text.split():
            negative_counts[word] += 1
            total_negative_words += 1
            vocabulary.add(word)
    
    # Calculate class priors
    p_positive = len(positive_texts) / len(train_data)
    p_negative = len(negative_texts) / len(train_data)
    
    return {
        'positive_counts': positive_counts,
        'negative_counts': negative_counts,
        'total_positive_words': total_positive_words,
        'total_negative_words': total_negative_words,
        'p_positive': p_positive,
        'p_negative': p_negative,
        'vocabulary': vocabulary
    }

# Function to calculate the likelihood of a word given a class
def word_likelihood(word, counts, total_words, vocab_size, smoothing=1):
    return (counts[word] + smoothing) / (total_words + smoothing * vocab_size)

# Function to predict the class of a new text
def predict(text, model):
    vocab_size = len(model['vocabulary'])
    
    # Calculate log probabilities (for numerical stability)
    log_p_positive = np.log(model['p_positive'])
    log_p_negative = np.log(model['p_negative'])
    
    for word in text.split():
        log_p_positive += np.log(word_likelihood(word, model['positive_counts'], model['total_positive_words'], vocab_size))
        log_p_negative += np.log(word_likelihood(word, model['negative_counts'], model['total_negative_words'], vocab_size))
    
    # Return the class with the higher log probability
    if log_p_positive > log_p_negative:
        return 1
    else:
        return 0

# Preprocess data
train_data['processed_text'] = train_data['text'].apply(preprocess)
evaluation_data['processed_text'] = evaluation_data['text'].apply(preprocess)
test_data['processed_text'] = test_data['text'].apply(preprocess)

# Train Naive Bayes model
nb_model = train_naive_bayes(train_data['processed_text'], train_data['score'])

# Predict on evaluation data
evaluation_data['predicted_score'] = evaluation_data['processed_text'].apply(lambda x: predict(x, nb_model))

# Accuracy on evaluation data
evaluation_accuracy = (evaluation_data['predicted_score'] == evaluation_data['score']).mean()
print(f'Evaluation Accuracy: {evaluation_accuracy}')

# Predict on test data
test_data['predicted_score'] = test_data['processed_text'].apply(lambda x: predict(x, nb_model))

# Test accuracy
accuracy = (test_data['predicted_score'] == test_data['score']).mean()
print(f'Accuracy: {accuracy}')

