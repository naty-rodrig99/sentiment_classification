import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

#Clean data
def preprocess(text):
    #Remove HTML tags
    processed_text = ''
    inside_tag = False
    for char in text:
        if char == '<':
            inside_tag = True
        elif char == '>':
            inside_tag = False
        elif not inside_tag:
            processed_text += char
    
    #Remove @, #, http, and www
    processed_text = processed_text.replace('@', '').replace('#', '')
    if 'http' in processed_text:
        processed_text = processed_text.split('http')[0]
    if 'www.' in processed_text:
        processed_text = processed_text.split('www.')[0]

    #Convert to lowercase
    processed_text = processed_text.lower()

    #Remove punctuation and non-alphabetic characters (emojis) - keep only letters and spaces
    processed_text = ''.join([char if char.isalpha() or char == ' ' else '' for char in processed_text])

    #Split into individual words
    tokens = processed_text.split()

    #Join into a cleaned string
    return ' '.join(tokens)

#Load datasets
train_data = pd.read_csv('train.csv')
evaluation_data = pd.read_csv('evaluation.csv')
test_data = pd.read_csv('test.csv')

#preprocess text data
train_data['clean_text'] = train_data['text'].apply(preprocess)
#evaluation_data['clean_text'] = evaluation_data['text'].apply(preprocess)
test_data['clean_text'] = test_data['text'].apply(preprocess)

#Define sample size
sample_size = 5000
train_data_sample = train_data.head(sample_size)
#evaluation_data_sample = evaluation_data.head(sample_size)
test_data_sample = test_data.head(sample_size)

#Feature extraction
def convert_text_to_vectors(dataSet, vocab=None):
    if vocab is None:
        vocab = set()
        for text in dataSet:
            vocab.update(text.split())
        vocab = sorted(vocab)
    
    feature_vectors = []
    for text in dataSet:
        tokens = text.split()
        token_counts = Counter(tokens) #count the frequency of each word
        vector = [token_counts.get(word, 0) for word in vocab]
        feature_vectors.append(vector)
    
    return np.array(feature_vectors), vocab

#Convert text to vectors
X_train, vocab = convert_text_to_vectors(train_data_sample['clean_text'])
#X_evaluation, _ = convert_text_to_vectors(evaluation_data_sample['clean_text'], vocab=vocab)
X_test, _ = convert_text_to_vectors(test_data_sample['clean_text'], vocab=vocab)

#If score is 0, then value is changed to -1, otherwise to 1
y_train = np.where(train_data_sample['score'].values == 0, -1, 1)
#y_evaluation = np.where(evaluation_data_sample['score'].values == 0, -1, 1)
y_test = np.where(test_data_sample['score'].values == 0, -1, 1)

#SVM Model training
def fit(X, y, learning_rate=0.001, lambda_param=0.01, n_iters=100):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    y_ = np.where(y <= 0, -1, 1)  #ensure labels are -1 and 1
    
    for _ in range(n_iters):
        for idx, x_i in enumerate(X):
            #applying formulas
            if y_[idx] * (np.dot(x_i, w) - b) >= 1:
                w -= learning_rate * (2 * lambda_param * w)
            else:
                w -= learning_rate * (2 * lambda_param * w - np.dot(x_i, y_[idx]))
                b -= learning_rate * y_[idx]
    
    return w, b

def predict(X, w, b):
    approx = np.dot(X, w) - b
    return np.sign(approx)

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    cax = ax.matshow(cm, cmap='Blues')  # Display the matrix with a blue color map
    plt.title(title)
    fig.colorbar(cax)

    # Labeling axes
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'])
    
    # Adding text annotations (the confusion matrix values)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

#train model
w, b = fit(X_train, y_train)

#Make predictions on the test dataset
#evaluation_predictions = predict(X_evaluation, w, b)
predictions = predict(X_test, w, b)


plot_confusion_matrix(y_test, predictions, title="SVM Confusion Matrix")

#evaluation_accuracy = np.mean(evaluation_predictions == y_evaluation)
#print("SVM classification evaluation accuracy:", evaluation_accuracy*100,"%")
#accuracy
test_accuracy = np.mean(predictions == y_test)
report = classification_report(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
print("SVM classification accuracy:", test_accuracy*100,"%")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)