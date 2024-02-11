#!/usr/bin/env python
# coding: utf-8

# # Resit Assignment
Group Member: Oğuzhan Taşçı              Student No: 2200356842
Group Member: İbrahim Enes Genişyürek    Student No: 21892757
# ## Required Libraries

# In[2]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string
import re
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from nltk.util import ngrams
from sklearn.preprocessing import normalize


# ## Understanding the data

# In[3]:


# Read dataset
data = pd.read_csv("book_data.csv")
data.head()


# In[4]:


data.genre.unique()


# In[5]:


data.groupby('genre').describe().iloc[:,0]


# ### Preprocessing

# In[6]:


# Split the dataset into training and testing sets (80% training, 20% testing)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


# In[7]:


stop_words_list = list(ENGLISH_STOP_WORDS)


# In[8]:


def custom_preprocessor(text):
    # Keep only English letters and spaces
    return re.sub(r'[^a-zA-Z\s]', '', text)


# In[9]:


# Assuming 'train_data' and 'test_data' are DataFrames with columns 'Genre' and 'Summary'
train_summary = train_data['summary'].values
test_summary = test_data['summary'].values
train_genre = train_data['genre'].values
test_genre = test_data['genre'].values


# # Naive Bayes

# In[10]:


# Initialize and fit a CountVectorizer with a custom preprocessor for unigram features
vectorizer_unigram = CountVectorizer(preprocessor=custom_preprocessor)
X_train_unigram = vectorizer_unigram.fit_transform(train_summary)
X_test_unigram = vectorizer_unigram.transform(test_summary)


# In[11]:


# Initialize and fit a CountVectorizer with a custom preprocessor for bigram features
vectorizer_bigram = CountVectorizer(preprocessor=custom_preprocessor, analyzer='word', ngram_range=(2, 2))
X_train_bigram = vectorizer_bigram.fit_transform(train_summary)
X_test_bigram = vectorizer_bigram.transform(test_summary)


# In[12]:


# Initialize and fit a CountVectorizer with a custom preprocessor for TD-IDF features
tfidf_transformer = TfidfVectorizer(preprocessor=custom_preprocessor)
X_train_tfidf = tfidf_transformer.fit_transform(train_summary)
X_test_tfidf = tfidf_transformer.transform(test_summary)


# In[13]:


# Initialize and fit a CountVectorizer with a custom preprocessor for Unigram with Removed Stopwords features
vectorizer_unigram_stop = CountVectorizer(preprocessor=custom_preprocessor, stop_words=stop_words_list)
X_train_unigram_stop = vectorizer_unigram_stop.fit_transform(train_summary)
X_test_unigram_stop = vectorizer_unigram_stop.transform(test_summary)


# In[14]:


# Initialize and fit a CountVectorizer with a custom preprocessor for Bigram With Removed Stopwords features
vectorizer_bigram_stop = CountVectorizer(preprocessor=custom_preprocessor, analyzer='word', ngram_range=(2, 2),stop_words=stop_words_list)
X_train_bigram_stop = vectorizer_bigram_stop.fit_transform(train_summary)
X_test_bigram_stop = vectorizer_bigram_stop.transform(test_summary)


# In[15]:


# Initialize and fit a CountVectorizer with a custom preprocessor for TF-IDF With Removed Stopwords features
tfidf_transformer_stop = TfidfVectorizer(preprocessor=custom_preprocessor, stop_words=stop_words_list)
X_train_tfidf_stop = tfidf_transformer_stop.fit_transform(train_summary)
X_test_tfidf_stop = tfidf_transformer_stop.transform(test_summary)


# In[16]:


def naive_bayes_train(X_train, y_train):
    num_samples, num_features = X_train.shape
    num_classes = len(np.unique(y_train))
    
    # Initialize parameters
    class_counts = np.zeros(num_classes)
    word_counts = np.zeros((num_classes, num_features))
    
    # Calculate class and word counts
    for i in range(num_samples):
        class_counts[y_train[i]] += 1
        word_counts[y_train[i]] += X_train[i]
    
    # Laplace smoothing
    alpha = 1
    class_priors = (class_counts + alpha) / (num_samples + num_classes * alpha)
    word_probs = (word_counts + alpha) / (np.sum(word_counts, axis=1, keepdims=True) + num_features * alpha)
    
    # Take logarithm
    log_class_priors = np.log(class_priors)
    log_word_probs = np.log(word_probs)
    
    return log_class_priors, log_word_probs


# In[17]:


def naive_bayes_predict(X_test, log_class_priors, log_word_probs):
    num_samples, num_classes = X_test.shape[0], log_class_priors.shape[0]
    log_probs = np.zeros((num_samples, num_classes))
    
    # Calculate log probabilities
    for i in range(num_samples):
        log_probs[i] = log_class_priors + np.sum(X_test[i] * log_word_probs, axis=1)
    
    # Predict class with maximum log probability
    predictions = np.argmax(log_probs, axis=1)
    
    return predictions


# In[18]:


# Convert genre labels to integers
genre_mapping = {'crime': 0, 'thriller': 1, 'fantasy': 2, 'horror': 3, 'history': 4, 'science': 5}
train_genre_int = np.array([genre_mapping[genre] for genre in train_genre])


# ### Functions that compute Accuracy, Precision and Recall For Naive Bayes

# In[82]:


# Calculation of Accuracy function
def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    accuracy = correct_predictions / total_samples
    return accuracy


# In[20]:


# Calculation of Precision function
def calculate_precision(y_true, y_pred, positive_class):
    true_positives = np.sum((y_true == positive_class) & (y_pred == positive_class))
    predicted_positives = np.sum(y_pred == positive_class)
    
    if predicted_positives == 0:
        return 0  # To avoid division by zero
    
    precision = true_positives / predicted_positives
    return precision


# In[21]:


# Calculation of Recall function
def calculate_recall(y_true, y_pred, positive_class):
    true_positives = np.sum((y_true == positive_class) & (y_pred == positive_class))
    actual_positives = np.sum(y_true == positive_class)
    
    if actual_positives == 0:
        return 0  # To avoid division by zero
    
    recall = true_positives / actual_positives
    return recall


# ### Unigram

# In[22]:


# Train a Naive Bayes classifier using unigram features
class_priors_unigram, word_probs_unigram = naive_bayes_train(X_train_unigram.toarray(), train_genre_int)
predictions_unigram = naive_bayes_predict(X_test_unigram.toarray(), class_priors_unigram, word_probs_unigram)


# In[23]:


ground_truth_int = np.array([genre_mapping[genre] for genre in test_genre])
predicted_labels_int = predictions_unigram


# In[24]:


# Calculate accuracy, precision, and recall for unigram
accuracy_unigram = calculate_accuracy(ground_truth_int, predicted_labels_int)
precision_unigram = calculate_precision(ground_truth_int, predicted_labels_int, positive_class=1)
recall_unigram = calculate_recall(ground_truth_int, predicted_labels_int, positive_class=1)


# In[77]:


print(f"Accuracy: {accuracy_unigram:.2f}")
print(f"Precision: {precision_unigram:.2f}")
print(f"Recall: {recall_unigram:.2f}")


# ### Bigram

# In[25]:


# Train a Naive Bayes classifier using bigram features
class_priors_bigram, word_probs_bigram = naive_bayes_train(X_train_bigram.toarray(), train_genre_int)
predictions_bigram = naive_bayes_predict(X_test_bigram.toarray(), class_priors_bigram, word_probs_bigram)


# In[26]:


predicted_labels_int = predictions_bigram


# In[27]:


# Calculate accuracy, precision, and recall for bigram
accuracy_bigram = calculate_accuracy(ground_truth_int, predicted_labels_int)
precision_bigram = calculate_precision(ground_truth_int, predicted_labels_int, positive_class=1)
recall_bigram = calculate_recall(ground_truth_int, predicted_labels_int, positive_class=1)


# In[28]:


print(f"Accuracy: {accuracy_bigram:.2f}")
print(f"Precision: {precision_bigram:.2f}")
print(f"Recall: {recall_bigram:.2f}")


# ### TD-IDF

# In[29]:


# Train a Naive Bayes classifier using TD-IDF features
class_priors_tfidf, word_probs_tfidf = naive_bayes_train(X_train_tfidf.toarray(), train_genre_int)
predictions_tfidf = naive_bayes_predict(X_test_tfidf.toarray(), class_priors_tfidf, word_probs_tfidf)


# In[30]:


predicted_labels_int = predictions_tfidf


# In[31]:


# Calculate accuracy, precision, and recall for TD-IDF
accuracy_tfidf = calculate_accuracy(ground_truth_int, predicted_labels_int)
precision_tfidf = calculate_precision(ground_truth_int, predicted_labels_int, positive_class=1)
recall_tfidf = calculate_recall(ground_truth_int, predicted_labels_int, positive_class=1)


# In[78]:


print(f"Accuracy: {accuracy_tfidf:.2f}")
print(f"Precision: {precision_tfidf:.2f}")
print(f"Recall: {recall_tfidf:.2f}")


# ### Unigram With Stopwords Removed

# In[32]:


# Train a Naive Bayes classifier using Unigram With Stopwords Removed features
class_priors_unigram_stop, word_probs_unigram_stop = naive_bayes_train(X_train_unigram_stop.toarray(), train_genre_int)
predictions_unigram_stop = naive_bayes_predict(X_test_unigram_stop.toarray(), class_priors_unigram_stop, word_probs_unigram_stop)


# In[33]:


predicted_labels_int = predictions_unigram_stop


# In[34]:


# Calculate accuracy, precision, and recall for Unigram With Stopwords Removed
accuracy_unigram_stop = calculate_accuracy(ground_truth_int, predicted_labels_int)
precision_unigram_stop = calculate_precision(ground_truth_int, predicted_labels_int, positive_class=1)
recall_unigram_stop = calculate_recall(ground_truth_int, predicted_labels_int, positive_class=1)


# In[79]:


print(f"Accuracy: {accuracy_unigram_stop:.2f}")
print(f"Precision: {precision_unigram_stop:.2f}")
print(f"Recall: {recall_unigram_stop:.2f}")


# ### Biagram With Stopwords Removed

# In[35]:


# Train a Naive Bayes classifier using Bigram With Stopwords Removed features
class_priors_bigram_stop, word_probs_bigram_stop = naive_bayes_train(X_train_bigram_stop.toarray(), train_genre_int)
predictions_bigram_stop = naive_bayes_predict(X_test_bigram_stop.toarray(), class_priors_bigram_stop, word_probs_bigram_stop)


# In[36]:


predicted_labels_int = predictions_bigram_stop


# In[37]:


# Calculate accuracy, precision, and recall for Biagram With Stopwords Removed
accuracy_bigram_stop = calculate_accuracy(ground_truth_int, predicted_labels_int)
precision_bigram_stop = calculate_precision(ground_truth_int, predicted_labels_int, positive_class=1)
recall_bigram_stop = calculate_recall(ground_truth_int, predicted_labels_int, positive_class=1)


# In[80]:


print(f"Accuracy: {accuracy_bigram_stop:.2f}")
print(f"Precision: {precision_bigram_stop:.2f}")
print(f"Recall: {recall_bigram_stop:.2f}")


# ### TD-IDF With Stopwords Removed

# In[38]:


# Train a Naive Bayes classifier using TD-IDF With Stopwords Removed features
class_priors_tfidf_stop, word_probs_tfidf_stop = naive_bayes_train(X_train_tfidf_stop.toarray(), train_genre_int)
predictions_tfidf_stop = naive_bayes_predict(X_test_tfidf_stop.toarray(), class_priors_tfidf_stop, word_probs_tfidf_stop)


# In[39]:


predicted_labels_int = predictions_tfidf_stop


# In[40]:


# Calculate accuracy, precision, and recall for TD-IDF With Stopwords Removed
accuracy_tfidf_stop = calculate_accuracy(ground_truth_int, predicted_labels_int)
precision_tfidf_stop = calculate_precision(ground_truth_int, predicted_labels_int, positive_class=1)
recall_tfidf_stop = calculate_recall(ground_truth_int, predicted_labels_int, positive_class=1)


# In[81]:


print(f"Accuracy: {accuracy_tfidf_stop:.2f}")
print(f"Precision: {precision_tfidf_stop:.2f}")
print(f"Recall: {recall_tfidf_stop:.2f}")


# # K-Means Clustering

# In[41]:


summary = data['summary'].values
genre = data['genre'].values


# In[42]:


# Unigram
X_unigram = vectorizer_unigram.fit_transform(summary)


# In[43]:


# Biagram
X_bigram = vectorizer_bigram.fit_transform(summary)


# In[44]:


# TF-IDF
X_tfidf = tfidf_transformer.fit_transform(summary)


# In[45]:


# Unigram with Removed Stopwords
X_unigram_stop = vectorizer_unigram_stop.fit_transform(summary)


# In[46]:


# Bigram With Removed Stopwords
X_bigram_stop = vectorizer_bigram_stop.fit_transform(summary)


# In[47]:


# TF-IDF With Removed Stopwords
X_tfidf_stop = tfidf_transformer_stop.fit_transform(summary)


# In[48]:


#  Initialize centroids for k-means clustering
def initialize_centroids(k, n_features):
    return np.random.rand(k, n_features)


# In[49]:


#  Assign data points to the nearest cluster based on Euclidean distance.
def assign_to_clusters(X, centroids):
    distances = np.linalg.norm(X - centroids[:, np.newaxis], axis=2)
    return np.argmin(distances, axis=0)


# In[50]:


# Update centroids based on the mean of data points in each cluster
def update_centroids(X, clusters, k):
    centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        cluster_points = X[clusters == i]
        if len(cluster_points) > 0:
            centroids[i] = np.mean(cluster_points, axis=0)
    return centroids


# In[51]:


# Perform k-means clustering on the input data.
def kmeans(X, k, max_iters=100):
    n_samples, n_features = X.shape
    centroids = initialize_centroids(k, n_features)

    for _ in range(max_iters):
        clusters = assign_to_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids


# ### Functions that compute Accuracy, Precision and Recall For K-Means Clustering

# In[52]:


# Calculation of Accuracy function
def calculate_accuracy(true_labels, predicted_labels):
    correct_predictions = np.sum(true_labels == predicted_labels)
    total_samples = len(true_labels)
    accuracy = correct_predictions / total_samples
    return accuracy


# In[53]:


# Calculation of Precision function
def calculate_precision(true_labels, predicted_labels, positive_class):
    true_positive = np.sum((true_labels == positive_class) & (predicted_labels == positive_class))
    false_positive = np.sum((true_labels != positive_class) & (predicted_labels == positive_class))
    
    if true_positive + false_positive == 0:
        precision = 0  # Avoid division by zero
    else:
        precision = true_positive / (true_positive + false_positive)
    
    return precision


# In[54]:


# Calculation of Recall function
def calculate_recall(true_labels, predicted_labels, positive_class):
    true_positive = np.sum((true_labels == positive_class) & (predicted_labels == positive_class))
    false_negative = np.sum((true_labels == positive_class) & (predicted_labels != positive_class))
    
    if true_positive + false_negative == 0:
        recall = 0  # Avoid division by zero
    else:
        recall = true_positive / (true_positive + false_negative)
    
    return recall


# In[55]:


# Convert genre labels to integers
genre_mapping = {'crime': 0, 'thriller': 1, 'fantasy': 2, 'horror': 3, 'history': 4, 'science': 5}
ground_truth_int = np.array([genre_mapping[genre] for genre in genre])


# In[56]:


# Assuming you want to cluster into 6 genres
k = 6


# ### Unigram

# In[57]:


# K-Means on Unigram
clusters_unigram, centroids_unigram = kmeans(X_unigram.toarray(), k)


# In[58]:


# Evaluate Unigram K-Means model
accuracy_unigram_K = calculate_accuracy(ground_truth_int, clusters_unigram)
precision_unigram_K = calculate_precision(ground_truth_int, clusters_unigram, positive_class=0)
recall_unigram_K = calculate_recall(ground_truth_int, clusters_unigram, positive_class=0)


# In[59]:


print("Accuracy (Unigram BoW):", accuracy_unigram_K)
print("Precision (Unigram BoW):", precision_unigram_K)
print("Recall (Unigram BoW):", recall_unigram_K)


# ### TF_IDF

# In[60]:


# K-Means on TF_IDF
clusters_tfidf, centroids_tfidf = kmeans(X_tfidf.toarray(), k)


# In[61]:


# Evaluate TF_IDF K-Means model
accuracy_tfidf_K = calculate_accuracy(ground_truth_int, clusters_tfidf)
precision_tfidf_K = calculate_precision(ground_truth_int, clusters_tfidf, positive_class=0)
recall_tfidf_K = calculate_recall(ground_truth_int, clusters_tfidf, positive_class=0)


# In[62]:


print("Accuracy (Unigram BoW):", accuracy_tfidf_K)
print("Precision (Unigram BoW):", precision_tfidf_K)
print("Recall (Unigram BoW):", recall_tfidf_K)


# ### Unigram With Stopwords Removed

# In[63]:


# Unigram With Stopwords Removed
clusters_unigram_stop, centroids_unigram_stop = kmeans(X_unigram_stop.toarray(), k)


# In[64]:


# Evaluate Unigram With Stopwords Removed
accuracy_unigram_stop_K = calculate_accuracy(ground_truth_int, clusters_unigram_stop)
precision_unigram_stop_K = calculate_precision(ground_truth_int, clusters_unigram_stop, positive_class=0)
recall_unigram_stop_K = calculate_recall(ground_truth_int, clusters_unigram_stop, positive_class=0)


# In[65]:


print("Accuracy (Unigram BoW):", accuracy_unigram_stop_K)
print("Precision (Unigram BoW):", precision_unigram_stop_K)
print("Recall (Unigram BoW):", recall_unigram_stop_K)


# ### TF-IDF With Removed Stopwords

# In[66]:


# K-Means on TF-IDF With Removed Stopwords
clusters_tfidf_stop, centroids__tfidf_stop = kmeans(X_tfidf_stop.toarray(), k)


# In[73]:


# Evaluate TF-IDF With Removed Stopwords
accuracy_tfidf_stop_K = calculate_accuracy(ground_truth_int, clusters_tfidf_stop)
precision_tfidf_stop_K = calculate_precision(ground_truth_int, clusters_tfidf_stop, positive_class=0)
recall_tfidf_stop_K = calculate_recall(ground_truth_int, clusters_tfidf_stop, positive_class=0)


# In[74]:


print("Accuracy (Unigram BoW):", accuracy_tfidf_stop_K)
print("Precision (Unigram BoW):", precision_tfidf_stop_K)
print("Recall (Unigram BoW):", recall_tfidf_stop_K)


# # Analyze results

# In[76]:


# Display results in a table format
print("Feature       | Stopwords | Algorithm | Accuracy | Precision | Recall")
print("---------------------------------------------------------------------------")
print(f"BoW (Unigram) |           |    NB     |   {accuracy_unigram:.2f}   |   {precision_unigram:.2f}    |  {recall_unigram:.2f}")
print("---------------------------------------------------------------------------")
print(f"BoW (Unigram) | Removed   |    NB     |   {accuracy_bigram:.2f}   |   {precision_bigram:.2f}    |  {recall_bigram:.2f}")
print("---------------------------------------------------------------------------")
print(f"BoW (Bigram)  |           |    NB     |   {accuracy_tfidf:.2f}   |   {precision_tfidf:.2f}    |  {recall_tfidf:.2f}")
print("---------------------------------------------------------------------------")
print(f"BoW (Bigram)  | Removed   |    NB     |   {accuracy_unigram_stop:.2f}   |   {precision_unigram_stop:.2f}    |  {recall_unigram_stop:.2f}")
print("---------------------------------------------------------------------------")
print(f"    TF-IDF    |           |    NB     |   {accuracy_bigram_stop:.2f}   |   {precision_bigram_stop:.2f}    |  {recall_bigram_stop:.2f}")
print("---------------------------------------------------------------------------")
print(f"    TF-IDF    | Removed   |    NB     |   {accuracy_tfidf_stop:.2f}   |   {precision_tfidf_stop:.2f}    |  {recall_tfidf_stop:.2f}")
print("---------------------------------------------------------------------------")
print(f"BoW (Unigram) |           |  K-Means  |   {accuracy_unigram_K:.2f}   |   {precision_unigram_K:.2f}    |  {recall_unigram_K:.2f}")
print("---------------------------------------------------------------------------")
print(f"BoW (Unigram) | Removed   |  K-Means  |   {accuracy_unigram_stop_K:.2f}   |   {precision_unigram_stop_K:.2f}    |  {recall_unigram_stop_K:.2f}")
print("---------------------------------------------------------------------------")
print(f"    TF-IDF    |           |  K-Means  |   {accuracy_tfidf_K:.2f}   |   {precision_tfidf_K:.2f}    |  {recall_tfidf_K:.2f}")
print("---------------------------------------------------------------------------")
print(f"    TF-IDF    | Removed   |  K-Means  |   {accuracy_tfidf_stop_K:.2f}   |   {precision_tfidf_stop_K:.2f}    |  {recall_tfidf_stop_K:.2f}")
print("---------------------------------------------------------------------------")


# ### Naive Bayes (NB):

# Generally, removing stopwords tends to decrease accuracy but might increase precision.
# - Bigram BoW performs better than Unigram BoW in terms of accuracy and recall.
# - TF-IDF, in this case, doesn't show superior performance compared to BoW.
# - For NB, removing stopwords generally had a negative impact on accuracy. It led to a notable decrease in accuracy for BoW (Unigram) and a slight decrease for BoW (Bigram).

# ### K-Means:

# - K-Means seems to perform poorly compared to Naive Bayes, especially with low accuracy and recall.
# - The removal of stopwords has a significant impact on K-Means performance, leading to lower accuracy.
# - Removing stopwords significantly decreased accuracy for K-Means, especially for BoW (Unigram).

# # Conclusion

# - Naive Bayes (NB) appears more suitable for this dataset, with better overall performance across different feature extraction methods.
# - Bigram BoW with Naive Bayes might be a good choice considering its higher accuracy and recall.
# - TF-IDF with Naive Bayes also shows promising results, especially with the stopwords included.
# - Removing stopwords does not consistently improve classification performance in this specific case.

# # References

# https://web.cs.hacettepe.edu.tr/~abc/teaching/bbm406/index.php
# 
# https://stackoverflow.com
# 
# https://theflyingmantis.medium.com/text-classification-in-nlp-naive-bayes-a606bf419f8c
