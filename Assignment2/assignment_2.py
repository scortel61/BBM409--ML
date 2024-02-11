#!/usr/bin/env python
# coding: utf-8

# # Assignment 2
# 
# ## Part 1
# 
# Group Member: Oğuzhan Taşçı
# 
# Group Member: İbrahim Enes Genişyürek

# In[1]:


pip install opencv-python


# In[2]:


# Import necessary libraries
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


# In[3]:


from skimage.filters import gabor

def load_images_and_labels(folder):
    images = []
    labels = []
    label_dict = {}  # Dictionary to map folder names to numerical labels
    label_counter = 0

    for class_folder in os.listdir(folder):
        class_path = os.path.join(folder, class_folder)
        if os.path.isdir(class_path):
            label_dict[class_folder] = label_counter
            label_counter += 1

            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Change flags based on your needs
                if img is not None:
                    # resize image
                    img = cv2.resize(img, (64, 64))  # Resize the image
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

                    # Apply Gabor filters
                    gabor_features = []
                    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
                        for frequency in [0.1, 0.5, 1.0]:
                            gabor_img, _ = gabor(gray_img, frequency, theta=theta)
                            gabor_features.extend(gabor_img.flatten())

                    # Extract histogram features using HSV color space
                    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    hist_features = []
                    for channel in range(3):  # For each color channel (H, S, V)
                        hist = cv2.calcHist([hsv_img], [channel], None, [256], [0, 256])
                        hist_features.extend(hist.flatten())

                    # Concatenate features
                    all_features = np.concatenate([gabor_features, hist_features])
                    
                    # Append the features to the images list
                    images.append(all_features)
                    
                    # Append the numerical label to the labels list
                    labels.append(label_dict[class_folder])
                    
    # Convert lists to NumPy arrays and return them
    return np.array(images), np.array(labels)


# In[4]:


# paths to folders
train_folder = '/Users/oguzhantasci/Downloads/flowers/train'
validation_folder = '/Users/oguzhantasci/Downloads/flowers/validation'
test_folder = '/Users/oguzhantasci/Downloads/flowers/test'


# In[5]:


# training, validation, and testing data
X_train, y_train = load_images_and_labels(train_folder)
X_validation, y_validation = load_images_and_labels(validation_folder)
X_test, y_test = load_images_and_labels(test_folder)


# In[6]:


class Node:
    def __init__(self, data, target, depth, max_depth):
        # Initialize a node with data, target labels, depth, and maximum depth
        self.data = data
        self.target = target
        self.depth = depth
        self.max_depth = max_depth
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.prediction = self.most_common_class()
# Calculate the most common class in the target labels of the node
    def most_common_class(self):
        unique_classes, counts = np.unique(self.target, return_counts=True)
        index = np.argmax(counts) # Find the index of the most frequent class
        return unique_classes[index] # Return the most common class


# In[7]:


# entropy function calculates the entropy of a set of labels
def entropy(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy_value = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy_value


# In[8]:


# gain function
def information_gain(data, target, feature, threshold):
    subset_left = target[data[:, feature] <= threshold]
    subset_right = target[data[:, feature] > threshold]

    total_entropy = entropy(target)
    weighted_entropy = (len(subset_left) / len(target)) * entropy(subset_left) + \
                       (len(subset_right) / len(target)) * entropy(subset_right)

    info_gain = total_entropy - weighted_entropy
    return info_gain


# In[9]:


def find_best_split(data, target):
    num_features = data.shape[1]
    best_info_gain = -1
    best_feature = None
    best_threshold = None

    for feature in range(num_features):
        unique_values = np.unique(data[:, feature])
        for value in unique_values:
            info_gain = information_gain(data, target, feature, value)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = feature
                best_threshold = value

    return best_feature, best_threshold


# In[10]:


def build_tree(data, target, depth, max_depth):
    if len(np.unique(target)) == 1 or depth == max_depth:
        return Node(data, target, depth, max_depth)

    best_feature, best_threshold = find_best_split(data, target)

    if best_feature is None:
        return Node(data, target, depth, max_depth)

    left_indices = data[:, best_feature] <= best_threshold
    right_indices = data[:, best_feature] > best_threshold

    left_data, left_target = data[left_indices], target[left_indices]
    right_data, right_target = data[right_indices], target[right_indices]

    node = Node(data, target, depth, max_depth)
    node.feature = best_feature
    node.threshold = best_threshold

    node.left = build_tree(left_data, left_target, depth + 1, max_depth)
    node.right = build_tree(right_data, right_target, depth + 1, max_depth)

    return node


# In[11]:


def predict_tree(node, sample):
    if node.left is None and node.right is None:
        return node.prediction

    if sample[node.feature] <= node.threshold:
        return predict_tree(node.left, sample)
    else:
        return predict_tree(node.right, sample)


# In[12]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# In[13]:


max_depths = [1, 2, 3, 4, 5]
for depth in max_depths:
    root_node = build_tree(X_train, y_train, depth=depth, max_depth=depth)
    y_pred = np.array([predict_tree(root_node, sample) for sample in X_test])

    # Evaluate Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTree Depth {depth}, Accuracy: {accuracy}")
    
    # Calculate precision
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    print(f"Tree Depth {depth}, Precision: {precision}")

    # Calculate recall
    recall = recall_score(y_test, y_pred, average='weighted')
    print(f"Tree Depth {depth}, Recall: {recall}")

    # Calculate F1 Score
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Tree Depth {depth}, F1 Score: {f1}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Tree Depth {depth}, Confusion Matrix:\n")
    print(conf_matrix)


# In[14]:


def print_rules(node, rule="if ", depth=0):
    if node.left is None and node.right is None:
        print(f"{'  ' * depth}then class {node.prediction}")
        return

    print(f"{'  ' * depth}{rule}Feature {node.feature} <= {node.threshold}")
    print_rules(node.left, "and ", depth + 1)
    print_rules(node.right, "and not ", depth + 1)
print("Decision Tree Rules:")
print_rules(root_node)


# ## Error Analysis for Classification
# 
# It seems that the performance metrics (Accuracy, Precision, Recall, F1 Score) and the confusion matrix are consistent across different tree depths. All the values are the same for each tree depth, which is unusual and might indicate an issue in my implementation or data.

# ## Conclusion
# In this assignment, we learnt to generate Decision trees according to ID3 algorithm.

# ## References
# https://medium.com/geekculture/step-by-step-decision-tree-id3-algorithm-from-scratch-in-python-no-fancy-library-4822bbfdd88f
# 
# https://iq.opengenus.org/id3-algorithm/
# 
# https://stackoverflow.com
# 
# BBM406 Lecture Notes
# 
# https://towardsdatascience.com/id3-decision-tree-classifier-from-scratch-in-python-b38ef145fd90
