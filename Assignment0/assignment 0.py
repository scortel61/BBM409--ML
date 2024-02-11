#!/usr/bin/env python
# coding: utf-8

# # BBM409 Assignment_0
# Group Member: Oğuzhan Taşçı
# 
# Group Member: İbrahim Enes Genişyürek

# In[1]:


# import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


# In[2]:


# Read dataset from PC
df = pd.read_csv("star_classification.csv")


# In[3]:


df


# ## Pre-processing

# In[4]:


df.columns


# In[5]:


df.describe()


# In[6]:


# Looking for data types and checking whether datas from columns is null or not
df.info()


# In[7]:


# data type conversion for consistency between data types
df['run_ID'] = df['run_ID'].astype(float)
df['rerun_ID'] = df['rerun_ID'].astype(float)
df['cam_col'] = df['cam_col'].astype(float)
df['field_ID'] = df['field_ID'].astype(float)
df['plate'] = df['plate'].astype(float)
df['MJD'] = df['MJD'].astype(float)
df['fiber_ID'] = df['fiber_ID'].astype(float)


# In[8]:


#Looking for if columns attributes are unique or not
for column_name in df.columns:
    column_data = df[column_name]
    is_unique = column_data.nunique() == len(column_data)
    if is_unique:
        print('Column ' + column_name + ' contains unique values.')
    else:
        print('Column ' + column_name + ' does not contain unique values.')


# In[9]:


# Deleting unique column from data set
df = df.drop("spec_obj_ID", axis=1)


# In[10]:


df["class"].unique()


# In[11]:


df['class'].value_counts()


# In[12]:


df["class"] = df["class"].map({"GALAXY":0,"QSO":1, "STAR":2})


# In[13]:


X = df.drop("class", axis=1)
y = df["class"]


# ## Split the dataset

# In[14]:


# %80 percent Training %20 Testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# ## kNN

# In[15]:


# import necessary libary for kNN
from sklearn.neighbors import KNeighborsClassifier


# In[16]:


knn = KNeighborsClassifier(n_neighbors=3)


# In[17]:


knn = knn.fit(X_train, y_train)


# In[18]:


knn_pred = knn.predict(X_test)


# In[19]:


print(classification_report(y_test, knn_pred))


# In[20]:


print(confusion_matrix(y_test, knn_pred))


# ## Naive Bayes

# In[21]:


# import necessary libary for Naive Bayes
from sklearn.naive_bayes import GaussianNB


# In[22]:


nb = GaussianNB()


# In[23]:


nb= nb.fit(X_train,y_train)


# In[24]:


nb_pred = nb.predict(X_test)


# In[25]:


print(classification_report(y_test,nb_pred))


# In[26]:


print(confusion_matrix(y_test,nb_pred))


# ## Random Forest

# In[27]:


# import necessary libary for Random Forest
from sklearn.ensemble import RandomForestClassifier


# In[28]:


rf= RandomForestClassifier(n_estimators=100,random_state=42)


# In[29]:


rf = rf.fit(X_train, y_train)


# In[30]:


rf_pred=rf.predict(X_test)


# In[31]:


print(classification_report(y_test,rf_pred))


# In[32]:


print(confusion_matrix(y_test,rf_pred))


# ## SVM

# In[33]:


from sklearn.svm import SVC


# In[34]:


svm = SVC()


# In[35]:


svm = svm.fit(X_train,y_train)


# In[36]:


svm_pred = svm.predict(X_test)


# In[37]:


print(classification_report(y_test,svm_pred))


# In[38]:


print(confusion_matrix(y_test,svm_pred))


# ## 5-fold cross validation

# ## kNN 

# In[39]:


from sklearn.model_selection import cross_val_score, KFold


# In[40]:


knn_5_fold = KNeighborsClassifier(n_neighbors=3)


# In[41]:


k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)


# In[42]:


scores = cross_val_score(knn_5_fold, X_train, y_train, cv=kf)


# In[43]:


print(f'Cross-validation scores: {scores}')
print(f'Average accuracy: {np.mean(scores):.2f}')


# ## Naive Bayes

# In[44]:


nb_5_fold = GaussianNB()


# In[45]:


scores = cross_val_score(nb_5_fold, X_train, y_train, cv=kf)


# In[46]:


print(f'Cross-validation scores: {scores}')
print(f'Average accuracy: {np.mean(scores):.2f}')


# ## Random Forest

# In[47]:


rf_5_fold = RandomForestClassifier(n_estimators=100,random_state=42)


# In[48]:


scores = cross_val_score(rf_5_fold, X_train, y_train, cv=kf)


# In[49]:


print(f'Cross-validation scores: {scores}')
print(f'Average accuracy: {np.mean(scores):.2f}')


# ## SVM

# In[50]:


svm_5_fold = SVC()


# In[ ]:


scores = cross_val_score(svm_5_fold, X_train, y_train, cv=kf)


# In[ ]:


print(f'Cross-validation scores: {scores}')
print(f'Average accuracy: {np.mean(scores):.2f}')

