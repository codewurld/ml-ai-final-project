#!/usr/bin/env python
# coding: utf-8

# ## Library Import

# In[117]:




import pandas as pd
#array and matrices operations
import numpy as np
#splits data to train & test
from sklearn.model_selection import train_test_split
# library for logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
#accurate score measure
from sklearn.metrics import accuracy_score
#label encoder
from sklearn.preprocessing import LabelEncoder
#visualise dataset
import seaborn as sms
#import matplotlib as plt
import matplotlib.pyplot as plt
#avoid unnecessary information
import warnings
#to display plot continuously
get_ipython().run_line_magic('matplotlib', 'inline')
#parameters for warning
warnings.filterwarnings("ignore")


# # Data Analysis

# In[49]:


#loads dataset
data = pd.read_csv('Desktop/store-data/heart.csv')

#top of dataset
data.head()

#bottom of dataset
data.tail()


# In[62]:


#amount of rows and columns in data
data.shape


# In[63]:


#analyse data
data.info()


# In[87]:


#check for missing values - returned data is clean
data.isnull().sum()


# In[65]:


# get statistical top level insight about data 
data.describe()


# In[66]:


#distribution of sex
data['Sex'].value_counts()


# In[67]:


#distribution of exercise induced angina
data['ExerciseAngina'].value_counts()


# ### Data Cleaning

# In[68]:


#remove sex column
updated_data = data.drop(columns="Sex", axis=1)


# In[69]:


#input and output values
x = updated_data
y = updated_data['HeartDisease'] # 0 = False, 1 = True


# In[77]:


# separating training and test data - 70:30

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.3,random_state=0)


# In[78]:


print(x.shape, x_train.shape, x_test.shape)


# ## Data Encoding

# In[ ]:


#encoding data with label encoder to convert columns with text values

le = LabelEncoder()

data['ChestPainType'] = le.fit_transform(data['ChestPainType'])
data['RestingECG'] = le.fit_transform(data['RestingECG'])
data['ExerciseAngina'] = le.fit_transform(data['ExerciseAngina'])
data['ST_Slope'] = le.fit_transform(data['ST_Slope'])

data.tail()


# # Model Training

# In[79]:


model = LogisticRegression(solver='liblinear',
                           C=0.05,multi_class='ovr',
                           random_state=0)

model.fit(x_train, y_train)


# In[ ]:


# applying test data on model

y_pred = model.predict(x_test)


# ### Model Evaluation

# #### Accuracy Score

# In[88]:


#determine model accuracy on train data and likelihood of overfitting
model.score(x_train,y_train)


# In[89]:


#determine model accuracy on test data and likelihood of overfitting

model.score(x_test,y_test)


# #### Classification Report

# In[96]:


#determining precision, recall, etc.

print(classification_report(y_test,y_pred))
        
        


# ## K-Nearest Neighbour

# In[124]:


k=6
neigh = KNeighborsClassifier(n_neighbors=k)
neigh.fit(x_train, y_train)
pred_y = neigh.predict(x_test)

print(classification_report(y_test,pred_y))

print("Accuracy of model at K=6 is",metrics.accuracy_score(y_test, pred_y))




# In[ ]:




