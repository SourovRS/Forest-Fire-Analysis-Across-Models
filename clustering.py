#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix


# In[2]:


import pandas as pd

# Make sure to properly escape backslashes in the file path
file_path = r"C:\Users\KIIT01\Downloads\archive (1)\Algerian_forest_fires_dataset.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display the DataFrame
print(df)


# In[3]:


df.isnull().sum()


# In[4]:


# CHECK MISSING VALUES

df[df.isnull().any(axis=1)] # checking if there any null value in any column 


# In[5]:


# Region 1 = Bejaia Region
df.loc[:122,'Region']=1
#Region 2 = Sidi-Bel Abbes Region
df.loc[122:,'Region']=2
df[['Region']] = df[['Region']].astype(int)


# In[6]:


df.isnull().sum()


# In[7]:


# remove null value
df=df.dropna().reset_index(drop=True)
df


# In[8]:


df.isnull().sum()


# In[9]:


df.iloc[[122]]


# In[10]:


# remove 122th column
df= df.drop(122).reset_index(drop=True)


# In[11]:


# fix spaces in column name
df.columns=df.columns.str.strip()
df.columns


# In[12]:


# change the required column as integer data type
df[['month','day','year','Temperature','RH','Ws']]= df[['month','day','year','Temperature','RH','Ws']].astype(int)
# Changing the other columns to Float data type
df[['Rain','FFMC','DMC', 'DC', 'ISI', 'BUI', 'FWI']]=df[['Rain','FFMC','DMC', 'DC', 'ISI', 'BUI', 'FWI']].astype(float)


# In[13]:


df1=df.drop(['day','month','year'],axis=1)


# In[14]:


## encoding of the categories in classes
df1['Classes']=np.where(df1['Classes'].str.contains('not fire'),0,1)


# In[15]:


#sns.set_style('seaborn')
plt.style.use('seaborn')
df1.hist(bins=50,figsize=(20,15))
plt.show()


# In[16]:


percentage=df1["Classes"].value_counts(normalize=True)*100
classlabels=['Fire','Not Fire']
plt.figure(figsize=(12,7))
plt.pie(percentage, labels=classlabels,autopct='%1.1f%%')
plt.title("Pie Chart of Classes")
plt.show()


# In[17]:


## correlation 
df1.corr()


# In[18]:


#pairplot
# sns.pairplot(df1.corr())


# In[19]:


dftemp= df.loc[df['Region']== 2]
plt.subplots(figsize=(13,6))
sns.set_style('whitegrid')
sns.countplot(x='month',hue='Classes',data= df,ec = 'black', palette= 'Set2')
plt.title('Fire Analysis Month wise for Sidi-Bel Abbes Region', fontsize=18, weight='bold')
plt.ylabel('Count', weight = 'bold')
plt.xlabel('Months', weight= 'bold')
plt.legend(loc='upper right')
plt.xticks(np.arange(4), ['June','July', 'August', 'September',])
plt.grid(alpha = 0.5,axis = 'y')
plt.show()


# In[20]:


dftemp= df.loc[df['Region']== 1]
plt.subplots(figsize=(13,6))
sns.set_style('whitegrid')
sns.countplot(x='month',hue='Classes',data= df,ec = 'black', palette= 'Set2')
plt.title('Fire Analysis Month wise for Bejaia Region', fontsize=18, weight='bold')
plt.ylabel('Count', weight = 'bold')
plt.xlabel('Months', weight= 'bold')
plt.legend(loc='upper right')
plt.xticks(np.arange(4), ['June','July', 'August', 'September',])
plt.grid(alpha = 0.5,axis = 'y')
plt.show()


# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# In[22]:


df1.corr()


# In[23]:


df1


# In[24]:


# fix spaces in column name
df1.columns=df1.columns.str.strip()
df1.columns


# In[25]:


# Split the data into features (X) and target variable (y)
X = df1.drop(['RH', 'Ws', 'Rain','Region'], axis=1)
y = df1['Classes']
# separate dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=0)


# In[26]:


from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data (replace this with your actual data loading)
X_train, _ = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)

# Create KMeans clustering model
model = KMeans(n_clusters=2)  # Assuming you want to create 2 clusters

# Fit the model to the training data
model.fit(X_train)

# Predict cluster labels for the training data
y_pred = model.predict(X_train)

# Print the cluster labels for the training data
print("Cluster labels for the training data:")
print(y_pred)



# In[27]:


from sklearn.metrics import silhouette_score

# Assuming you have trained a KMeans clustering model `model` on your training data

# Calculate silhouette score
silhouette_avg = silhouette_score(X_train, y_pred)

# Print silhouette score
print("Silhouette Score:", silhouette_avg)




# In[28]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

# Assuming you have ground truth labels `y_true` available for evaluation
# For demonstration purposes, let's generate random ground truth labels
import numpy as np
y_true = np.random.randint(0, 2, size=len(y_pred))  # Random binary labels (0 or 1)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Calculate precision
precision = precision_score(y_true, y_pred)

# Calculate recall (sensitivity)
recall = recall_score(y_true, y_pred)

# Calculate specificity
# Specificity is not directly available from confusion matrix in multiclass setting.
# It's usually calculated in binary classification problems.
# However, you can use it as a measure of clustering performance if you have binary labels.

# Calculate F1-score
f1 = f1_score(y_true, y_pred)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print("Precision:", precision)
print("Recall (Sensitivity):", recall)
print("F1 Score:", f1)


# In[29]:


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Assuming you have calculated the confusion matrix and stored it in `conf_matrix`
# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.unique(y_true))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




