#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
sns.pairplot(df1.corr())


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


from sklearn.svm import SVC

# Create SVM model
model = SVC()
model.fit(X_train, y_train)


# In[27]:


# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred


# In[28]:


# check accuracy and cofusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
from sklearn import metrics

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)


# In[29]:


loreg_cm = ConfusionMatrixDisplay.from_estimator( model,X_test, y_test)


# In[30]:


# Precision
Precision = metrics.precision_score(y_test, y_pred)
Sensitivity_recall = metrics.recall_score(y_test, y_pred)
Specificity = metrics.recall_score(y_test, y_pred, pos_label=0)
F1_score = metrics.f1_score(y_test, y_pred)
#metrics
print({"Accuracy":accuracy , "Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"Specificity":Specificity,"F1_score":F1_score})


# In[ ]:


sns.pairplot(df1.corr())

