#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


df=pd.read_csv('train.csv')


# In[6]:


df


# In[7]:


df.head()


# In[8]:


df.shape


# In[14]:


# Performing Data processing
#1 Removing columns
df = df.drop ( columns=['PassengerId','Name','Cabin','Ticket'],axis=1)


# In[15]:


df.describe()


# In[17]:


df.dtypes


# In[18]:


#2 to know unique value
df.nunique()


# In[21]:


#3 for missing vslues
df.isnull().sum()


# In[24]:


#4 replacing missing values
df['Age']=df['Age'].replace(np.nan,df['Age'].median(axis=0))
df['Embarked']=df['Embarked'].replace(np.nan,'s')


# In[25]:


#5 casting age to integer
df['Age']=df['Age'].astype(int)


# In[28]:


#6 replacing with 1 and 0
df['Sex'] = df['Sex'].apply(lambda x  :1 if x == 'male' else 0)


# In[30]:


#7 creating age group (0-18),(18-30),(30-50),(50-100)
df['Age']=pd.cut(x=df['Age'],bins=[0,5,20,30,40,50,60,100],labels = ['Infant','Teen','20s','30s','40s','50s','60s'])


# In[32]:


#8 visulizing the count of the features
fig, ax = plt.subplots(2,4,figsize=(20,20))
sns.countplot(x = 'Survived',data = df,ax= ax[0,0])
sns.countplot(x = 'Pclass',data = df,ax= ax[0,1])
sns.countplot(x = 'Sex',data = df,ax= ax[0,2])
sns.countplot(x = 'Age',data = df,ax= ax[0,3])
sns.countplot(x = 'Embarked',data = df,ax= ax[1,0])
sns.countplot(x = 'Fare',data = df,ax= ax[1,1])
sns.countplot(x = 'SibSp',data = df,ax= ax[1,2])
sns.countplot(x = 'Parch',data = df,ax= ax[1,3])


# In[40]:


fig, ax = plt.subplots(2,4,figsize=(20,20))
sns . countplot(x = 'Sex', data = df,hue = 'Survived' , ax= ax[0,0])
sns.countplot(x = 'Age', data = df,hue = 'Survived',ax = ax[0,1])
sns.boxplot(x ='Sex',y='Fare',data = df,hue='Pclass', ax=ax[0,2])
sns.countplot(x ='SibSp', data = df,hue ='Survived',ax=ax[0,3] )
sns.countplot(x = 'Parch', data = df,hue = 'Survived',ax=ax[1,0] )
sns.scatterplot(x = 'SibSp',  y ='Parch',data=df,hue='Survived',ax=ax[1,1])
sns.boxplot(x ='Embarked',y ='Fare',data= df,ax=ax[1,2])
sns.pointplot(x ='Pclass',y ='Survived',data = df, ax=ax[1,3])


# In[42]:


# Data Preprocessing
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(['S','C','Q'])
df['Embarked']= le.transform(df['Embarked'])


# In[44]:


age_mapping = {
    'infant':0,
    'teen':1,
    '20s':2,
    '30s':3,
    '40s':4,
    '50s':5,
    'elder':6
}
df['Age'] = df['Age'].map(age_mapping)
df.dropna(subset=['Age'], axis= 0, inplace = True)


# In[45]:


#1 Heatmap
sns.heatmap(df.corr(), annot= True)


# In[46]:


#2 seperating the target and independat veriable
y= df['Survived']
x= df.drop(columns=['Survived'])


# In[49]:


#3 logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr


# In[50]:


LogisticRegression()


# In[51]:


lr.fit(x,y)
lr.score(x,y)


# In[53]:


#4 decision tree classifier
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree

dtree.fit(x,y)
dtree.score(x,y)
# In[55]:


dtree.fit(x,y)
dtree.score(x,y)


# In[58]:


#5 support vector (SVM)
from sklearn.svm import SVC
svm = SVC()
svm


# In[59]:


from sklearn.svm import SVC
svm = SVC()
svm


# In[60]:


svm.fit(x,y)
svm.score(x,y)


# In[62]:


#6 k-nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn


# In[63]:


knn.fit(x,y)
knn.score(x,y)


# In[64]:


# Import test set data
df2 = pd.read_csv('test.csv')

df2.head()
# In[66]:


df2= df2.drop(columns = ['PassengerId','Name','Cabin','Ticket'],axis=1)


# In[69]:


#1 data processing
df2['Age'] = df2 ['Age'].replace(np.nan,df2['Age'].median(axis=0))
df2['Embarked']= df2['Embarked'].replace(np.nan,'S')


# In[70]:


df2['Age'] = df2['Age'].astype(int)


# In[73]:


#2 replace with 1 and female with 0
df2['Sex'] = df2['Sex'].apply(lambda x : 1 if x =='male' else 0)


# In[74]:


df2['Age']=pd.cut(x=df2['Age'],bins=[0,5,20,30,40,50,60,100], labels = [0,1,2,3,4,5,6])


# In[75]:


le.fit(['S','C','Q'])
df2['Embarked'] = le.transform(df2['Embarked'])


# In[76]:


df.dropna(subset=['Age'],axis= 0, inplace = True)


# In[77]:


df.head()


# In[82]:


#3 separating
x = df2.drop(columns=['Survived'])
y = df2['Survived']


# In[84]:


#4 Predict tree classifier
tree_pred = dtree.predict(x)


# In[85]:


from sklearn.metrics import accuracy_score
accuracy_score(y, tree_pred)


# In[88]:


#5 confusion matrix
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(y,tree_pred),annot = True, cmap = 'Blues')
plt.ylabel('predicted Values')
plt.xlabel('Actual Values')
plt.title('confusion matrix')
plt.show()


# In[ ]:




