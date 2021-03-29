#!/usr/bin/env python
# coding: utf-8

# In[121]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_palette("pastel")
sns.set_context('talk')
sns.set_style('white')

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Details of the Dataset
# The dataset contains a mix of categorical and numeric type data.
# 
# A)Categorical Attributes
# 
# 1)workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# Individual work category
# 
# 2)education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# 
# 3)Individual’s highest education degree
# 
# 4)marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# Individual marital status
# 
# 5)occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspect, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# 
# 6)Individual’s occupation
# 
# 7)relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# 
# 8)Individual’s relation in a family
# 
# 9)race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# 
# 10)Race of Individual
# 
# 11)sex: Female, Male.
# 
# 12)native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinidad Tobago, Peru, Hong, Holland-Netherlands.
# 
# 13)Individual’s native country
# 
# 
# B)Continuous Attributes
# 
# 1)age: continuous. Age of an individual
# 
# 2)fnlwgt: final weight, continuous.
# The weights on the CPS files are controlled to independent estimates of the civilian noninstitutional population of the US. These are prepared monthly for us by Population Division here at the Census Bureau.
# 3)capital-gain: continuous.
# 
# 4)capital-loss: continuous.
# 
# 5)hours-per-week: continuous.
# 
# 6)Individual’s working hour per week

# ## Exploring the Data

# In[84]:


data = pd.read_csv('adult.csv')


# In[85]:


data.head()


# In[86]:


data.isnull().sum()


# In[87]:


data.dtypes


# In[88]:


data.nunique()


# In[89]:


data.info()


# In[90]:


data.describe(include='all').T


# In[125]:


df = data.copy()


# In[92]:


total_records = data['income'].count()
n_greater_50k = len(data[(data.income == '>50K')])
n_at_most_50k = len(data[(data.income == '<=50K')])
greater_percent = (n_greater_50k / total_records) * 100

print("Total number of records: {}".format(total_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))


# In[ ]:





# In[65]:


sns.histplot(data=data, x='income',hue='gender',palette='husl')
plt.title('Total Income Distribution')


# In[126]:


# Drop fnlwgt because it's unique
df.drop('fnlwgt',axis=1,inplace=True)


# In[127]:


# Dealing with the ‘?’by replacing it with the ‘MODE’.
df['workclass'] = df['workclass'].replace('?','Private')
df['occupation'] = df['occupation'].replace('?','Prof-specialty')
df['native-country'] = df['native-country'].replace('?','United-States')


# In[68]:


sns.pairplot(df)


# In[69]:


sns.histplot(data['capital-gain'])


# In[70]:


ax = sns.histplot(df['age'],kde=False,color='purple')
#plt.title('Age')


# In[71]:


ax=sns.histplot(data=df,x='workclass')

plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)
#plt.figure(figsize=(10,10))
plt.tight_layout()


# In[72]:


sns.histplot(data=df,x='education')
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)

plt.tight_layout()


# In[73]:


sns.histplot(data=df, x= 'hours-per-week')
plt.tight_layout()


# In[131]:


sns.histplot(data=df,x='occupation')
plt.ylim(0,10000)

plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
   fontsize='x-large'  
)

plt.tight_layout()


# In[75]:


df.plot(kind='box',figsize=(11,11),layout=(3,3),sharex=False,subplots=True);
plt.tight_layout()


# In[77]:


sns.heatmap(df.corr(),annot=True,cmap='Blues')


# ## Key Findings
# The minimum age is 17 and the maximum is 90 years, most of the working age group lies between 20-40
# 
# The minimum hours-per-week is 1 and maximum is 90, with most of the count lying between 30-40
# 
# outliers observed in almost all the numeric features, these are the extreme values that are present in the data.
# 
# Not very strong correlation observed among variables

# ## Feature Scaling

# In[93]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
workclass = le.fit_transform(df['workclass'])
sex = le.fit_transform(df['gender'])
edu = le.fit_transform(df['education'])
occu = le.fit_transform(df['occupation'])
race = le.fit_transform(df['race'])
native_country = le.fit_transform(df['native-country'])
marital_status = le.fit_transform(df['marital-status'])
relationship = le.fit_transform(df['relationship'])
income = le.fit_transform(df['income'])


# In[94]:


df.drop(['workclass','education','occupation','race','gender','native-country','marital-status','relationship','income'], inplace=True, axis=1)


# In[95]:


df['workclass'] = workclass
df['gender'] = sex
df['education'] = edu
df['occupation'] = occu
df['race'] = race
df['native-country'] = native_country
df['marital_-status'] = marital_status
df['relationship'] = relationship
df['income'] = income
df.head()


# In[96]:



from sklearn.preprocessing import StandardScaler
df[['age','educational-num','capital-gain','capital-loss','hours-per-week']] = StandardScaler().fit_transform(df[['age','educational-num','capital-gain','capital-loss','hours-per-week']])


# ## Training the Model and Making Predictions

# In[97]:


y = df.iloc[:,-1]
x = df.iloc[:,:-1]

print(x.shape)
print(y.shape)


# In[101]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)


# In[111]:


from sklearn.linear_model import LogisticRegression

reg = LogisticRegression()
reg.fit(x_train, y_train)
prediction1 = reg.predict(x_test)
print('Accuracy on test set:{:.2f}'.format(accuracy_score(y_test,prediction1)))


# In[ ]:





# In[109]:


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
model2 = dtc.fit(x_train,y_train)
prediction2=model2.predict(x_test)
print('Accuracy on test set:{:.2f}'.format(accuracy_score(y_test,prediction2)))


# In[ ]:





# In[114]:


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
model3=rfc.fit(x_train,y_train)
prediction3 = model3.predict(x_test)

print('Accuracy on test set:{:.2f}'.format(rfc.score(x_test,y_test)))


# In[ ]:





# In[115]:


from sklearn.metrics import confusion_matrix,classification_report

print(classification_report(y_test,prediction3))


# In[116]:


cm = confusion_matrix(y_test,prediction3)
cm


# In[118]:


TP= cm[0][0]
FP=cm[1][0]
TN = cm[1][1]
FN = cm[0][1]
print('Precision:',TP/(TP+TN))

print('Recall:',TP/(TP+FN))


# In[119]:


# for the other class 1 (>50)
print('Precision:', TN/(FN+TN) )
print('Recall:', TN/(TN+FP))


# ## Key Findings
# Random Forest Classifier is giving the best accuracy on test data: 86%
# 
# Logistic Regression Classifier accuracy is: 77%
# 
# Decision Tree Classifier accuracy is: 82%
# 
# Tried to explain Confusion Matrix along with mentioning the formula and how it can be calculated for both the classes.
# 
# Further Scope:
# Apply Boosting Algorithms, can go parameter tuning to improve the performance of the test results

# In[ ]:




