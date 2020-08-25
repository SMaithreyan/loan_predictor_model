import pandas as pd
import numpy as np


train=pd.read_csv(r'C:\Users\raghu\Downloads\train_ctrUa4K.csv')
train.Loan_Status=train.Loan_Status.map({'Y':1,'N':0})
train.isnull().sum()

Loan_status=train.Loan_Status
train.drop('Loan_Status',axis=1,inplace=True)
test=pd.read_csv(r'C:\Users\raghu\Downloads\test_lAUu6dG.csv')
Loan_ID=test.Loan_ID
data=train.append(test)
data.head()

data.describe()

data.isnull().sum()

data.Dependents.dtypes

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)

data.Gender=data.Gender.map({'Male':1,'Female':0})
data.Gender.value_counts()


corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)

data.Married=data.Married.map({'Yes':1,'No':0})

data.Married.value_counts()

data.Dependents=data.Dependents.map({'0':0,'1':1,'2':2,'3+':3})

data.Dependents.value_counts()

corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


data.Education=data.Education.map({'Graduate':1,'Not Graduate':0})


data.Education.value_counts()

data.Self_Employed=data.Self_Employed.map({'Yes':1,'No':0})
data.Self_Employed.value_counts()

data.Property_Area.value_counts()

data.Property_Area=data.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})
data.Property_Area.value_counts()
corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)

data.head()

data.Credit_History.size

data.Credit_History.fillna(np.random.randint(0,2),inplace=True)

data.isnull().sum()

data.Married.fillna(np.random.randint(0,2),inplace=True)

data.isnull().sum()

data.LoanAmount.fillna(data.LoanAmount.median(),inplace=True)


data.Loan_Amount_Term.fillna(data.Loan_Amount_Term.mean(),inplace=True)

data.isnull().sum()

data.Gender.value_counts()

from random import randint 
data.Gender.fillna(np.random.randint(0,2),inplace=True)

data.Gender.value_counts()

data.Dependents.fillna(data.Dependents.median(),inplace=True)

data.isnull().sum()

corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)

data.Self_Employed.fillna(np.random.randint(0,2),inplace=True)

data.isnull().sum()

data.head()

data.drop('Loan_ID',inplace=True,axis=1)

data.isnull().sum()

train_X=data.iloc[:614,]
train_y=Loan_status
X_test=data.iloc[614:,]
seed=7


from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(train_X,train_y,random_state=seed)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


models=[]
models.append(("logreg",LogisticRegression()))
models.append(("tree",DecisionTreeClassifier()))
models.append(("lda",LinearDiscriminantAnalysis()))
models.append(("svc",SVC()))
models.append(("knn",KNeighborsClassifier()))
models.append(("nb",GaussianNB()))

seed=7
scoring='accuracy'

from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
result=[]
names=[]


for name,model in models:
    #print(model)
    kfold=KFold(n_splits=10,random_state=seed)
    cv_result=cross_val_score(model,train_X,train_y,cv=kfold,scoring=scoring)
    result.append(cv_result)
    names.append(name)
    print("%s %f %f" % (name,cv_result.mean(),cv_result.std()))


df_output=pd.DataFrame()

outp=svc.predict(X_test).astype(int)
outp

df_output['Loan_ID']=Loan_ID
df_output['Loan_Status']=outp

df_output['Loan_Status'] = df_output['Loan_Status'].map({0:'N', 1:'Y'})
df_output.head()

df_output[['Loan_ID','Loan_Status']].to_csv(r'C:\Users\raghu\Downloads\output.csv',index=False)