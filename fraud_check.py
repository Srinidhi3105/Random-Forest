import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

df = pd.read_csv('Fraud_check.csv')
df.columns
df.head()

#box plot representation of a dependent variable
df['Taxable.Income'].plot(kind='box',vert=False,figsize=(14,6))

#density plot 
ax=df['Taxable.Income'].plot(kind='density',figsize=(14,6))

#plot to find out the mean and the median
ax.axvline(df['Taxable.Income'].mean(),color='red')
ax.avline(df['Taxable.Income'].mean(),color='green')

df['Undergrad'].value_counts()

#plotting a histogram
ax = df['Taxable.Income'].plot(kind = 'hist',figsize=(14,6))

#drop unwanted columns
df.drop(['Marital.Status'],axis=1,inplace=True)
df.drop(['Urban'],axis=1,inplace=True)

df.Undergrad[df.Undergrad=='NO']=1
df.Undergrad[df.Undergrad=='YES']=2

df['Taxable.Income'][df['Taxable.Income']>30000]="good"
df['Taxable.Income'][df['Taxable.Income']<30000]="Risky"

#define dependent variable Y
Y = df['Taxable.Income'].values

X= df.drop(labels=['Taxable.Income'],axis=1)

#split data into training and  test
from sklearn.model_selection import train_test_split

X_train,Y_train,X_test,Y_test = train_test_split(X,Y,test_size =0.2,random_state=20)

#Model building 
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=10,random_state =20)

model.fit = X_train,Y_train

prediction_test = model.predict(X_test)

from sklearn import metrics 

#measuring the accuracy between the predicted data and actual data
print("Accuracy= ",metrics.accuracy_score(Y_test,prediction_test))