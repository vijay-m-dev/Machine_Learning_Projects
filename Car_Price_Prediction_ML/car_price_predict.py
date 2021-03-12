# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

#Reading dataset
dataset = pd.read_csv('./toyota.csv')
print(dataset.head())
print(dataset.describe())
print(dataset.info())

#Categotical Columns
cat_list=['model','transmission','fuelType']
dataset_cat=dataset[cat_list]
for i in cat_list:
    print("--------------------------")
    print(i,dataset[i].unique())
    print(pd.value_counts(dataset[i]))
    print("--------------------------")

#Visualizing Categorical Columns
def count_plot(data_set):
    for i in data_set.columns:
        plt.figure()
        sns.countplot(x=data_set[i])
        plt.show()
count_plot(dataset_cat)

#One Hot Encoding
dataset_cat=pd.get_dummies(dataset_cat)
dataset_cat.drop(['model_ Verso-S','transmission_Other','fuelType_Other'],axis=1,inplace=True)

#Numerical columns
num_list=['year','price','mileage','tax','mpg','engineSize']
dataset_num=dataset[num_list]
for i in num_list:
    print(i,dataset[i].isnull().sum())


#Visualizing numerical columns
def dist_plot(data_set):
    for i,col in enumerate(data_set.columns):
        plt.figure()
        sns.distplot(data_set[col])
dist_plot(dataset_num)

#Correlation
plt.figure(figsize=(15, 10))
sns.heatmap(dataset.corr(),linecolor='white',linewidths=1,cmap='coolwarm',annot=True)
plt.show()

dataset=pd.concat([dataset_num,dataset_cat],axis=1)

plt.figure(figsize=(20,20))
sns.heatmap(dataset.corr(),linecolor='white',linewidths=1,cmap='coolwarm',annot=True)
plt.show()

#Independent and Dependent columns
X=dataset.copy()
X.drop(['price'],axis=1,inplace=True)
Y=dataset['price']

#Training Testing split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=0)

#Feature Scaling
sc=StandardScaler()
X_Train=sc.fit_transform(X_train)
X_Test=sc.transform(X_test)



models=[LinearRegression(),
        Lasso(),
        Ridge(),
        ElasticNet(),
        DecisionTreeRegressor(random_state=0),
        RandomForestRegressor(n_estimators=10,random_state=0)
        ]
model_names=['Linear Regression',
             'Lasso Regression',
             'Ridge Regression',
             'Elastic Regresion',
             'Decision Tree',
             'Random Forest'
             ]
train_scores=[]
test_scores=[]
mse={}

#Training and testing models
for i in range(len(models)):
  model=models[i]
  #Training the model
  model.fit(X_train,Y_train)
  train_score=model.score(X_train,Y_train)
  test_score=model.score(X_test,Y_test)
  train_scores.append(train_score)
  test_scores.append(test_score)
  #Predicting the test data
  y_pred=model.predict(X_test)
  #MSE for model
  mse[model_names[i]]=metrics.mean_squared_error(Y_test,y_pred)
  #Displaying R-Square values
  print("---------------------------------------------------------------------")
  print("R-Squared for ",model_names[i],"training data is:",train_scores[i])
  print("R-Squared for ",model_names[i],"testing data is:",test_scores[i])
  print("---------------------------------------------------------------------")

#Sorting Mean Squared Error of models
#Random Forest has low MSE value
sort_mse=sorted(mse.items(),key=lambda x: x[1])
for i in sort_mse:
    print("MSE for ",i[0],"is: ",i[1])






