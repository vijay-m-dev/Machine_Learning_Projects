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

#Reading dataset
dataset = pd.read_csv('./insurance.csv')
print(dataset.head())
print(dataset.describe())
print(dataset.info())

#Categotical Columns
cat_cols=(dataset.dtypes=="object")
cat_list = list(cat_cols[cat_cols].index)
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

#Label Encoding Categorical columns
maping_cat={"gender":{"male":1,"female":0},
              "smoker":{"yes":1,"no":0},
              "region":{"southwest":0,"southeast":1,"northwest":2,"northeast":3}}
dataset.replace(maping_cat,inplace=True)

#One Hot Encoding Categorical columns
dummies_region=pd.get_dummies(dataset['region'],prefix='region')
dataset.drop("region",axis=1,inplace=True)
dataset=pd.concat([dataset,dummies_region],axis=1)
dataset.drop("region_0",axis=1,inplace=True)

#Numeric Columns
num_list=['age','bmi','charges']
dataset_num=dataset[num_list]
for i in num_list:
    print(i,dataset[i].isnull().sum())

#Visualizing Distribution of Numeric Columns
def dist_plot(data_set):
    for i,col in enumerate(data_set.columns):
        plt.figure()
        sns.distplot(data_set[col])
dist_plot(dataset_num)

#Visualising Correlation
plt.figure(figsize=(15, 10))
sns.heatmap(dataset.corr(),linecolor='white',linewidths=1,cmap='coolwarm',annot=True)
plt.show()

#Visualising all columns with more correlated 'smoker' column
def hist_plot(data_set,dep_col):
    for col in data_set.columns:
        if col!=dep_col:
            plt.figure()
            sns.histplot(data=data_set,x=col,hue=dep_col,multiple='dodge')
hist_plot(dataset_cat,'smoker')

#Dropping Unwanted columns
dataset.drop(['children','region_1','region_2','region_3'],axis=1,inplace=True)

#Visualising using pair plot
sns.pairplot(dataset) 
sns.pairplot(dataset,hue="smoker") 

#Independent and dependent variables
X_cols=['age','bmi','smoker']
X=dataset[X_cols]
Y=dataset['charges']

#Splitting to training and testing set
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=0)


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



