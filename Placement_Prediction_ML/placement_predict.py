#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE

#Importing dataset
dataset = pd.read_csv('./Placement_Data_Full_Class.csv')
print(dataset.head())
print(dataset.describe())
print(dataset.info())

#Dropping unwanted column
dataset.drop("sl_no",axis=1,inplace=True)

#### Categerocal columns
cat_cols=(dataset.dtypes=="object")
cat_list = list(cat_cols[cat_cols].index)
for i in cat_list:
    print(i,dataset[i].unique(),pd.value_counts(dataset[i]).unique())

#Label Encoding Categorical Columns
maping_cat={"gender":{"M":1,"F":0},
              "ssc_b":{"Central":1,"Others":0},
              "hsc_b":{"Central":1,"Others":0},
              "hsc_s":{"Commerce":1,"Science":2,"Arts":0},
              "degree_t":{"Sci&Tech":1,"Comm&Mgmt":2,"Others":0},
              "workex":{"No":0,"Yes":1},
              "specialisation":{"Mkt&HR":1,"Mkt&Fin":0},
              "status":{"Placed":1,"Not Placed":0}}
dataset.replace(maping_cat,inplace=True)

#One Hot Encoding Categorical Columns
dummies_hsc_s=pd.get_dummies(dataset['hsc_s'],prefix='hsc_s')
dummies_degree_t=pd.get_dummies(dataset['degree_t'],prefix='degree_t')
dataset.drop("hsc_s",axis=1,inplace=True)
dataset.drop("degree_t",axis=1,inplace=True)
dataset=pd.concat([dataset,dummies_hsc_s],axis=1)
dataset=pd.concat([dataset,dummies_degree_t],axis=1)
dataset.drop("hsc_s_0",axis=1,inplace=True)
dataset.drop("degree_t_0",axis=1,inplace=True)

#### Numeric Columns
num_cols=(dataset.dtypes=="float64")
num_list=list(num_cols[num_cols].index)
for i in num_list:
    print(i,dataset[i].isnull().sum())

# Taking care of missing data in numeric columns
dataset['salary'].fillna(dataset['salary'].mean(),inplace=True)


# Extracting categorical and numerical columns for Analyzing
cat_cols=['gender','ssc_b','hsc_b','hsc_s_1','hsc_s_2','degree_t_1','degree_t_2','workex','specialisation','status']
dataset_cat=dataset[cat_cols]
num_cols=['ssc_p','hsc_p','degree_p','etest_p','mba_p','salary']
dataset_num=dataset[num_cols]

# Heat map to find correlation
plt.figure(figsize=(15, 10))
sns.heatmap(dataset.corr(),linecolor='white',linewidths=1,cmap='coolwarm',annot=True)
plt.show()

#Analyzing Numeric columns
def dist_plot(data_set):
    for i,col in enumerate(data_set.columns):
        plt.figure()
        sns.distplot(data_set[col])
dist_plot(dataset_num)

dataset_num_corr=pd.concat([dataset_num,dataset['status']],axis=1)
sns.pairplot(dataset_num_corr,height=3,hue='status')

#Analzing Categorical Columns
def count_plot(data_set):
    for i in data_set.columns:
        plt.figure()
        sns.countplot(x=data_set[i])
        plt.show()
count_plot(dataset_cat)

def hist_plot(data_set,dep_col):
    for col in data_set.columns:
        if col!=dep_col:
            plt.figure()
            sns.histplot(data=data_set,x=col,hue=dep_col,multiple='dodge')
hist_plot(dataset_cat,'status')

#Independent and dependent columns
X_cols=['gender','ssc_b','hsc_b','hsc_s_1','hsc_s_2','degree_t_1','degree_t_2','specialisation','ssc_p','hsc_p','degree_p','etest_p','mba_p']
X=dataset[X_cols]
Y=dataset['status']

#Splitting to training and testing set
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=0)
X_valid,X_test,Y_valid,Y_test=train_test_split(X_test,Y_test,test_size=0.5,random_state=3)

#Feature Scaling
norm=Normalizer().fit(X)
norm.transform(X_train)
norm.transform(X_test)


models={
    'GaussianNB':GaussianNB(),
    'MultinomialNB':MultinomialNB(),
    'BernoulliNB':BernoulliNB(),
    'LogisticRegression':LogisticRegression(),
    'RandomForestClassifier':RandomForestClassifier(),
    'SupportVectorMachine':SVC(),
    'DecisionTreeClassifier':DecisionTreeClassifier(),
    'KNeighborsClassifier':KNeighborsClassifier(),
    'GradientBoostingClassifier':GradientBoostingClassifier(),
    'Stochastic Gradient Descent':SGDClassifier(max_iter=5000,random_state=0),
    'XGBClassifier':XGBClassifier()
}

modelNames=[
    "GaussianNB",
    "MultinomialNB",'BernoulliNB',
    'LogisticRegression',
    'RandomForestClassifier',
    'SupportVectorMachine',
    'DecisionTreeClassifier',
    'KNeighborsClassifier',
    'GradientBoostingClassifier',
    'Stochastic Gradient Descent',
    'XGBClassifier'
    ]


train_accuracies=[]
validation_accuracies=[]
test_accuracies=[]

#Training models and predicting the test data
for i in models:
  model=models[i]
  model.fit(X_train,Y_train)
  print(i)
  train_accuracy=model.score(X_train,Y_train)
  validation_accuracy=model.score(X_valid,Y_valid)
  test_accuracy=model.score(X_test,Y_test)
  print("Train accuracy:",train_accuracy*100)
  print('Validation accuracy:',validation_accuracy*100)
  print('Test accuracy:',test_accuracy*100)
  train_accuracies.append(train_accuracy*100)
  validation_accuracies.append(validation_accuracy*100)
  test_accuracies.append(test_accuracy*100)
  y_pred=model.predict(X_test)
  cm=confusion_matrix(y_pred,Y_test)
  print('Confussion Matrix:',cm)
  tn=cm[0,0]
  fp=cm[0,1]
  tp=cm[1,1]
  fn=cm[1,0]
  accuracy=(tp + tn)/(tp+fp+tn+fn)
  precision=tp/(tp+fp)
  recall=tp/(tp+fn)
  f1score=2*precision*recall/(precision+recall)
  specificity=tn/(tn+fp)
  print('Accuracy:',accuracy)
  print('Precision:',precision)
  print('Recall:',recall)
  print('F1 score:',f1score)
  print('Specificity:',specificity) 
  print("#######################################")


#Visualizing Traing,Validataion and Testing Accuracies
barWidth=0.30
b1=train_accuracies
b2=validation_accuracies
b3=test_accuracies 
l1=np.arange(len(b1))
l2=[x+barWidth for x in l1]
l3=[x+barWidth for x in l2]

plt.figure(figsize=(30,15))
sns.set_style('darkgrid')
plt.title('Train,Validation,Test Accuracies of Models',fontweight='bold',size=30)
plt.bar(l1,b1,color='blue',width=barWidth,edgecolor='white',label='train',yerr=0.5,ecolor="black",capsize=10)
plt.bar(l2,b2,color='green',width=barWidth,edgecolor='white',label='validation',yerr=0.5,ecolor="black",capsize=10)
plt.bar(l3,b3,color='red',width=barWidth,edgecolor='white',label='test',yerr=0.5,ecolor="black",capsize=10)
plt.xlabel('Algorithms',fontweight='bold',size=30)
plt.ylabel('Accuracies',fontweight='bold',size=30)
plt.xticks([r+barWidth for r in range(len(b1))],modelNames,rotation=60,size=25)
plt.legend()
plt.show()

#Displaying Testing Accuracies of models
for i in range(len(models)):
    print('Accuracy of',modelNames[i],'is:',test_accuracies[i])

#Feature Selection using RFE
model=LogisticRegression()
rfe=RFE(model,6)
fit=rfe.fit(X,Y)
print("Num Features",fit.n_features_)
print("Selected Features",fit.support_)
print("Features Ranking",fit.ranking_)

selected_col_index=[]
for i,b in enumerate(fit.support_):
    if b==True:
        selected_col_index.append(X_cols[i])
X_select=X[selected_col_index]
print('The Selected columns are:\n',X_select.columns)

#Splitting to traing and testing set
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)
X_valid,X_test,Y_valid,Y_test=train_test_split(X_test,Y_test,test_size=0.5,random_state=3)

#Feature Scaling
norm=Normalizer().fit(X)
norm.transform(X_train)
norm.transform(X_test)

#Training model
model=LogisticRegression()
model.fit(X_train,Y_train)

#Predicting test data
y_pred=model.predict(X_test)

#Confusion matrix and accuracies
cm=confusion_matrix(y_pred,Y_test)
test_accuracy=model.score(X_test,Y_test)
print("Confusion matrix\n",cm)
print("Accuracy",test_accuracy*100)
