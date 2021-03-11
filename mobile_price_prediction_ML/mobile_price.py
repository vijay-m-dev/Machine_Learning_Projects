#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#Reading dataset
dataset = pd.read_csv('./dataset.csv')

print(dataset.head())
print(dataset.shape)
print(dataset.describe())
print(dataset.info())
print(dataset.isnull().sum())

#EDA
plt.figure(figsize=(15, 10))
sns.heatmap(dataset.corr(),linecolor='white',linewidths=1,cmap='coolwarm',annot=True)
plt.show()

data_num=dataset[['battery_power','clock_speed','fc','int_memory','m_dep','mobile_wt','n_cores','pc','px_height','px_width','ram','sc_h','sc_w','talk_time']]
data_cat=dataset[['blue','dual_sim','four_g','three_g','touch_screen','wifi','price_range']]

def dist_plot(data_set):
    for i,col in enumerate(data_set.columns):
        plt.figure()
        sns.displot(data_set[col])

def count_plot(data_set):
    for i in data_set.columns:
        plt.figure()
        sns.countplot(x=data_set[i])
        plt.show()

def hist_plot(data_set,dep_col):
    for col in data_set.columns:
        if col!=dep_col:
            plt.figure()
            sns.histplot(data=data_set,x=col,hue=dep_col,multiple='dodge')


dist_plot(data_num)
count_plot(data_cat)
hist_plot(data_cat,'price_range')


X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#Splitting Training and Testing data
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#Feature Scaling
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Logistic Regression
classifier_log=LogisticRegression()
classifier_log.fit(X_train,Y_train)
print("Score of Logistic regression: {}".format(classifier_log.score(X_train,Y_train)*100))
y_pred_log=classifier_log.predict(X_test)
acc_log=accuracy_score(Y_test,y_pred_log)*100
print("Accuracy of Logistic Regression Classifier: {}%".format(acc_log))
print("Confusion Matrix of Logistic Regression Classifier: \n{}".format(confusion_matrix(Y_test,y_pred_log)))
print(classification_report(Y_test,y_pred_log))

#Naive Bayes
classifier_nb=GaussianNB()
classifier_nb.fit(X_train,Y_train)
print("Score of Naive Bayes: {}".format(classifier_nb.score(X_train,Y_train)*100))
y_pred_nb=classifier_nb.predict(X_test)
acc_nb=accuracy_score(Y_test,y_pred_log)*100
print("Accuracy of Naive Bayes Classifier: {}%".format(acc_nb))
print("Confusion Matrix of Naive Bayes Classifier: \n{}".format(confusion_matrix(Y_test,y_pred_nb)))
print(classification_report(Y_test,y_pred_nb))

#SVM
classifier_svm=SVC()
classifier_svm.fit(X_train,Y_train)
print("Score of SVM: {}".format(classifier_svm.score(X_train,Y_train)*100))
y_pred_svm=classifier_svm.predict(X_test)
acc_svm=accuracy_score(Y_test,y_pred_svm)*100
print("Accuracy of SVM Classifier: {}%".format(acc_svm))
print("Confusion Matrix of SVC Classifier: \n{}".format(confusion_matrix(Y_test,y_pred_svm)))
print(classification_report(Y_test,y_pred_svm))

#Decision Tree
classifier_dt=DecisionTreeClassifier(criterion='entropy')
classifier_dt.fit(X_train,Y_train)
print("Score of Decision Tree: {}".format(classifier_dt.score(X_train,Y_train)*100))
y_pred_dt=classifier_dt.predict(X_test)
acc_dt=accuracy_score(Y_test,y_pred_dt)*100
print("Accuracy of Decision Tree Classifier: {}%".format(acc_dt))
print("Confusion Matrix of Decision Tree Classifier: \n{}".format(confusion_matrix(Y_test,y_pred_dt)))
print(classification_report(Y_test,y_pred_dt))

#Random Forest
classifier_rf=RandomForestClassifier(n_estimators=500)
classifier_rf.fit(X_train,Y_train)
print("Score of Random Forest: {}".format(classifier_rf.score(X_train,Y_train)*100))
y_pred_rf=classifier_rf.predict(X_test)
acc_rf=accuracy_score(Y_test,y_pred_rf)*100
print("Accuracy of Random Forest Classifier: {}%".format(acc_rf))
print("Confusion Matrix of Random Forest Classifier: \n{}".format(confusion_matrix(Y_test,y_pred_rf)))
print(classification_report(Y_test,y_pred_rf))

#Visualizing Accuracy
classifiers_names=['Logistic Regression','Naive Bayes','SVM','Decision Tree','Random Forest']
acc_list=[acc_log,acc_nb,acc_svm,acc_dt,acc_rf]
result=pd.DataFrame({'Model':classifiers_names,'Accuracy':acc_list})
plt.figure(figsize=(10,5))
sns.barplot(data=result,x='Model',y='Accuracy')

#Cross Validation
k_fold=KFold(n_splits=10)
result_mean=[]
accuracy=[]
classifiers=['Logistic Regression','Naive Bayes','SVM','Decision Tree','Random forest']
models=[LogisticRegression(),GaussianNB(),SVC(kernel='rbf'),DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=500,random_state=0)]

for model in models:
    cv_result=cross_val_score(model,X_train,Y_train,cv=k_fold,scoring="accuracy")
    result_mean.append(cv_result.mean())
    accuracy.append(cv_result)

cv_dataframe=pd.DataFrame(result_mean,index=classifiers)   
cv_dataframe.columns=['cv_Mean']
cv_dataframe.sort_values(['cv_Mean'],ascending=[0])

#Visualising Cross Validation accuracies
cv_dataframe1=pd.DataFrame(accuracy,index=[classifiers])
boxT = cv_dataframe1.T
plt.figure(figsize=(10,10))
ax=sns.boxplot(data=boxT)
ax.set_title('Cross validation accuracies of different classifiers')
ax.set_ylabel('Accuracy')
plt.show()

#Testing new data
test_data=pd.read_csv('./test.csv')
X_data=test_data.iloc[:,1:].values
X_data=sc.transform(X_data)
y_test_pred_log=classifier_log.predict(X_data)
y_test_pred_log=pd.DataFrame(y_test_pred_log)
plt.figure()
plt.hist(y_test_pred_log)
plt.title('Predicted Price Range for new data')
plt.xlabel('Price Range')
plt.ylabel('Count')

