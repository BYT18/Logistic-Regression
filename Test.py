
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

from sklearn import preprocessing 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

##dataframe = pd.read_csv('CSV monthly average indicator.csv', header = 1)
col_names = ['VALUE (30 TERM)', 'VALUE (90 TERM)','IA VOLUME', 'IA Volume Difference', 'Indicator']

dataframe = pd.read_csv('test2.csv', header = 0, names = col_names)
dataframe.head()

#dataset = dataframe.values

#dataframe = dataframe.dropna()
print (dataframe.shape)
#print(list(dataframe.columns))

print(dataframe.head())
print (dataframe['Indicator'].value_counts())

feature_cols = ['VALUE (30 TERM)', 'VALUE (90 TERM)' ]
#feature_cols = [dataframe.iloc[0, 0:49]]
print(feature_cols)
X = dataframe[feature_cols] # Features
y = dataframe.Indicator 
##print (y)

#feature_cols = [list(dataframe.iloc[:, 0:49])]
#feature_cols = [dataframe.iloc[0, 0:49]]
#print(feature_cols)
#X = dataframe[list(dataframe.columns)] # Features
#X = list(dataframe.iloc[:, 0:49]) # Features
#y = dataframe.Indicator 
#y = dataframe['Indicator '] 

#train, test = train_test_split(dataframe, test_size=0.2)
#train, val = train_test_split(train, test_size=0.2)
#print(len(train), 'train examples')
#print(len(val), 'validation examples')
#print(len(test), 'test examples')


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#feature_cols = [list(dataframe.columns(1,20))]


logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))

#(x_train, y_train), (x_test, y_test) = train_test_split(dataframe['Indicator '],test_size=0.2)

#print (dataframe['Indicator '].value_counts())

##(x_train, y_train), (x_test, y_test) = dataset
##x_train.shape
##y_train.shape

#Dummy variables
#cat_vars=[list(dataframe.columns)]
#data_final=dataframe
#data_final.columns.values


#SMOTE in python
#X = data_final.loc[:, data_final.columns != 'Indicator ']
#y = data_final.loc[:, data_final.columns == 'Indicator ']
#from imblearn.over_sampling import SMOTE
#os = SMOTE(random_state=0)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#columns = X_train.columns
#os_data_X,os_data_y=os.fit_sample(X_train, y_train)
#os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
#os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
#print("length of oversampled data is ",len(os_data_X))
#print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
#print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
#print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
#print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))

#Recursive feature elimination 


