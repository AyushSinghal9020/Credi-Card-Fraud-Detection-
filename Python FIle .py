

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

data = pd.read_csv('creditcard.csv')
data.shape
data.describe()

fraud = data[data['Class'] == 1]
print('Fraud cases : ', len(fraud))

valid = data[data['Class'] == 0]
print('Valid cases : ', len(valid))

outlier_fraction = len(fraud)/len(valid)
print(outlier_fraction)

print('Amount details of the fraud transaction : \n', fraud.Amount.describe())
print('Amount details of the valid transaction : \n', valid.Amount.describe())

corrmat = data.corr()
fig = plt.figure(figsize = (12 , 9))
sns.heatmap(corrmat , vmax = 0.8 , square = True)
plt.show()

X = data.drop(['Class'], axis = 1)
print(X.shape) 
xData = X.values

Y = data['Class']
print(Y.shape)
yData = Y.values

xTrain , xTest , yTrain , yTest = train_test_split(xData , yData , test_size = 0.2 , random_state = 42)

rfc = RandomForestClassifier()
rfc.fit(xTrain , yTrain)
yPred = rfc.predict(xTest)

n_outliers = len(fraud)
n_errors = (yPred != yTest).sum()

print('The model used is Random Forest CLassifier ')

print('The accuracy is : ', accuracy_score(yTest, yPred))
print('The precision is : ', precision_score(yTest , yPred))
print('The recall score is : ', recall_score(yTest , yPred))
print('The F-1 score is : ', f1_score(yTest , yPred))
print('The Matthews correlation coefficient is : ', matthews_corrcoef(yTest , yPred))

LABELS = ['Normal' , 'Fraud']
conf_matrix = confusion_matrix(yTest , yPred)

plt.figure(figsize = (12 , 12))
sns.heatmap(conf_matrix , xticklabels = LABELS , yticklabels = LABELS , annot = True , fmt = 'd')
plt.title("Confusion Matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
