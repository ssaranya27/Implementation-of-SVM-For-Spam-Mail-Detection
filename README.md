# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Load and Preprocess Data**: Load the spam dataset, label "spam" as 1 and "ham" as 0, and clean the text data.

2. **Split Data**: Divide the dataset into training and testing sets.

3. **Vectorize Text**: Convert text messages to numerical data using `CountVectorizer`.

4. **Train SVM Model**: Train an SVM model with a linear kernel on the training set.

5. **Evaluate Model**: Test the model on the test set and evaluate using accuracy, precision, recall, and a confusion matrix.
## Program:
```
import chardet
file='spam.csv'
with open(file,'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')
data.head()
data.tail()
data.info()
data.isnull().sum()
x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
# y_train.shape
from sklearn.feature_extraction.text import CountVectorizer
#countvetorizer is a methon to convert text to numerical data.the text is transformed to a  sparse matrix
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
x_test.shape
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:

PREDICTED VALUE:
![image](https://github.com/user-attachments/assets/423baeda-b82d-47cc-b60c-b9d46848b627)

ACCURACY:
![image](https://github.com/user-attachments/assets/3ae74333-b846-42e8-aca5-89139112b960)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
