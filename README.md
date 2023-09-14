# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 
1.Import the required libraries.

2.Load the dataset and check for null data values and duplicate data values in the dataframe.

3.Import label encoder from sklearn.preprocessing to encode the dataset.

4.Apply Logistic Regression on to the model.

5.Predict the y values.


6.Calculate the Accuracy,Confusion and Classsification report.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Archana.k
RegisterNumber:  212222240011
*/
import pandas as pd
df=pd.read_csv("Placement_Data(1).csv")
df.head()

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
df1

x=df1.iloc[:,:-1]
x

y=df1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
## Placement Data:
![233679600-d7637871-ac7e-4ef8-8538-cfe8f8c1ddb3](https://github.com/22009150/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708624/4f49819c-a91d-42d7-8c50-80a30c42267f)

## Salary Data:

![233679823-32ae13cc-489d-436a-925a-6187d6de27ed](https://github.com/22009150/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708624/a998d05a-e4d2-4fe2-9707-8f14605eafaf)

## Checking the null() function:

![233679969-7a2b5524-270d-4377-9728-f78188177f6c](https://github.com/22009150/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708624/0a3acb0b-ebae-40bf-b1e1-4a0cec927762)

## Data Duplicate:

![233680057-efb79829-4a73-4fab-9e37-01f58b54898b](https://github.com/22009150/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708624/12a34c2a-3a03-4fbc-a9b2-93e5f27e8aab)

## Print Data: 

![233680198-69570dd4-1cce-4363-bce2-a4cae93e236e](https://github.com/22009150/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708624/6b6e09d6-7f98-4b8e-b38b-38aeabd2de6a)

## Data-status:


![233680590-861937d3-aba8-400c-8ccf-80c25444cd69](https://github.com/22009150/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708624/230c45e7-5ca7-4903-92db-6ba2c8e3ac59)

## y_prediction array:


![233680712-229c768c-f1c1-4ec8-b43f-0b0d2996ee31](https://github.com/22009150/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708624/6966622b-87ba-473b-adcf-19a7bdfdf5b1)

## Accuracy value:

![233680788-7cbdbe90-d08b-4076-aac7-50a4ad6c26b0](https://github.com/22009150/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708624/13082ac6-19c7-4f9e-ae1d-d9727ca357e6)

## Confusion array:

![233681147-aca68fa8-33ae-48e3-b8db-bfe884d619ee](https://github.com/22009150/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708624/23b6d966-f2a0-4b1d-89c6-f86510251d10)

## Classification report:

![233681332-f1ee5ca5-9812-40b9-8d7b-c3cda844fec3](https://github.com/22009150/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708624/dfc5b74c-3c86-4d3a-bac6-d6d3549ca904)

## Prediction of LR:

![233681412-e62e2859-e43f-4515-8a18-ae7ea8bc19cb](https://github.com/22009150/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118708624/72230657-edcf-48c5-9bf0-36d55c0b5386)










## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
