#Problem : We need to determine by asking the question that will user of specific age and earning some amount of salary specified in the record,
#purchase a laptop or not. We implement a Logical Regression Model and use the independent variables to predict the result.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv("purchases.csv")
#print(data.head(5))

#plot the relationship between age and buying laptop
plt.scatter(data['age'].values,data['buys_laptop'].values, label="AGE VS PURCHASE",color="red")
plt.show()

#Selecting independent and dependent variables from the dataset

X = data.drop("buys_laptop",axis=1)
Y = data["buys_laptop"]


#Splitting data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1)


#Final initialization of the model and training it

model = LogisticRegression()
model.fit(X_train,Y_train)

#Predict the output of the testing data
print(model.predict(X_test))    #It returns an array or a list that contains the value predicted for Y for the values of X_test, [1] means
                                #user will buy the laptop and 0 means he will not buy the laptop

#Predicting for user inputted values

age = int(input("Enter the value for the age you want to predict:"))
salary = int(input("Enter the salary:"))

X_pred = [[age,salary]]
result = model.predict(X_pred)

if result == 1:
    print("User will buy a laptop!")
else:
    print("User will not buy a laptop")

#Displaying the statistics for the model

print("The score for the model is:",accuracy_score(Y_test,model.predict(X_test)))
print(classification_report(Y_test,model.predict(X_test)))
print(confusion_matrix(Y_test,model.predict(X_test)))

