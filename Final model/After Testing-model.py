# Artificial Neural Network

# Part 1 - Data Preprocessing

#1 Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#2 Importing the dataset
HR_dataset = pd.read_excel("Vee_original_data_v4.xlsx")
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y_1 = LabelEncoder() #encode categorical Y to numerical Y
HR_dataset["Attrition"] = labelencoder_y_1.fit_transform(HR_dataset["Attrition"])

#HR_dataset.info()
#HR_dataset.size
#HR_dataset_object = HR_dataset.select_dtypes(include=['object'])
#HR_dataset_object.dtypes

X = HR_dataset.iloc[:, 2:21]
y = HR_dataset.iloc[:, 21]

#3 Encoding categorical data
columns = ['BusinessTravel','Department','EducationField','Gender','MaritalStatus']

X = pd.get_dummies(X, columns=columns, drop_first=True)

#5 Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Part 2 - Now let's make the Classification - ANN!

#1 Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#2 Initialising the ANN
classifier = Sequential()

# Adding the I/P layer and the 1st hidden layer
classifier.add(Dense(units = 19, kernel_initializer = 'uniform', activation = 'relu', input_dim = 38))

# Adding the 2nd hidden layer
classifier.add(Dense(units = 19, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the O/P layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X, y, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

#1 Predicting the Test set results
y_pred = classifier.evaluate(X,y)

print("%s: %.2f%%" % (classifier.metrics_names[1], y_pred[1]*100))

classifier.save("model.h5")
print("Saved model to disk")

'''https://machinelearningmastery.com/save-load-keras-deep-learning-models/'''
