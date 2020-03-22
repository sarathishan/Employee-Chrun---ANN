#1 Load the model
from keras.models import load_model
model = load_model('model.h5')
model.summary()
#2 Import Dataset
import pandas as pd
dataset = pd.read_excel("file.xlsx")
X = dataset.iloc[:, 2:21]
y = dataset.iloc[:, 21]

#3 Encoding categorical data

X = pd.get_dummies(X, sparse=True,drop_first=True)

#5 Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

score = model.evaluate(X, y)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
yhat = model.predict_classes(X)
print(yhat)