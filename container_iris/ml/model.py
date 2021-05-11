import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import joblib

# *** Подготовка датасета ***
# iris = datasets.load_iris()
# iris_frame = pd.DataFrame(iris.data)
# iris_frame.columns = iris.feature_names
# iris_frame['target'] = iris.target
# iris_name = iris.target_names
# # print(iris_frame.head(10))
# iris_frame.to_csv('../dataset/iris_frame.csv', index=False)

# *** Создание модели ***
# df = pd.read_csv('../dataset/iris_frame.csv')
# X = df.iloc[:, df.columns != 'target']
# y = df['target']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
# model = KNeighborsClassifier()
# model.fit(X_train, y_train)
# expected = y_test
# predicted = model.predict(X_test)
# accuracy = accuracy_score(expected, predicted)
# # print("Accuracy: %.2f%%" % (accuracy * 100.0))
# joblib.dump(model, 'kneighborsclassifier_model.pkl')

# *** Тест модели ***
# pred_args = [5.9, 3.0, 5.1, 1.8]
# preds = np.array(pred_args).reshape(1, -1)
# # print(preds)
# model_open = open('kneighborsclassifier_model.pkl', 'rb')
# kneighborsclassifier_model = joblib.load(model_open)
# model_prediction = kneighborsclassifier_model.predict(preds)
# print(model_prediction[0])
