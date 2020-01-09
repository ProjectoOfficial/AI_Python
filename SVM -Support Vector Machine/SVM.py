import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

c = datasets.load_breast_cancer()

x = c.data
y = c.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classi = ['malignant', 'benign']

classificatore = svm.SVC(kernel="linear")
classificatore.fit(x_train, y_train)

y_predict = classificatore.predict(x_test)

accuratezza = metrics.accuracy_score(y_test, y_predict)

print(accuratezza)
