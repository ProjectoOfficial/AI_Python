from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .5)

classifier= KNeighborsClassifier()
classifier.fit(X_train, y_train)

predictions =classifier.predict(X_test)

print( accuracy_score(y_test,predictions))