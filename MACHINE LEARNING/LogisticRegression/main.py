import numpy as np
from sklearn import datasets, model_selection
from random import random
from LogisticRegression import MyLogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

MULT = 1000

X, Y = datasets.load_breast_cancer(return_X_y=True)

# normalization
scaler = MinMaxScaler().fit(X)
X = scaler.transform(X)

lab, count = np.unique(Y, return_counts=True)
print("labels: {} - {}".format(lab, count))


if __name__ == '__main__':
    best_accuracy = 0
    best_seed = 0

    for i in range(50):
        # shuffle
        np.random.shuffle(X)

        x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2)

        seed = random()
        np.random.seed(int(seed * MULT))

        logistic_reg = MyLogisticRegression()

        # train
        logistic_reg.fit_gd(x_train, y_train, n_epochs=1000, learning_rate=0.05, verbose=True)

        # test
        predictions = logistic_reg.predict(x_test)

        accuracy = float(np.sum(predictions == y_test)) / y_test.shape[0] * 100
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            best_seed = np.round(seed * MULT)
        print('Test {} accuracy: {:.2f}% with seed:{:.2f}'.format(i, accuracy, seed * MULT))

    print("Best accuracy: {:.2f}%  with seed: {:.2f}". format(best_accuracy, best_seed))

    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2)
    clf = LogisticRegression(random_state=0).fit(x_train, y_train)
    clf.predict(x_test)
    clf.predict_proba(x_test)
    print("Sklearn Logistic Regression Score: {:.2f}%".format(clf.score(x_test, y_test) * 100))
