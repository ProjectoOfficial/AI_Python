from mnist import MNIST
import matplotlib.pyplot as plt
import numpy as np


class NaiveBayesClassifier:

    def __init__(self, use_gaussian=0, verbose=0):
        self.GAUSSIAN = use_gaussian
        self.verbose = verbose

        self._classes = None
        self._n_classes = 0
        self._eps = np.finfo(np.float32).eps

        self._class_priors = []
        self._likelyhoods = []
        self._gaussian_likelyhoods = []

    def fit(self, X_Train, Y_Train, lr_mi=1, lr_sigma2=1):
        # prendo quante classi ci sono in totale e come sono distribuite
        self._classes, counts = np.unique(Y_Train, return_counts=True)
        if self.verbose:
            for i in range(len(self._classes)):
                print("la classe {} ha {} elementi".format(self._classes[i], counts[i]))
            print("\n")

        # calcolo la probabilità a priori per ogni classe
        self._n_classes = len(self._classes)
        self._class_priors = counts / X_Train.shape[0]
        if self.verbose:
            for i in range(self._n_classes):
                print("la classe {} ha prior={}".format(self._classes[i], self._class_priors[i]))
            print("\n")

        N = X_Train.shape[0]
        # calcolo la likelyhood per ogni classe
        for i in range(self._n_classes):
            if self.GAUSSIAN:
                # likelyhood gaussiana
                mu = np.mean(X_Train[Y_Train == i], axis=0) * lr_mi
                sigma2 = np.mean(np.square(X_Train[Y_Train == i] - mu), axis=0) * lr_sigma2
                dist = (1 / np.sqrt(2 * np.pi * sigma2)) * 1 / (np.exp(np.square(X_Train[Y_Train == i] - mu) / (2 * sigma2)))
                likely = np.sum(np.log(dist), axis=0)
                self._gaussian_likelyhoods.append(likely)
            else:
                # likelyhood normale
                print(np.mean(X_Train[Y_Train == i], axis=0))
                self._likelyhoods.append(np.mean(X_Train[Y_Train == i], axis=0))
            # non facciamo la print della likelyhood perché è una matrice 28x28

        print("\n\n")

    def predict(self, X_Test):

        N = X_Test.shape[0]
        # vettorizziamo la matrice
        X_Test = np.reshape(X_Test, (X_Test.shape[0], X_Test.shape[1] * X_Test.shape[2]))
        # i risultati saranno 10000 predizioni fatte su 10 classi
        results = np.zeros((N, self._n_classes))

        for i in range(self._n_classes):
            # prendo la i-esima likelyhood e la vettorizzo
            likelyhood = None
            if self.GAUSSIAN:
                likelyhood = self._gaussian_likelyhoods[i]
            else:
                likelyhood = self._likelyhoods[i]
            likelyhood = np.reshape(likelyhood, (likelyhood.shape[0] * likelyhood.shape[1]))

            probs = np.sum(np.log((likelyhood * X_Test) + self._eps), axis=1)
            probs += np.log(self._class_priors[i])
            results[:, i] = probs

        return np.argmax(results, axis=1)

    def print_likelyhoods(self):
        if self.GAUSSIAN:
            for i, likely in enumerate(self._gaussian_likelyhoods):
                plt.imshow(likely, cmap='gray')
                plt.show()
        else:
            for i, likely in enumerate(self._likelyhoods):
                plt.imshow(likely, cmap='gray')
                plt.show()


''' ***********************************
************ LOAD DATASET *************
*********************************** '''

mndata = MNIST('/home/daniel/PycharmProjects/AIENV/BayesNaive/MNIST')
x_trainlist, y_trainlist = mndata.load_training()
x_testlist, y_testlist = mndata.load_testing()

x_train = np.array([np.reshape(x_trainlist[i], (28, 28)) for i in range(len(x_trainlist))])
x_test = np.array([np.reshape(x_testlist[i], (28, 28)) for i in range(len(x_testlist))])

y_train = np.array(y_trainlist)
y_test = np.array(y_testlist)
label_dict = {i: i for i in np.unique(y_test)}


''' ***********************************
********** CLASSIFICAZIONE ************
*********************************** '''

classifier = NaiveBayesClassifier()
classifier.fit(x_train, y_train)

# classifier.print_likelyhoods()
predictions = classifier.predict(x_test)
vals, numbers = np.unique(predictions, return_counts=True)
for i, _ in enumerate(vals):
    print("{}  {}".format(vals[i], numbers[i]))

accuracy = np.sum(np.uint8(predictions == y_test)) / len(y_test)
print("MODEL ACCURACY: {}".format(accuracy))

idx = np.random.randint(0, x_test.shape[0])
x = x_test[idx]
p = predictions[idx]
y = y_test[idx]
plt.imshow(x, cmap='gray')
plt.title('Target: {}, Prediction: {}'.format(label_dict[int(y)], label_dict[int(p)]))
plt.show()
