from sklearn import tree

features = [[2, 10], [6, 8], [4, 15], [20, 30], [16, 18], [30, 50]]
labels = [0, 0, 0, 1, 1, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

if(clf.predict([[5, 12]]) == 0):
    print("è un gatto")
else:
    print("è un cane")
