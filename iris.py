import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate

import seaborn as sns
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = iris.data[:,:]
y = iris.target
model_accuracies = []

grid_parameter = {
    'criterion' : ['gini', 'entropy'],
    'splitter' : ['best', 'random'],
    'max_depth' : [2,3,4,5,6,7,8],
    'max_features' : [1,2,3,4]
}

##### Custom search for the best accuracy
#for i in range(1000):
#    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7)
#    clf_entropy = DecisionTreeClassifier(splitter="best")
#    clf_entropy.fit(X_train, y_train)
#    model_accuracies.append(accuracy_score(clf_entropy.predict(X_test),y_test))
#sns.distplot(model_accuracies)
#plt.show()
#clf_gini = DecisionTreeClassifier(criterion="gini")

#clf_gini.fit(X_train, y_train)


#y_pred_entropy = clf_entropy.predict(X_test)
#y_pred_gini = clf_gini.predict(X_test)

#print(f"Accuracy for Gini : {accuracy_score(y_test,y_pred_gini)} and that for entropy : {accuracy_score(y_test, y_pred_entropy)}")

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=grid_parameter, cv=10, scoring='accuracy')

#export_graphviz(clf_entropy, out_file='entropy_tree_iris.dot')
#export_graphviz(clf_gini, out_file='gini_tree_iris.dot')

grid_search.fit(X,y)
print(f"The best score : {grid_search.best_score_} || The best parameters are : {grid_search.best_params_}")
print(f"The best estimator is : {grid_search.best_estimator_}")










