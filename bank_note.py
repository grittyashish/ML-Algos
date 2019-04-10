from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

df = pd.read_csv("data_banknote_authentication.txt", header=None)
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

grid_parameters = {
    'criterion' : ['entropy', 'gini'],
    'splitter' : ['best', 'random'],
    'max_depth' : [2,3,4,5,6,7,8,9,10],
    'max_features' : [1,2,3,4]
}
grid_search = GridSearchCV(cv=10, estimator=DecisionTreeClassifier(), param_grid=grid_parameters, scoring='accuracy')
grid_search.fit(X,y)
print(f"The best parametes : {grid_search.best_params_} and best accuracy : {grid_search.best_score_}")
print(f"The best estimator : {grid_search.best_estimator_}")

