#Decision Tree
#Ashish Kumar Choubey
#Classification and Regression Tree Algorithm uses gini impurity index to measure quality of split
#ID3 Decision Tree Algorithm uses Entropye Function and Information Gain as metrics to measure the quality of split
#Categorical dataset

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("tennis.csv", dtype={'outlook':'category', 'temp':'category', 'humidity':'category', 'windy':'category', 'play' : 'category'})

X_train, X_test, y_train, y_test, = train_test_split(dataset.iloc[:,:-1], dataset.iloc[:,-1], test_size=0.2)


#Returns the Gini Index for all groups obtained upon split on a particular feature
#Just for Learning's sake. NOT USED IN MAIN PROGRAM
def gini_index_probabs_only(dataset, feature):
    if feature not in dataset:
        raise ValueError('Given Feature not present in dataset')
    probab = []
    print(dataset)
    target_categories = list(dataset.iloc[:,-1].cat.categories)
    for category in list(dataset[feature].cat.categories):
        print(f"Working for category : {category}")
        sample = dataset.loc[dataset[feature] == category]
        count = []
        for target in target_categories:
            print(f"Working for target : {target}")
            count.append(len(sample[sample.iloc[:,-1] == target]))
        total = sum(count)
        probab.append(1 - sum(list(map(lambda x : (x/total)**2, count))))
    return probab

#Returns Gini Impurity for the split on the given feature
#It is the weighted average of the gini indices of all groups obtained upon split
def gini_index(dataset, feature):
    if feature not in dataset:
        raise ValueError('Given Feature not present in dataset')
    probab_and_count = []
    print(dataset)
    target_categories = list(dataset.iloc[:,-1].cat.categories)
    for category in list(dataset[feature].cat.categories):
        print(f"Working for category : {category}")
        sample = dataset.loc[dataset[feature] == category]
        count = []
        for target in target_categories:
            print(f"Working for target : {target}")
            count.append(len(sample[sample.iloc[:,-1] == target]))
        total = sum(count)
        probab_and_count.append((1 - sum(list(map(lambda x : (x/total)**2, count))), count ))
    print(probab_and_count)

    #Calculating weighted average:
    return sum(map(lambda x : (x[0]*x[1])/len(dataset.feature), ))

#print(gini_index_probabs_only(dataset, 'windy'))
print(gini_index(dataset, 'windy'))

