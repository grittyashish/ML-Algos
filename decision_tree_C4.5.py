#C4.5 Decision Tree Algorithm
#By Ashish Kumar Choubey
#A modification of ID3 algorithm with additional normalization
#Information Gain is biased towards attributes with larger set of values/categories : http://www.inf.unibz.it/dis/teaching/DWDM/slides2011/lesson5-Classification-2.pdf
#Normalization is : Split Info
#Heuristic : Gain Ratio
#Format of training is just like scikit-learn :  X, y
#Not highly efficient as no pre-pruning or post-pruning taken into account

import pandas as pd
import numpy as np
import math
from collections import Counter
import time
import random

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#Normalization Factor for a particular column
#X is the potential split column
def split_info(y):
    if isinstance(y, pd.Series):
        split_info = 0
        for _, count in y.value_counts().items():
            if count > 0:
                split_info += -1*(count/len(y))*math.log2(count/len(y))
        return split_info
    else:
        raise ValueError("Wrong input type for split_info")

#Calculates the information gain if dataset is split on the given feature
#X is the potential split column and y is the system
def information_gain(X, y, feature):
    #Calculating Entropy of the entire system
    #If feature is categorical
    if X[feature].dtype.name is 'category':
        count = dict(y.counts())
        system_total = len(y)
        system_entropy = 0
        for key, value in count.items():
            try:
                system_entropy = system_entropy + (value/system_total)*math.log2(value/system_total)
            except ValueError :
                pass

        system_entropy *= -1
        entropy = list()
        #Calculating Information Gain
        for category in list(X[feature].cat.categories):
            observation = y.iloc[X.loc[X[feature] == category].index.values]
            count = dict(observation.counts())
            total = len(observation)

            #Calculating Entropy
            ent = 0
            for key,value in count.items():
                if value != 0:
                    ent += -(value/total)*math.log2(value/total)
            #It stores the entropy and the #samples for which this particular category is present
            entropy.append((ent, total))
        summation = 0
        for ents, samples in entropy:
            summation += ents*(samples/system_total)

        #Returning information gain
        return system_entropy - summation
    #If feature is not categorical, there will only be binary split
    else:
        sorted_values = sorted(X[feature].unique())[:-1]#Excludes the last value as there can be no > split for it
        count = dict(y.counts())
        system_total = len(y)
        system_entropy = 0
        for key, value in count.items():
            try:
                system_entropy = system_entropy + (value/system_total)*math.log2(value/system_total)
            except ValueError :
                pass

        system_entropy *= -1

        #Calculating Information Gain for continuous feature
        max_gain = 0
        max_summation = system_entropy
        best_threshold = None
        for threshold in sorted_values:
            less_than_equal_to_observation = y.iloc[X.loc[X[feature] <= threshold].index.values]
            more_than_observation = y.iloc[X.loc[X[feature] > threshold].index.values]
            less_than_equal_to_count = dict(less_than_equal_to_observation.counts())
            more_than_count = dict(more_than_observation.counts())
            less_than_equal_to_total = len(less_than_equal_to_observation)
            more_than_total = len(more_than_observation)
            ent = 0
            entropy = list()
            #Calculating for less than equal to threshold split
            for key,value in less_than_equal_to_count.items():
                if value != 0:
                    ent += -(value/less_than_equal_to_total)*math.log2(value/less_than_equal_to_total)
            #It stores the entropy and the #samples for which this particular category is present
            entropy.append((ent, less_than_equal_to_total))
            ent = 0
            #Calculating for more than threshold split
            for key, value in more_than_count.items():
                try:
                    ent += -(value/more_than_total)*math.log2(value/more_than_total)
                except ValueError:
                    pass
            entropy.append((ent,more_than_total))
            summation = 0
            for ents, samples in entropy:
                summation += ents*(samples/system_total)
            if system_entropy - summation > system_entropy - max_summation:
                max_gain = system_entropy - summation
                max_summation = summation
                best_threshold = threshold

        #Returning information gain for continuous feature
        return (best_threshold, max_gain)


class Node:
    nodeCount = 0
    def __init__(self, X, y):
        Node.nodeCount += 1
        self.X, self.y = X.reset_index(drop=True),y.reset_index(drop=True)
        #Use majority label
        self.label = Counter(self.y.values).most_common(1)[0][0]
        self.splitted = False
        self.children = list(tuple())
        #splitting_feature can be a tuple of (feature_name, threshold) for continuous attribute or a float for categorical attribute
        self.splitting_feature = self.get_best_feature(self.X, self.y)
        #When Gain Ratio is 0 or pure split
        if self.splitting_feature == None:
            self.splitted = None
            self.label = Counter(self.y.values).most_common(1)[0][0]

    def get_best_feature(self, X, y):
        #If pure set => can't split
        if len(Counter(y.values)) == 1:
            return None
        max_ = 0
        split_info_ = split_info(y)
        for feature in X.columns:
            information_gain_ = information_gain(X, y, feature)
            if not isinstance(information_gain_, tuple):
                gain_ratio = information_gain_/split_info_
                if gain_ratio > max_:
                    max_ = gain_ratio
                    best_feature = feature

            #If continuous feature=> information_gain_ is of type (best_threshold, information_gain)
            else:
                gain_ratio = information_gain_[1]/split_info_
                if gain_ratio > max_:
                    max_ = gain_ratio
                    best_feature = (feature, information_gain_[0])
        if max_ == 0:
            return None
        return best_feature


    def get_dataset(self):
        return (self.X, self.y)

    #Here category can be name of category or a tuple of (threshold, 'greater') or (threshold, 'less) for continuous values
    def add_child(self,node, category):
        self.children.append((node, category))

    def __str__(self):
        return f"Dataset for this node is : \n {self.X} with labels as  : \n{self.y} \n label is :  {self.label} and it splits on : {self.splitting_feature}"

#Tree of depth = 1 means single node tree
class DecisionTree:
    def __init__(self, root, max_depth):
        self.root = root
        self.max_depth = max_depth

    def build_tree(self):
        for i in range(self.max_depth-1):
            self.build_tree_helper(self.root)

    #Builds tree in Breadth First or Level Order fashion, returning after creation of each level
    def build_tree_helper(self, node):
        if node is None:
            return
        queue = [node]
        while(len(queue) > 0):
            if queue[0].splitted == True:
                for child, category in queue[0].children:
                    queue.append(child)
            elif queue[0].splitted is not None:
                self.split_node(queue[0])
            del queue[0]

    #Splits the node and creates its children and returns them
    def split_node(self, node):
        #If node is pure
        if node.splitted is None:
            return
        best_feature = node.splitting_feature
        X, y = node.get_dataset()
        #Handling for categorical feature
        if not isinstance(best_feature, tuple):
            for category, count in dict(X[best_feature].counts()).items():
                #If this category is present in the subset
                if count > 0:
                    subset_X = X.loc[X[best_feature] == category]
                    node.add_child(Node(subset_X, y.iloc[subset_X.index.values]), category)
            node.splitted = True
            return
        #Handling for continuous feature
        #Requires addition of two children only
        feature, threshold = best_feature[0], best_feature[1]
        X_less_than_equal_to = X.loc[X[feature] <= threshold]
        X_greater_than = X.loc[X[feature] > threshold]
        node.add_child(Node(X_less_than_equal_to, y.iloc[X_less_than_equal_to.index.values]), (threshold, 'less_equal'))
        node.add_child(Node(X_greater_than, y.iloc[X_greater_than.index.values]), (threshold, 'greater'))
        node.splitted = True
        return


    #If only 1 node => height = 1
    def height(self):
        return self.height_helper(self.root)

    def height_helper(self, node):
        if node is None:
            return 0
        else:
            height_list = list([0])
            for child, category in node.children:
                height_list.append(self.height_helper(child))
            return max(height_list)+1

    #Testing data contains multiple samples
    def predict(self, X_test):
        result_set = list()
        for index, sample in X_test.iterrows():
            #Start checking from depth = 1 => root node
            result_set.append(self.predict_helper(self.root, sample))
        return result_set

    #Traverse the tree to find the appropriate "end node"
    def predict_helper(self, node, sample):
        #Leaf Node encountered and it is pure
        if node.splitted is None:
            return node.label

        #If Leaf Node encountered but it isn't pure
        if len(node.children) is 0 :
            return node.label

        for child, split_category in node.children:
            #Find the branch relevant to the sample
            #print(f"Split category is : {split_category}")
            if not isinstance(split_category, tuple):
                #print(node.splitting_feature)
                if split_category == sample[node.splitting_feature]:
                    return self.predict_helper(child, sample)
            #If continuous feature=>node.splitting_feature has to be a tuple of (feature_name, threshold)
            #Also, in this case split_category : (threshold, 'greater') or (threshold, 'less_than')
            elif isinstance(split_category, tuple):
                if sample[node.splitting_feature[0]] <= node.splitting_feature[1] and split_category[1] == 'less_equal':
                    return self.predict_helper(child, sample)
                elif sample[node.splitting_feature[0]] > node.splitting_feature[1] and split_category[1] == 'greater':
                    return self.predict_helper(child, sample)
        print("HIT")

    #Returns accuracy of prediction
    def get_accuracy(self, X_test, y_test):
        result = self.predict(X_test)
        correct_count = 0
        for hypothesis, actual in zip(result, y_test):
            if hypothesis == actual:
                correct_count += 1
        return correct_count/len(y_test)

    #Print tree in Level Order or Breadth First Manner
    def print_tree(self):
        queue = [self.root]
        while len(queue) > 0:
            print(queue[0])
            print(Counter(queue[0].y.values))
            for child, split_category in queue[0].children:
                queue.append(child)
            del queue[0]


data = pd.read_csv('titanic.csv', dtype={'Survived':'category','Pclass':'category', 'Sex':'category', 'Age':'float64', 'Fare':'float64', 'Cabin':'category', 'Embarked':'category'})
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

#Data Preparation
female_ages = data.loc[data['Sex'] == 'female'].Age.dropna()
male_ages = data.loc[data['Sex'] == 'male'].Age.dropna()

for index, row in data.iterrows():
    if pd.isnull(row['Age']) and row['Sex'] == 'female':
        data.at[index,'Age'] = np.random.uniform(female_ages.median() - female_ages.std(), \
                                                    female_ages.median() + female_ages.std())
    if pd.isnull(row['Age']) and row['Sex'] == 'male':
        data.at[index,'Age'] = np.random.uniform(male_ages.median() - male_ages.std(), \
                                                    male_ages.median() + male_ages.std())

data.drop(data[data['Embarked'].isna()].index, inplace=True)

data['Family'] = data['SibSp'] + data['Parch']
data.drop(['SibSp', 'Parch'], axis=1, inplace=True)

cols = data.columns.tolist()[1:] + [data.columns.tolist()[0]]
data = data[cols]

def train_test_split(X, y, test_size = 0.20):
    if isinstance(test_size, float):
        test_size = round(test_size*len(X))
    indices = X.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)
    return X.drop(test_indices), X.loc[test_indices], y.drop(test_indices), y.loc[test_indices]

X = data.iloc[:,:-1]
y = data.iloc[:,-1]
#Training and Testing
accuracy = list()
for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    C45_decision_tree = DecisionTree(Node(X_train, y_train), max_depth=10)
    C45_decision_tree.build_tree()
    print(f"Test size : {len(X_test)}")
    accuracy.append(C45_decision_tree.get_accuracy(X_test, y_test))
print(sum(accuracy)/len(accuracy))
