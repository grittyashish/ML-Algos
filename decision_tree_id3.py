#Iterative Dichotomer 3 - Decision Tree by Ross Quinlan
#By Ashish Kumar Choubey
#Heuristics: Information Gain
#Works with categorical data only

import pandas as pd
import numpy as np
import math
from collections import Counter
import time
import random

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


#Calculates the information gain if dataset is split on the given feature
#X is the potential split column and y is the system
def information_gain(X, y, feature):
    #Calculating Entropy of the entire system
    count = dict(y.value_counts())
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
        observation = y.iloc[X.loc[X[feature] == category].index.to_numpy()]
        count = dict(observation.value_counts())
        total = len(observation)

        #Calculating Entropy
        ent = 0
        for key,value in count.items():
            #divide by zero error may occur if there is no data for some cat.categories
            try:
                if value != 0:
                    ent += -(value/total)*math.log2(value/total)
            except ValueError:
                pass
        #It stores the entropy and the #samples for which this particular category is present
        entropy.append((ent, total))
    summation = 0
    for ents, samples in entropy:
        summation += ents*(samples/system_total)

    #Returning information gain
    return system_entropy - summation

class Node:
    nodeCount = 0
    def __init__(self, X, y):
        Node.nodeCount += 1
        self.X, self.y = X.reset_index(drop=True),y.reset_index(drop=True)
        #Use majority label
        self.label = Counter(self.y.values).most_common(1)[0][0]
        self.splitted = False
        self.children = list(tuple())
        self.splitting_feature = self.get_best_feature(self.X, self.y)
        #When information gain is 0 or pure split
        if self.splitting_feature == None:
            self.splitted = None
            self.label = Counter(self.y.values).most_common(1)[0][0]

    def get_best_feature(self, X, y):
        #If pure set => can't split
        if len(Counter(y.values)) == 1:
            return None
        max_ = 0
        for feature in X.columns:
            information_gain_ = information_gain(X, y, feature)
            if information_gain_ > max_:
                max_ = information_gain_
                best_feature = feature
        #If no information gain upon split => cannot split
        if max_ == 0:
            return None
        return best_feature

    def get_dataset(self):
        return (self.X, self.y)

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
        for category, count in dict(X[best_feature].value_counts()).items():
            #If this category is present in the subset
            if count > 0:
                subset_X = X.loc[X[best_feature] == category]
                node.add_child(Node(subset_X, y.iloc[subset_X.index.to_numpy()]), category)
        node.splitted = True


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
            if split_category == sample[node.splitting_feature]:
                return self.predict_helper(child, sample)

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
            print(Counter(queue[0].subset.iloc[:,-1].values))
            for child, split_category in queue[0].children:
                queue.append(child)
            del queue[0]

def train_test_split(X, y, test_size = 0.2):
    if isinstance(test_size, float):
        test_size = round(test_size*len(X))
    indices = X.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)
    return X.drop(test_indices), X.loc[test_indices], y.drop(test_indices), y.loc[test_indices]

data = pd.read_csv("zoo.csv")
data = data.iloc[:,1:]
for column in data.columns:
    data[column] = data[column].astype('category')

X = data.iloc[:,:-1]
y = data.iloc[:,-1]
start_time = time.time()
accuracy = list()
for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    ID3_decision_tree = DecisionTree(Node(X_train, y_train), max_depth=10)
    ID3_decision_tree.build_tree()
    accuracy.append(ID3_decision_tree.get_accuracy(X_test, y_test))

end_time = time.time()
print(f"Time to train : {end_time-start_time}")
print(accuracy)
print(f"Accuracy : {sum(accuracy)/len(accuracy)}")


