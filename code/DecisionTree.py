import pandas as pd
import numpy as np
import math
from collections import Counter
from TreeNode import TreeNode

class DecisionTree(object):
    '''
    A decision tree class.
    '''
    def __init__(self, data, features, target_index):
        '''
        INPUT: 2 DIMENSIONAL NUMPY ARRAY, LIST, INTEGER
        OUTPUT: NONE

        Build the decision tree.
        data is a 2 dimensional array with each column being a feature and each
        row a data point.
        features is a list containing the names of each of the features.
        target_index is the index of the feature we are predicting.
        '''
        self.target_index = target_index   # index of the column to predict
        self.features = features           # list of feature names (only needed
                                           # for visualizing and understanding
                                           # the tree, not for prediction)
        self.root = self.build_tree(data)

    def build_tree(self, data):
        '''
        INPUT: DECISIONTREE, 2 DIMENSIONAL NUMPY ARRAY
        OUTPUT: TREENODE

        Recursively build the decision tree. Return the root node.
        '''
        pass

    def __str__(self):
        return self.root.as_string()
