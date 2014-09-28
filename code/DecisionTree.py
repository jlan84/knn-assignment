import pandas as pd
import numpy as np
import math
from collections import Counter
from TreeNode import TreeNode

class DecisionTree(object):
    '''
    A decision tree class.
    '''
    def __init__(self):
        '''
        Initialize an empty DecisionTree.
        '''

        self.root = None
        self.feature_names = None

    def fit(self, X, y, feature_names=None):
        '''
        INPUT: DECISIONTREE, 2 DIMENSIONAL NUMPY ARRAY, NUMPY ARRAY, NUMPY ARRAY
        OUTPUT: None

        Build the decision tree.
        X is a 2 dimensional array with each column being a feature and each
        row a data point.
        y is a 1 dimensional array with each value being the corresponding 
        feature_names is an optional list containing the names of each of the
        features.
        '''

        if not feature_names or len(feature_names) != X.shape[0]:
            self.feature_names = np.arange(X.shape[0])
        else:
            self.feature_names = feature_names
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y):
        '''
        INPUT: DECISIONTREE, 2 DIMENSIONAL NUMPY ARRAY
        OUTPUT: TREENODE

        Recursively build the decision tree. Return the root node.
        '''
        
        pass

    def __str__(self):
        return self.root.as_string()
