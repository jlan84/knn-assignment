from collections import Counter

class TreeNode(object):
    '''
    A node class for a decision tree.
    '''
    def __init__(self):
        self.column = None  # (int)    index of feature to split on
        self.name = None    # (string) name of feature (or name of class in the
                            #          case of a list)
        self.children = {}  # (dict)   value of feature as key and child node
                            #          as key
        self.leaf = False   # (bool)   true if node is a leaf, false otherwise
        self.classes = Counter()  # (Counter) only necessary for leaf node:
                                  #           key is class name and value is
                                  #           count of the count of data points
                                  #           that terminate at this leaf

    def as_string(self, level=0, prefix=""):
        '''
        INPUT: TREENODE, INT, STRING
        OUTPUT: STRING

        Return a string representation of the tree rooted at this node.
        '''
        result = ""
        if prefix:
            indent = "  |   " * (level - 1) + "  |-> "
            result += indent + prefix + "\n"
        indent = "  |   " * level
        result += indent + "  " + str(self.name) + "\n"
        for key, node in self.children.iteritems():
            result += node.as_string(level + 1, str(key) + ":")
        return result
