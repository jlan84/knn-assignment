## Decision Trees

Decision trees are a recursive divide and conquer algorithm. They are a non-linear, non-parametric discriminative supervised classification algorithm.  There are a few names of decision tree algorithms you may have heard of (ID3, C4.5, CART, etc.) and each is a different specification of a decision tree model.  You can read about them [here](http://stackoverflow.com/questions/9979461/different-decision-tree-algorithms-with-comparison-of-complexity-or-performance) and [here](http://scikit-learn.org/stable/modules/tree.html#tree-algorithms-id3-c4-5-c5-0-and-cart).

### Play Golf Dataset

When implementing any ML algorithm for the first time, it is often easier to start with a trivially simple data set. You should always focus on one portion of the pipeline at a time: we do not want worry about cleaning data during feature selection just as we do not want to worry about feature engineering when writing our model building code.  We will be using the canonical 'Play Golf' [dataset](http://www2.cs.uregina.ca/~dbd/cs831/notes/ml/dtrees/c4.5/c4.5_prob1.html) when writing our algorithm.

Look at the [golf data](data/playgolf.csv). You will also see a dataset with just the categorial features and one with just the continuous features. Starting with just categorical features may be easier for implementation.

### Pseudo-code

Here's the pseudocode for the algorithm you will be implementing.

    function BuildTree:
        If every item in the dataset is in the same class
        or there is no feature left to split the data:
            return a leaf node with the class label
        Else:
            find the best feature and value to split the data 
            split the dataset
            create a node
            for each split
                call BuildTree and add the result as a child of the node
            return node

### Implementation

You've been given starter code in the [code](code) folder. Some of the instance variables chosen are not the only possible way of implementing a decision tree, so feel free to modify anything if it fits your implementation better.

* The `TreeNode` class is implemented. These are the instance variables:

    * `column` (int): index of feature to split on
    * `split_value` (object): value of the feature to split on
    * `categorical` (bool): whether or not node is split on a categorial feature (vs continuous)
    * `name` (string): name of the feature (or name of the class in the case of a list)
    * `left` (TreeNode): left child
    * `right` (Tree Node): right child
    * `leaf` (boolean): true or false depending on if the node is a leaf node.    
    * `classes` (Counter): if a leaf, a count of all the list of all the classes of the data points that terminate at this leaf.  Can be used to assess how "accurate" an individual leaf is.

    The `as_string` and `__str__` functions are designed to help you be able to print out decision tree (mostly for debugging).

* There is minimal starter code for the `DecisionTree` class. You will need to fill in the class so that you can use your decision tree code as follows, assuming `data` has been initalized to 2 dimensional numpy array containing the play golf dataset. In this example, `data` has 5 columns and 19 rows. The last row (index 4) is the result we are trying to classify.

    ```python
    tree = DecisionTree()
    tree.fit(X, y, df.columns[:-1])
    print tree
    y_predict = tree.predict(X)
    ```

    You can see that the `__str__` method is implemented for you. This enables you to print your tree for debugging purposes.

### Steps to Implementing

We will be implementing the **CART** algorithm. This means that every split will be binary. For categorical features, splits will be like: `sunny` or `not sunny`. For continuous features, splits will be like: `>80` or `<=80`.

Feel free to start by restricting yourself to categorical features to make things a little simpler.

1. Implement an `entropy` function, which is given by the following equation. Entropy measures the amount of "disorder" in a set. Here there are *m* elements in the set and *ci* is the class of the *i*-th element.

    ![shannon entropy](images/entropy.png)

    *P(c)* = (count of occurrences of class *c*) / size of *y*

    Note that to calculate entropy, you only need to labels (`y` values) and none of the feature values.

2. To write the `_build_tree` method which recursively builds the tree, you will probably want the following methods:
    * `information_gain`: Given a binary split of the dataset, returns the information gain based on this formula:

        ![information gain](images/gain.png)

        *D* is the set of sets which make up *S* based on our split. In our case, since we're only doing binary splits, the information gain is as follows.
        ![binary information gain](images/binary_gain.png)
    * `choose_split`: Determine the best feature and value to split the dataset on.
    * `make_split`: Given feature and value, return the two subsets that are created from splitting on that feature and value (note that this works differently depending if the feature is continuous or categorical).

    We've intentionally left design decisions to you. However, if you're having trouble figuring out how exactly to structure things, ask your neighbor or call someone over. And of course you are welcome to modify any of the instance variables already created, these are there to give you an idea.

3. Implement a `predict` method for the `DecisionTree` class which takes a new feature matrix `X` and predicts its class based on the decision tree. It may be helpful to have a recursive `predict` method in the `TreeNode` class as well. And try doing just a single data point first.


### Plot Decision Boundaries

`sklearn` has an example of how to plot decision boundaries for non-parametric learners [here](http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html#example-tree-plot-iris-py).

1. Plot the decision boundaries created by your decision tree for the play golf dataset, the iris dataset (the one used in the example), yesterday's dataset or any made up dataset. In order to plot it, you should have exactly two continuous features.


### Decision Trees for Regression

**Note:** Before starting this, make sure you commit your code with a `git commit`! Don't lose your past results with your new changes!

You can use decision trees for predicting continuous values as well. Instead of using entropy to calculate the disorder in the set, we use the variance.

To get to value of a leaf node, average all of the values.

1. Make your decision tree able to predict continuous values. You can modify your decision tree class so that it can do either continuous or categorical depending on what parameters you pass it, or just copy and create a new class. For checking out if your code is implemented correctly, you can use the same dataset and predict one of the continuous variables.


### A Real Dataset

1. Try running your decision tree code on yesterday's dataset.

2. Use sklearn's [Decision Tree](http://scikit-learn.org/stable/modules/tree.html#classification) and [k Nearest Neighbors](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) classifiers on the same dataset. How well do they do compared to logistic regression?


### Extra Credit

You can do these in any order. Prepruning and decision boundaries are probably the most important.

*Pruning* is designed to simplify the tree so it doesn't go so deep. It is a way of stopping earlier or merging leaves that helps deal with overfitting. The first two extra credit problems are implementing prepruning and postpruning. A well designed decision tree would have these implemented.

1. *Prepruning* is making the decision tree algorithm stop early. Here are a few ways that we preprune:
    * leaf size: Stop when the number of data points for a leaf gets below a threshold
    * depth: Stop when the depth of the tree (distance from root to leaf) reaches a threshold
    * mostly the same: Stop when some percent of the data points are the same (rather than all the same)
    * error threshold: Stop when the error reduction (information gain) isn't improved significantly.
    
    Implement some of the prepruning thresholds and play around with using them.

2. Implement *postpruning* for your decision tree. You build the tree the same as before, but after you've built the tree, merge some nodes together if doing so reduces the error. Here's the psuedocode:

        function Prune:
            if either left or right is not a leaf:
                call Prune on that split
            if both left and right are leaf nodes:
                calculate error associated with merging two nodes
                calculate error associated without merging two nodes
                if merging results in lower error:
                    merge the leaf nodes

    You can find more detail in section 9.4.2 in Machine Learning in Action.

3. Use the Gini impurity instead of entropy to choose the best split.

4. Implement model trees, which are predictors which start by using a decision tree, but use linear regression to predict the value on each leaf node. Details can be found in 9.5 of Machine Learning in Action.
