Main teacher: Jonathan

## Overview -- Decision trees

__ML pipeline:__ Obtain/Prepare (features) __->__ Train __->__ Validate/Test __->__ Predict 
 
In this assignment, we will using a [decision tree classifier](http://en.wikipedia.org/wiki/Decision_tree_learning) to predict.  Decision trees are one of the most popular and widely used algorithms.  Most classifiers (SVM, kNN, Neural Nets) are great at giving you a (somewhat) accurate result, but are often black boxes.  With these algorithms it can be hard to interpret their results and understand ___why___ a certain instance was assigned a label.  Decision trees are unique in that they are very flexible and accurate while also being easily interpreted.

![c4.5](http://www2.cs.uregina.ca/~dbd/cs831/notes/ml/dtrees/c4.5/golftree.gif)


__INPUTS:__ Nominal (discrete) or Continuous

__OUTPUTS:__ Nominal (discrete) or Continuous

__(basically anything in and anything out)__

## Why Decision Trees

* Easily interpretable
* Handles missing values and outliers
* [non-parametric](http://en.wikipedia.org/wiki/Non-parametric_statistics#Non-parametric_models)/[non-linear](http://www.yaksis.com/static/img/02/cows_and_wolves.png)/model complex phenomenom
* Computationally _cheap_ to ___predict___
* Can handle irrelevant features
* Mixed data (nominal and continuous)

## Why Not Decision Trees

* Computationally _expensive_ to ___train___
* Greedy algorithm (local optima)
* Very easy to overfit

## Goals

* Train vs. Test
* Non-parametric models
* CART algorithm
* Conditional Independence
* Maximum Likelihood
* Conditional Probability Table
* Gini coefficient vs. information gain 

## Reading

* [Applied Data Science](http://columbia-applied-data-science.github.io/appdatasci.pdf): Chapter 9.4 (p. 100 - p. 104)
* Machine Learning in Action: Chapter 3 ([ID3](http://en.wikipedia.org/wiki/ID3_algorithm))
* [scikit-learn docs: Decision Trees](http://scikit-learn.org/stable/modules/tree.html)

## Assignment

Decision trees are a recursive divide and conquer algorithm.  You should have a good handle on recursion before beginning this sprint.  There are a few names of decision tree algorithms you may have heard of (ID3, C4.5, CART, etc.) and each is a different specification of a decision tree model.  You can read about them [here](http://stackoverflow.com/questions/9979461/different-decision-tree-algorithms-with-comparison-of-complexity-or-performance) and [here](http://scikit-learn.org/stable/modules/tree.html#tree-algorithms-id3-c4-5-c5-0-and-cart). 

For this assignment you are the new data science hire on the [Eventbrite](http://www.eventbrite.com/jobs/?jobvite=true&jvi=olDVXfwA%2CJob#current_positions) data team:

![win](http://media.giphy.com/media/Xd5ayHbo4Umc0/giphy.gif)

(congrats)

Now you have been given your first assignment... prevent fraudulent events.  Now to detect future fraudulent events, you have decided to look for any patterns in past events.  Using a decision tree you want to not only classify fraudulent events, but understand why they are fraudulent to take the appropriate action.  But let's not get ahead of ourselves... we first have to understand how to build a tree (and walk before we run).

### Implementation

When implementing any ML algorithm for the first time, it is often easier to start with a trivially simple data set.  You should always focus on one portion of the pipeline at a time: we do not want worry about cleaning data during feature selection just as we do not want to worry about feature engineering when writing our model building code.  We will be using the canonical 'Play Golf' [dataset](http://www2.cs.uregina.ca/~dbd/cs831/notes/ml/dtrees/c4.5/c4.5_prob1.html) when writing our algorithm.  Reasons for this:

* Small enough to understand with dimensionality reduction/plotting
* Very well defined (and known) solution
* Can compute all quantities (entropy, splits, etc.) used in the algorithm by hand.

### Pseudo-code

```
If every item in the dataset is in the same class: 
	return the class label
Else
	find the best feature to split the data 
	split the dataset
	create a branch node
	for each split
		call createBranch and add the result to the branch node
	return branch node
```

1. Download the [data](http://www2.cs.uregina.ca/~dbd/cs831/notes/ml/dtrees/c4.5/golf.data).  For our implementation of a decision tree, we will be implementing the [C4.5](http://en.wikipedia.org/wiki/C4.5_algorithm) algorithm.  

2. To store our tree, we will create a very simple `Node` class.  Create a `TreeNode` class in Python.  It will need the following attributes to store information at each split:

* column (int): index of feature to split on
* children (dict): dictionary representing child nodes.  Should have the value of the feature as a key and the child node as the value.  
* leaf (boolean): true or false depending on if the node is a leaf node.    
* classes (list): if a leaf, a list of all the classes of the data points that terminate at this leaf.  Can be used to assess how "accurate" an individual leaf is.

3. Now that we have a way to store our tree, we need a way to build the tree.  There are a few components to this, first we need to be able to assess the quality of each split. First write a function that computes the [entropy](http://en.wikipedia.org/wiki/Entropy_%28information_theory%29) of a data set. 

![shannon](http://imed.med.ucm.es/icons/shanon_equ.png)

4. In order to compute the best feature to split on, we need to be able to consider all the possibilities of splits.  For this we need a function to segment our dataset.  Write a function that takes as arguments a __dataset__ a __feature__ index to split on, and a __value__ to split on.  It should return the data points from the input __dataset__ with the given __value__ for the __feature__.

5. Now that we know how much variability is contained in a set (or subset) of our instances and can efficiently segment our dataset, we can calculate the [information gain](http://en.wikipedia.org/wiki/Information_gain_in_decision_trees) of a potential split. Write a function that computes the information gain of a potential split. It should take an only an input dataset.  We need a few quantities to calculate the best split:

* base entropy: entropy of the input dataset
* entropy of each feature split: sum the entropies of a split on each value of a feature.

For each feature in the dataset, we will calculate a sum of the entropy of the splits.  Use the previously defined function to split the dataset on all possible values of that feature.  For each split calculate the entropy of the subset data contained in the split, and add this the total entropy of the feature split. Calculate the difference of this total entropy from the base entropy of the initial dataset before the split.   Do this for each feature and select the one with the most information gain.

![gain](http://dms.irb.hr/tutorial/images/gain_eq.gif)

5. Now we can begin building the tree.  At each level of the tree we want to compute the gain for a split on each feature.  For the C4.5 algorithm, it 'consumes' the features, i.e. once you split (and build nodes) for feature 2, you remove it from the instances of your data.  Write a recursive function that returns a tree.  Each step of the recursion represents one decision node and recursively partitions your data set.  The recursion stops when you run out of features to split on, or all of the classes of the data points in a node are equal (pure leaf).

6. In the recursion, for the feature split that results in the highest gain create new nodes in the tree.  You should create a child node for each unique value of the chosen feature.  Ex: For `Outlook` there are 3 unique values: Sunny, Overcast, and Rainy. Add these new nodes as children in your tree.  There should be a unique path through the tree.

7. Once you have built (trained) your tree, you can start predicting.  Write a method in the `TreeNode` class called `predict()`.  It should take a data point and return the predicted class.  If the node has children, it should call predict on the correct child and pass the instance as an argument.  If predict() is called on a leaf, it sinmply returns the `value` at that node.

8. Test that you get the same tree as output as the golf example.  Once you have gotten the same result, you can start experimenting with more interesting datasets.

## Extra

Play around with some more interesting data sets.  Run you tree algorithm on a breast cancer [data set](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29) to predict malignant tumors.  You are essentially a doctor now.  Go into the world and heal.

### Decision boundaries

![boundary](http://www.yaksis.com/static/img/02/cows_and_wolves.png)

![boundary](http://scikit-learn.org/stable/_images/plot_iris_11.png)

### Serialization

To serialize a decision tree, we simply need to output the decision rules.  This can be in the form of a data structure (Python class, dictionary of dictionaries, etc.) or code (Python conditional statements).  These two forms are illustrated below.

#### Data Structure

```javascript
{
	column:,
	value:,
	tnode: {
			column:,
			...
		   },
	fnode: {
			column:,
			...
		   },
}
```

#### OR

```python
class DecisionNode:
	def __init__(self, )
		# index of feature for decision (column in vector)
		self.column = col
		# value for true split
		self.value = val
		# next true and false nodes respectively
		self.tnode = split_true
		self.fnode = split_false

classifier = DecisionTreeClassifier()
classifier.fit(X, Y)

# returns a DecisionNode instance representing the root of the tree
tree = classifier.model()

# serialize the model
pickle(tree, open('tree.pkl', 'wb'))

```

#### Code

```python
if feature[2] == "Male":
	if feature[1] > 14:
		return "man-child"
	else:
		return "child"
else:
	return "adult"
```

## References

* Machine Learning in Action: Chapter 9 ([CART](http://en.wikipedia.org/wiki/Predictive_analytics#Classification_and_regression_trees))
* Collective Intelligence: Chapter 7
* [Practical Machine Learning -- Classification](slides.pdf)
* [Tom Mitchell's Book](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/mlbook/ch3.pdf)