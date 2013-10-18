Main teacher: Jonathan

## Overview -- Decision trees

__ML pipeline:__ Obtain/Prepare (features) __->__ Train __->__ Validate/Test __->__ Predict 
 
In this assignment, we will be implementing a [decision tree classifier](http://en.wikipedia.org/wiki/Decision_tree_learning) in Python (no scikit-learn).  Decision trees are one of the most popular and widely used algorithms.  Most classifiers (SVM, kNN, Neural Nets) are great at giving you a (somewhat) accurate result, but are often black boxes.  With these algorithms it can be hard to interpret their results and understand ___why___ a certain instance was assigned a label.  Decision trees are unique in that they are very flexible and accurate while also being easily interpreted.

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

* [scikit-learn docs: Decision Trees](http://scikit-learn.org/stable/modules/tree.html)

## Assignment

Decision trees are a recursive divide and conquer algorithm.  You should have a good handle on recursion before beginning this sprint.  

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

![boundary](http://www.yaksis.com/static/img/02/cows_and_wolves.png)


## Extra credit

Try to run each of these other classifiers over the dataset and compare the results.  You probably don't want to implement all these in SQL (though it is [possible](http://madlib.net/) and [companies](https://alpine.atlassian.net/wiki/display/DOC/MADlib+Operators) have been built on it). I recommend exporting our bags into a CSV and using scikit-learn. 

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


```


#### Logistic Regression

#### SVM
* Perceptron and MIRA
* Kernel methods

#### Smoothing
* Laplace Smoothing
* Linear Interpolation

## References

* Machine Learning in Action: Chapter 3 ([ID3](http://en.wikipedia.org/wiki/ID3_algorithm)) and 9 ([CART](http://en.wikipedia.org/wiki/Predictive_analytics#Classification_and_regression_trees))

* [Practical Machine Learning -- Classification](slides.pdf)
* [Tom Mitchell's Book](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/mlbook/ch3.pdf)


Glossary
-------
I need to create an image to show all these.  Make a table of data points and label each of these.

* non-parametric
* decision rules
* 