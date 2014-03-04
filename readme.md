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

#### Decision Tree

* [Applied Data Science](http://columbia-applied-data-science.github.io/appdatasci.pdf): Chapter 9.4 (p. 100 - p. 104)
* Machine Learning in Action: Chapter 3 ([ID3](http://en.wikipedia.org/wiki/ID3_algorithm))
* [scikit-learn docs: Decision Trees](http://scikit-learn.org/stable/modules/tree.html)

#### Random Forest

* [Random Forests in Python](http://blog.yhathq.com/posts/random-forests-in-python.html)
* Machine Learning in Action: Chapter 7 (AdaBoost and Ensambles) 
* [Random Forest Original paper](http://oz.berkeley.edu/~breiman/randomforest2001.pdf)
* [Feature Engineering](http://www.cs.berkeley.edu/~jordan/courses/294-fall09/lectures/feature/slides.pdf)

## Assignment

Decision trees are a recursive divide and conquer algorithm. In the parlance of the last sprint, they are a non-linear, non-parametric discriminative supervised classification algorithm.  There are a few names of decision tree algorithms you may have heard of (ID3, C4.5, CART, etc.) and each is a different specification of a decision tree model.  You can read about them [here](http://stackoverflow.com/questions/9979461/different-decision-tree-algorithms-with-comparison-of-complexity-or-performance) and [here](http://scikit-learn.org/stable/modules/tree.html#tree-algorithms-id3-c4-5-c5-0-and-cart). 

We will start by using [BigML](https://bigml.com) to quickly explore different datasets to get a feel for how decision trees work.  Some videos on how to use it can be found [here](https://bigml.com/features#models) but it is pretty intuitive. If you would like to code a decision tree from scratch, let me know and I can give you access to an old sprint.

Now you have been given your first assignment... prevent fraudulent events.  Now to detect future fraudulent events, you have decided to look for any patterns in past events.  Using a decision tree you want to not only classify fraudulent events, but understand why they are fraudulent to take the appropriate action.  But let's not get ahead of ourselves... we first have to understand how to build a tree (and walk before we run).

### Implementation

When implementing any ML algorithm for the first time, it is often easier to start with a trivially simple data set.  You should always focus on one portion of the pipeline at a time: we do not want worry about cleaning data during feature selection just as we do not want to worry about feature engineering when writing our model building code.  We will be using the canonical 'Play Golf' [dataset](http://www2.cs.uregina.ca/~dbd/cs831/notes/ml/dtrees/c4.5/c4.5_prob1.html) when writing our algorithm.

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

1. Download the [data](http://www2.cs.uregina.ca/~dbd/cs831/notes/ml/dtrees/c4.5/golf.data).  Upload this dataset to Bigml.  You will have to create a free account.  We will be using BigML because it has a nice intuitive interface, but more importantly has one of the best decision tree visualizations out there.  It has much information in the output itself and has parameters you can tune.

2.  Click on your recently uploaded dataset.  This should bring you to a window that displays the columns.  The dataset itself doesn't have column names.  Edit the column names to whatever you want (is most understandable to you).  Once done, there is a button in the top right called 'Configure Dataset'.  Click this and it should show you histograms of all your columns.

3.  With such a trivial dataset there are not too many interesting things to explore with this interface.  Click on the lightning cloud icon in the top right of the dashboard to run a '1-click model'.  This will build our tree.

For each feature in the dataset, a DT calculates a sum of the entropy of the splits.  It then splits the dataset on all possible values of that feature.  For each split calculate the entropy of the subset data contained in the split, and add this the total entropy of the feature split. It tries to split each feature and select the one with the most information gain.

![gain](http://dms.irb.hr/tutorial/images/gain_eq.gif)

4. Do you notice anything about this tree?  How does it compare to [this](http://www2.cs.uregina.ca/~dbd/cs831/notes/ml/dtrees/c4.5/c4.5_prob1.html) picture?

5. Apparently BigML has chosen to standardize on the [CART](http://statweb.stanford.edu/~lpekelis/talks/13_datafest_cart_talk.pdf) algorithm.  CART is one of the most popular algorithms for decision trees because it handles both categorical and continuous inputs and outputs.  This makes it very flexible.

6. Support is the percentage of data points that pass through a node.  Confidence is the likelihood of the predicted outcome, given that the node's rule has been satisfied.  This is a ratio of the classes at a given node.  BigML seems to give strange answers for this sometimes.

7. To play with these values we will use the Titanic dataset.  It should be preloaded in your dashboard.  Run a model.  Play with the confidence slider.  What can we say about surviving the Titanic with 85% confidence?  Who was most likely to survive the Titanic?  Who was most likely to die?

8. Confidence is not the only metric of importance.  While a certain branch may not be the most likely (confidence), it may dictate the fate of much of the data.  Looking at the width of the edges (paths) in the tree, which features direct most of the data?  What is the first feature that splits the data?

9.  Decision trees are often used for their interpretablity and actionability.  It's output may not be the most accurate predictor, but compared to other black box algorithms, decision trees can work wonders in terms of guiding business decisions.

### Fraud

1. We will now switch gears and try to predict fraudulent events.


### Random Forests (and Ensembles)

Random forests are one of the most widely used classifiers for their robustness, generalizability, and accuracy.  There are a number of components intrinsic to the Random Forest classifier that make it both robust to overfitting while at the same time very accurate.  This is often true of many ensembles of models.  While we will not directly use ensemble methods (meta-algorithms -- AdaBoost and Bagging), Random Forests are a Bagging (Bootstrap Aggregating) algorithm.

Because of this sampling (bootstrapping), Random Forests can both perform feature selection and a form of cross validation as the algorithm trains.  They also can perform regression and classification on continuous and categorical inputs.  Basically anything you can throw at a Random forest, it can predict on... accurately.

### Exercise

For this exercise, we will be using scikit-learn to get some familiarity with these aspects of the algorithm.  See the `random_forests.ipynb` notebook for this part of the exercise.  

_Much of the Random Forest notebooks were contributed by Nitin Borwankar._

## Extra

Play around with some more interesting data sets.  Run a decision tree on a breast cancer [data set](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29) to predict malignant tumors.  You are essentially a doctor now.  Go into the world and heal.

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