## Overview -- Decision trees

__ML pipeline:__ Obtain/Prepare (features) __->__ Train __->__ Validate/Test __->__ Predict 
 
We will using a [decision tree classifier](http://en.wikipedia.org/wiki/Decision_tree_learning) to predict.  Decision trees are one of the most popular and widely used algorithms.  Most classifiers (SVM, kNN, Neural Nets) are great at giving you a (somewhat) accurate result, but are often black boxes.  With these algorithms it can be hard to interpret their results and understand ___why___ a certain instance was assigned a label.  Decision trees are unique in that they are very flexible and accurate while also being easily interpreted.

![c4.5](images/golftree.gif)

__INPUTS:__ Nominal (discrete) or Continuous

__OUTPUTS:__ Nominal (discrete) or Continuous

__(basically anything in and anything out)__

### Why Decision Trees

* Easily interpretable
* Handles missing values and outliers
* [non-parametric](http://en.wikipedia.org/wiki/Non-parametric_statistics#Non-parametric_models)/[non-linear](http://www.yaksis.com/static/img/02/cows_and_wolves.png)/model complex phenomenom
* Computationally _cheap_ to ___predict___
* Can handle irrelevant features
* Mixed data (nominal and continuous)

### Why Not Decision Trees

* Computationally _expensive_ to ___train___
* Greedy algorithm (local optima)
* Very easy to overfit

## k Nearest Neighbors (kNN)

__INPUTS:__ Nominal (discrete) or Continuous

__OUTPUTS:__ Nominal (discrete) or Continuous

k Nearest Neighbors is another classifier that is very simple to implement. Using a distance metric, you determine which k data points are closest to the one you put in and give each of them a vote on the prediction.