---
toc: true
layout: post
description: "From Decision Trees to ensembles with Random Forests"
categories: [post]
tags: [Decision Tress, Random Forest, Ensembles]
title: "Intro to Ensembles"
image: "images/posts/DTPost.png"
comments: true
featured: true
code-block-font-size: \normalsize
---

## **Decision Trees**

- The main idea of a decision tree is to **identify the features which contain the most information regarding the target feature** **and then split the dataset** along the values of these features **such that the target feature values at the resulting nodes are as pure as possible**. **A feature that best separates the uncertainty from information about the target feature is said to be the most informative feature.**
- **It's useful to think of a decision tree as a flow of data rows. When we make a split, some rows will go to the right, and some will go to the left. As we build the tree deeper and deeper, each node "receives" fewer and fewer rows.**
- The process of building a decision tree involves asking a question at every instance and then continuing with the split, When there are multiple features that decide the target value of a particular instance, **which feature should be chosen as the root node to start the splitting process? And in which order should we continue choosing the features at every further split at a node?**
    - Here comes the need to measure the informativeness of the features and use the feature with the most information as the feature to split the data on. This informativeness is given by a measure called ‘**information gain**’.
- Decision trees typically suffer from high variance. The entire structure of a decision tree can change if we make a minor alteration to its training data. By restricting the depth of the tree, we increase the bias and decrease the variance. If we restrict the depth too much, we increase bias to the point where it underfits.
<br/><br/>

### Entropy

- It is used to **measure the impurity or randomness of a dataset**. Imagine choosing a yellow ball from a box of just yellow balls (say 100 yellow balls). Then this box is said to have 0 entropy which implies 0 impurity or total purity.
    - $Entropy(x)=-\sum[P(x=k)*log_2(P(x=k))]$ , **where x is the target feature we are predicting for.**
        
    - **Where $P(x=k)$ is the probability that a target feature takes a specific value, k**. This $P(x=k)$ simply means the proportion of value $k$ over all the number of samples in the target feature. Ex: a target feature of colors (30 red & 70 blue), then $P(x=red)=0.3$ & $P(x=blue)=0.7$.
        
        ![]({{ "images/posts/DT1.png" | relative_url }})
        
    - **Logarithm of fractions (log with base 2) in the equation gives a negative value and hence a ‘-‘ sign is used in entropy formula to negate these negative values**. The maximum value for entropy depends on the number of classes:
        - 2 classes: Max entropy is 1
        - 4 Classes: Max entropy is 2
        - 8 Classes: Max entropy is 3
        - 16 classes: Max entropy is 4<br/><br/>
        
### Information Gain

- $IG(T,A)=Entropy(T)−∑_{v∈A}\frac{|T_v|}{|T|}*Entropy(T_v)$
- We first calculate the entropy of the target feature $T$ before the split. —→>> $Entropy(T)$
- Variable $A$ is the feature we will split on. For each unique value $v$ in $A$, we compute the number of rows of $T$ in which $A$ takes on the value $v$ and divide it by the total number of rows in $A$. This simply will give weighted entropy for each value $v$. —→>> $\frac{|T_v|}{|T|}$
- Next, we multiply the results by the entropy of the target feature where the rows of $A$ is $v$. **(Simply, we isolate the rows where $A=v$ & calculate post-split entropy for feature T)**
- We add all of these subset entropies together, then subtract from the overall entropy to get information gain.
- **We choose the feature that gives the HIGHEST information gain, as this indicates lower post-split entropies for the selected feature to split on & more informative & pure nodes.**<br/><br/>

    Example:

    ![]({{ "images/posts/DT2.png" | relative_url }})

    - Here when $P(age=0)=4/5$ & $P(age=1)=1/5$, so these are the weights ($\frac{|T_v|}{|T|}$) of which we will multiply post-split entropies by.
    - We split the dataset into two parts, one part where $age=0$ & another when $age=1$. We calculate the entropy for the resulting two part.
    - The first part entropy when $age=0$ will have 4 samples of the target $T$ & will equal $-(\frac{2}{4}log_2\frac{2}{4}+\frac{2}{4}log_2\frac{2}{4})$
    - The second part entropy when $age=1$ will have 1 sample of the target $T$ & will equal $-(\frac{1}{5}log_2\frac{1}{5}+\frac{0}{5}log_2\frac{0}{5})$<br/><br/>

### Gini Index 

- $GI=1-\sum(P(x=k))^2$
- It is calculated by subtracting the sum of squared probabilities of each value $k$ from target feature $T$, from one, in the current node we are splitting to.
- **A feature with a lower Gini index is chosen for a split.**
- It favors larger partitions and easy to implement whereas information gain favors smaller partitions with distinct values.
<br/><br/>

### Overfitting

Three ways to combat overfitting:

- "**Prune**" the tree after we build it to remove unnecessary leaves.
- Use **ensembles of trees** to blend the predictions of many trees.
- **Restrict the depth** of the tree while we're building it.<br/><br/><br/>

## **Ensembles**

### `predict_proba()`

- `DecisionTreeClassifier.predict_proba()` will predict a probability from 0 to 1 that a given class is the right one for a row. This calculation is done through naive bayes, & independence between feature is taken into consideration [ P(class | feature 1, feature 2, feature 3 ...) = P(class)*P(feature 1)*P(feature 2)*P(feature 3)].
    - Because `0` and `1` are our two classes, we'll get a matrix containing the number of rows in the dataset, and two columns.  `predict_proba()` will return a result that looks like this:
    - **0        1**
    0.7    0.3
    0.2    0.8
    0.1    0.9
    - Each row will correspond to a prediction.  The first column is the probability that the prediction is a `0`, and the second column is the probability that the prediction is a `1`.  Each row adds up to `1`.
<br/><br/>    
- The more "diverse" or dissimilar the models we use to construct an ensemble are, the stronger their combined predictions will be (assuming that all of the models have about the same accuracy). Ensembling a decision tree and a logistic regression model, for example, will result in stronger predictions than ensembling two decision trees with similar parameters.  That's because those two models use very different approaches to arrive at their answers.
    - If we build two different decision trees, the models are approaching the same problem in slightly different ways, and building different trees because we used different parameters for each one.  Each tree makes different predictions in different areas.  Even though both trees have about the same accuracy, when we combine them, the result is stronger because it leverages the strengths of both approaches.<br/><br/>
- On the other hand, if the models we ensemble are very similar in how they make predictions, ensembling will result in a negligible boost.
- Ensembling models with very different accuracies generally won't improve overall accuracy.  Ensembling a model with a `0.75` AUC and a model with a `0.85` AUC on a test set will usually result in an AUC somewhere in between the two original values.
- **min_impurity_decrease** float, default=0.0
A node will be split if this split induces a decrease of the impurity greater than or equal to this value.<br/><br/>

### Ensembles Prediction

- There are multiple methods to get predictions for ensembles of models.
- **Majority Voting:**
    - One method is majority voting, in which **each classifier gets a "vote," and the most commonly voted value for each row "wins.**"  This only works if there are more than two classifiers (and ideally an odd number, so we don't have to write a rule to break ties).
    - DT1     DT2    DT3    Final Prediction
    0           1         0                0
    1           1         1                1
    0           0         1                0
    1           0         0                0
- **Row mean:**
    - If we use the `predict_proba()` method on two classifiers to generate probabilities as results, take the mean for each row, and then round the results, we'll get ensemble predictions.<br/><br/>

## **Random Forests**

- A random forest is an ensemble of decision trees. In order to make ensembling effective, we have to introduce variation into each individual decision tree model.
- When we instantiate a `RandomForestClassifier` or `RandomForestRegressor`, we pass in an `n_estimators` parameter that **indicates how many trees to build**.  While adding more trees usually improves accuracy, it also increases the overall time the model takes to train.
    - `clf = **RandomForestClassifier(**n_estimators=5, random_state=1, min_samples_leaf=2**)**`
- **Bootstrapping** is a statistical resampling technique that involves **random sampling of a dataset *with replacement***.
- There are two main ways to introduce variation in a random forest -- **bagging** and **random feature subsets**.<br/><br/>
- Bagging, “Bootstrap Aggregation”:
    - Bootstrap aggregation means we have groups (aggregations) of bags where each bag a sampled (bootstrapped) samples of the original dataset.
    - In a random forest, we don't train each tree on the entire data set. **We train it on a random sample of the data, or a "bag,"** instead. We perform this sampling **with replacemen**t, which means that after we select a row from the data we're sampling, we put the row back in the data so it can be picked again. Some rows from the original data may appear in the "bag" multiple times.
    - When bagging with decision trees, we are less concerned about individual trees overfitting the training data. For this reason and for efficiency, the individual decision trees are grown deep (e.g. few training samples at each leaf-node of the tree) and the trees are not pruned. These trees will have both high variance and low bias. These are important characteristics of sub-models when combining predictions using bagging.<br/><br/>
- Random Feature Subsets:
    - We can also repeat our random subset selection process in scikit-learn.  We just set the `splitter` parameter on `DecisionTreeClassifier` to `"random"`, and the `max_features` parameter to`"auto"`.  If we have N columns, this will pick a subset of features of size √N, compute the Gini coefficient for each (or information gain), and split the node on the best column in the subset.
    - **splitter**{“best”, “random”}, default=”best”
        - The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.
    - **max_features** int, float or {“auto”, “sqrt”, “log2”}, default=None
        - The number of features to consider when looking for the best split
    - ```python
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=5, random_state=1, min_samples_leaf=2)
        ```
<br/>

- The main strengths of a random forest are:
    - Very accurate predictions - Random forests achieve near state-of-the-art performance on many machine learning tasks. Along with neural networks and gradient-boosted trees, they're typically one of the top-performing algorithms.
    - Resistance to overfitting - Due to their construction, random forests are fairly resistant to overfitting. We still need to set and tweak parameters like `max_depth` though.

### Seeding

- The random seed isn't strictly a hyperparameter, but we introduce it here to highlight that this external parameter can play a role in the effectiveness of training. While this is usually minor, if the model is very complex, and/or the amount of data available is small, the test-set performance of the model can be markedly different if two different seeds are used. In such situations, often it pays to run training with multiple different seeds to assess to what degree your model design is adequate, and to what degree your performance is simply ‘blind luck’.
    
    

```python {font_size :}
#Using Bagging & Random feature subsets to form a Random Forest
# We'll build 10 trees
tree_count = 10

# Each "bag" will have 60% of the number of original rows
bag_proportion = 0.6

predictions = []
for i in range(tree_count):
	#**BAGGING**
    # We select 60% of the rows from train, sampling with replacement
    # We set a random state to ensure we'll be able to replicate our results
    # We set it to i instead of a fixed value so we don't get the same sample every time
    bag = train.sample(**frac=bag_proportion**, replace=True, random_state=i)
    
    # Fit a decision tree model to the "bag"
	**#Random Feature Subsets**
    clf = DecisionTreeClassifier(random_state=1, min_samples_leaf=2, **splitter="random", max_features="auto"**)
    clf.fit(bag[columns], bag["high_income"])
    
    # Using the model, make predictions on the test data
    predictions.append(clf.predict_proba(test[columns])[:,1])

combined = numpy.sum(predictions, axis=0) / 10
rounded = numpy.round(combined)

print(roc_auc_score(test["high_income"], rounded))
```

<br/><br/>
<u>**Useful links**</u>:
<br/><br/>
[Entropy, Information gain and Gini Index; the crux of a Decision Tree](https://blog.clairvoyantsoft.com/entropy-information-gain-and-gini-index-the-crux-of-a-decision-tree-99d0cdc699f4)

[Decision Tree Adventures 2 - Explanation of Decision Tree Classifier Parameters](https://medium.datadriveninvestor.com/decision-tree-adventures-2-explanation-of-decision-tree-classifier-parameters-84776f39a28)