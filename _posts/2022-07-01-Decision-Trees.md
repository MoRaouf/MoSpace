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

---
<h2 id="decision-trees"><strong>Decision Trees</strong></h2>
<ul>
<li>The main idea of a decision tree is to <strong>identify the features
which contain the most information regarding the target feature</strong>
<strong>and then split the dataset</strong> along the values of these
features <strong>such that the target feature values at the resulting
nodes are as pure as possible</strong>. <strong>A feature that best
separates the uncertainty from information about the target feature is
said to be the most informative feature.</strong></li>
<li><strong>It’s useful to think of a decision tree as a flow of data
rows. When we make a split, some rows will go to the right, and some
will go to the left. As we build the tree deeper and deeper, each node
“receives” fewer and fewer rows.</strong></li>
<li>The process of building a decision tree involves asking a question
at every instance and then continuing with the split, When there are
multiple features that decide the target value of a particular instance,
<strong>which feature should be chosen as the root node to start the
splitting process? And in which order should we continue choosing the
features at every further split at a node?</strong>
<ul>
<li>Here comes the need to measure the informativeness of the features
and use the feature with the most information as the feature to split
the data on. This informativeness is given by a measure called
‘<strong>information gain</strong>’.</li>
</ul></li>
<li>Decision trees typically suffer from high variance. The entire
structure of a decision tree can change if we make a minor alteration to
its training data. By restricting the depth of the tree, we increase the
bias and decrease the variance. If we restrict the depth too much, we
increase bias to the point where it underfits. <br/><br/></li>
</ul>
<h3 id="entropy">Entropy</h3>
<ul>
<li>It is used to <strong>measure the impurity or randomness of a
dataset</strong>. Imagine choosing a yellow ball from a box of just
yellow balls (say 100 yellow balls). Then this box is said to have 0
entropy which implies 0 impurity or total purity.
<ul>
<li><p><span
class="math inline">\(Entropy(x)=-\sum[P(x=k)*log_2(P(x=k))]\)</span> ,
<strong>where x is the target feature we are predicting
for.</strong></p></li>
<li><p><strong>Where <span class="math inline">\(P(x=k)\)</span> is the
probability that a target feature takes a specific value, k</strong>.
This <span class="math inline">\(P(x=k)\)</span> simply means the
proportion of value <span class="math inline">\(k\)</span> over all the
number of samples in the target feature. Ex: a target feature of colors
(30 red &amp; 70 blue), then <span
class="math inline">\(P(x=red)=0.3\)</span> &amp; <span
class="math inline">\(P(x=blue)=0.7\)</span>.</p>
<p>[]({{ “images/posts/DT1.png” | relative_url }})</p></li>
<li><p><strong>Logarithm of fractions (log with base 2) in the equation
gives a negative value and hence a ’-‘ sign is used in entropy formula
to negate these negative values</strong>. The maximum value for entropy
depends on the number of classes:</p>
<ul>
<li>2 classes: Max entropy is 1</li>
<li>4 Classes: Max entropy is 2</li>
<li>8 Classes: Max entropy is 3</li>
<li>16 classes: Max entropy is 4<br/><br/></li>
</ul></li>
</ul></li>
</ul>
<h3 id="information-gain">Information Gain</h3>
<ul>
<li><p><span
class="math inline">\(IG(T,A)=Entropy(T)−∑_{v∈A}\frac{|T_v|}{|T|}*Entropy(T_v)\)</span></p></li>
<li><p>We first calculate the entropy of the target feature <span
class="math inline">\(T\)</span> before the split. —→&gt;&gt; <span
class="math inline">\(Entropy(T)\)</span></p></li>
<li><p>Variable <span class="math inline">\(A\)</span> is the feature we
will split on. For each unique value <span
class="math inline">\(v\)</span> in <span
class="math inline">\(A\)</span>, we compute the number of rows of <span
class="math inline">\(T\)</span> in which <span
class="math inline">\(A\)</span> takes on the value <span
class="math inline">\(v\)</span> and divide it by the total number of
rows in <span class="math inline">\(A\)</span>. This simply will give
weighted entropy for each value <span class="math inline">\(v\)</span>.
—→&gt;&gt; <span
class="math inline">\(\frac{|T_v|}{|T|}\)</span></p></li>
<li><p>Next, we multiply the results by the entropy of the target
feature where the rows of <span class="math inline">\(A\)</span> is
<span class="math inline">\(v\)</span>. <strong>(Simply, we isolate the
rows where <span class="math inline">\(A=v\)</span> &amp; calculate
post-split entropy for feature T)</strong></p></li>
<li><p>We add all of these subset entropies together, then subtract from
the overall entropy to get information gain.</p></li>
<li><p><strong>We choose the feature that gives the HIGHEST information
gain, as this indicates lower post-split entropies for the selected
feature to split on &amp; more informative &amp; pure
nodes.</strong><br/><br/></p>
<p>Example:</p>
<p>[]({{ “images/posts/DT2.png” | relative_url }})</p>
<ul>
<li>Here when <span class="math inline">\(P(age=0)=4/5\)</span> &amp;
<span class="math inline">\(P(age=1)=1/5\)</span>, so these are the
weights (<span class="math inline">\(\frac{|T_v|}{|T|}\)</span>) of
which we will multiply post-split entropies by.</li>
<li>We split the dataset into two parts, one part where <span
class="math inline">\(age=0\)</span> &amp; another when <span
class="math inline">\(age=1\)</span>. We calculate the entropy for the
resulting two part.</li>
<li>The first part entropy when <span
class="math inline">\(age=0\)</span> will have 4 samples of the target
<span class="math inline">\(T\)</span> &amp; will equal <span
class="math inline">\(-(\frac{2}{4}log_2\frac{2}{4}+\frac{2}{4}log_2\frac{2}{4})\)</span></li>
<li>The second part entropy when <span
class="math inline">\(age=1\)</span> will have 1 sample of the target
<span class="math inline">\(T\)</span> &amp; will equal <span
class="math inline">\(-(\frac{1}{5}log_2\frac{1}{5}+\frac{0}{5}log_2\frac{0}{5})\)</span><br/><br/></li>
</ul></li>
</ul>
<h3 id="gini-index">Gini Index</h3>
<ul>
<li><span class="math inline">\(GI=1-\sum(P(x=k))^2\)</span></li>
<li>It is calculated by subtracting the sum of squared probabilities of
each value <span class="math inline">\(k\)</span> from target feature
<span class="math inline">\(T\)</span>, from one, in the current node we
are splitting to.</li>
<li><strong>A feature with a lower Gini index is chosen for a
split.</strong></li>
<li>It favors larger partitions and easy to implement whereas
information gain favors smaller partitions with distinct values.
<br/><br/></li>
</ul>
<h3 id="overfitting">Overfitting</h3>
<p>Three ways to combat overfitting: - “<strong>Prune</strong>” the tree
after we build it to remove unnecessary leaves. - Use <strong>ensembles
of trees</strong> to blend the predictions of many trees. -
<strong>Restrict the depth</strong> of the tree while we’re building
it.<br/><br/><br/></p>
<h2 id="ensembles"><strong>Ensembles</strong></h2>
<h3 id="predict_proba"><code>predict_proba()</code></h3>
<ul>
<li><code>DecisionTreeClassifier.predict_proba()</code> will predict a
probability from 0 to 1 that a given class is the right one for a row.
This calculation is done through naive bayes, &amp; independence between
feature is taken into consideration [ P(class | feature 1, feature 2,
feature 3 …) = P(class)<em>P(feature 1)</em>P(feature 2)*P(feature 3)].
<ul>
<li>Because <code>0</code> and <code>1</code> are our two classes, we’ll
get a matrix containing the number of rows in the dataset, and two
columns. <code>predict_proba()</code> will return a result that looks
like this:</li>
<li><strong>0 1</strong> 0.7 0.3 0.2 0.8 0.1 0.9</li>
<li>Each row will correspond to a prediction. The first column is the
probability that the prediction is a <code>0</code>, and the second
column is the probability that the prediction is a <code>1</code>. Each
row adds up to <code>1</code>. <br/><br/><br />
</li>
</ul></li>
<li>The more “diverse” or dissimilar the models we use to construct an
ensemble are, the stronger their combined predictions will be (assuming
that all of the models have about the same accuracy). Ensembling a
decision tree and a logistic regression model, for example, will result
in stronger predictions than ensembling two decision trees with similar
parameters. That’s because those two models use very different
approaches to arrive at their answers.
<ul>
<li>If we build two different decision trees, the models are approaching
the same problem in slightly different ways, and building different
trees because we used different parameters for each one. Each tree makes
different predictions in different areas. Even though both trees have
about the same accuracy, when we combine them, the result is stronger
because it leverages the strengths of both approaches.<br/><br/></li>
</ul></li>
<li>On the other hand, if the models we ensemble are very similar in how
they make predictions, ensembling will result in a negligible
boost.</li>
<li>Ensembling models with very different accuracies generally won’t
improve overall accuracy. Ensembling a model with a <code>0.75</code>
AUC and a model with a <code>0.85</code> AUC on a test set will usually
result in an AUC somewhere in between the two original values.</li>
<li><strong>min_impurity_decrease</strong> float, default=0.0 A node
will be split if this split induces a decrease of the impurity greater
than or equal to this value.<br/><br/></li>
</ul>
<h3 id="ensembles-prediction">Ensembles Prediction</h3>
<ul>
<li>There are multiple methods to get predictions for ensembles of
models.</li>
<li><strong>Majority Voting:</strong>
<ul>
<li>One method is majority voting, in which <strong>each classifier gets
a “vote,” and the most commonly voted value for each row
“wins.</strong>” This only works if there are more than two classifiers
(and ideally an odd number, so we don’t have to write a rule to break
ties).</li>
<li>DT1 DT2 DT3 Final Prediction 0 1 0 0 1 1 1 1 0 0 1 0 1 0 0 0</li>
</ul></li>
<li><strong>Row mean:</strong>
<ul>
<li>If we use the <code>predict_proba()</code> method on two classifiers
to generate probabilities as results, take the mean for each row, and
then round the results, we’ll get ensemble predictions.<br/><br/></li>
</ul></li>
</ul>
<h2 id="random-forests"><strong>Random Forests</strong></h2>
<ul>
<li>A random forest is an ensemble of decision trees. In order to make
ensembling effective, we have to introduce variation into each
individual decision tree model.</li>
<li>When we instantiate a <code>RandomForestClassifier</code> or
<code>RandomForestRegressor</code>, we pass in an
<code>n_estimators</code> parameter that <strong>indicates how many
trees to build</strong>. While adding more trees usually improves
accuracy, it also increases the overall time the model takes to train.
<ul>
<li><code>clf = **RandomForestClassifier(**n_estimators=5, random_state=1, min_samples_leaf=2**)**</code></li>
</ul></li>
<li><strong>Bootstrapping</strong> is a statistical resampling technique
that involves <strong>random sampling of a dataset <em>with
replacement</em></strong>.</li>
<li>There are two main ways to introduce variation in a random forest –
<strong>bagging</strong> and <strong>random feature
subsets</strong>.<br/><br/></li>
<li>Bagging, “Bootstrap Aggregation”:
<ul>
<li>Bootstrap aggregation means we have groups (aggregations) of bags
where each bag a sampled (bootstrapped) samples of the original
dataset.</li>
<li>In a random forest, we don’t train each tree on the entire data set.
<strong>We train it on a random sample of the data, or a “bag,”</strong>
instead. We perform this sampling <strong>with replacemen</strong>t,
which means that after we select a row from the data we’re sampling, we
put the row back in the data so it can be picked again. Some rows from
the original data may appear in the “bag” multiple times.</li>
<li>When bagging with decision trees, we are less concerned about
individual trees overfitting the training data. For this reason and for
efficiency, the individual decision trees are grown deep (e.g. few
training samples at each leaf-node of the tree) and the trees are not
pruned. These trees will have both high variance and low bias. These are
important characteristics of sub-models when combining predictions using
bagging.<br/><br/></li>
</ul></li>
<li>Random Feature Subsets:
<ul>
<li>We can also repeat our random subset selection process in
scikit-learn. We just set the <code>splitter</code> parameter on
<code>DecisionTreeClassifier</code> to <code>"random"</code>, and the
<code>max_features</code> parameter to<code>"auto"</code>. If we have N
columns, this will pick a subset of features of size √N, compute the
Gini coefficient for each (or information gain), and split the node on
the best column in the subset.</li>
<li><strong>splitter</strong>{“best”, “random”}, default=”best”
<ul>
<li>The strategy used to choose the split at each node. Supported
strategies are “best” to choose the best split and “random” to choose
the best random split.</li>
</ul></li>
<li><strong>max_features</strong> int, float or {“auto”, “sqrt”,
“log2”}, default=None
<ul>
<li>The number of features to consider when looking for the best
split</li>
</ul>
<div class="sourceCode" id="cb1"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb1-1"><a href="#cb1-1" aria-hidden="true" tabindex="-1"></a><span class="im">from</span> sklearn.ensemble <span class="im">import</span> RandomForestClassifier</span>
<span id="cb1-2"><a href="#cb1-2" aria-hidden="true" tabindex="-1"></a>clf <span class="op">=</span> RandomForestClassifier(n_estimators<span class="op">=</span><span class="dv">5</span>, random_state<span class="op">=</span><span class="dv">1</span>, min_samples_leaf<span class="op">=</span><span class="dv">2</span>)</span></code></pre></div>
<br/><br/></li>
</ul></li>
<li>The main strengths of a random forest are:
<ul>
<li>Very accurate predictions - Random forests achieve near
state-of-the-art performance on many machine learning tasks. Along with
neural networks and gradient-boosted trees, they’re typically one of the
top-performing algorithms.</li>
<li>Resistance to overfitting - Due to their construction, random
forests are fairly resistant to overfitting. We still need to set and
tweak parameters like <code>max_depth</code> though.<br/><br/></li>
</ul></li>
</ul>
<h3 id="seeding">Seeding</h3>
<ul>
<li>The random seed isn’t strictly a hyperparameter, but we introduce it
here to highlight that this external parameter can play a role in the
effectiveness of training. While this is usually minor, if the model is
very complex, and/or the amount of data available is small, the test-set
performance of the model can be markedly different if two different
seeds are used. In such situations, often it pays to run training with
multiple different seeds to assess to what degree your model design is
adequate, and to what degree your performance is simply ‘blind
luck’.</li>
</ul>
<div class="sourceCode" id="cb2"><pre
class="sourceCode python"><code class="sourceCode python"><span id="cb2-1"><a href="#cb2-1" aria-hidden="true" tabindex="-1"></a><span class="co">#Using Bagging &amp; Random feature subsets to form a Random Forest</span></span>
<span id="cb2-2"><a href="#cb2-2" aria-hidden="true" tabindex="-1"></a><span class="co"># We&#39;ll build 10 trees</span></span>
<span id="cb2-3"><a href="#cb2-3" aria-hidden="true" tabindex="-1"></a>tree_count <span class="op">=</span> <span class="dv">10</span></span>
<span id="cb2-4"><a href="#cb2-4" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb2-5"><a href="#cb2-5" aria-hidden="true" tabindex="-1"></a><span class="co"># Each &quot;bag&quot; will have 60% of the number of original rows</span></span>
<span id="cb2-6"><a href="#cb2-6" aria-hidden="true" tabindex="-1"></a>bag_proportion <span class="op">=</span> <span class="fl">0.6</span></span>
<span id="cb2-7"><a href="#cb2-7" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb2-8"><a href="#cb2-8" aria-hidden="true" tabindex="-1"></a>predictions <span class="op">=</span> []</span>
<span id="cb2-9"><a href="#cb2-9" aria-hidden="true" tabindex="-1"></a><span class="cf">for</span> i <span class="kw">in</span> <span class="bu">range</span>(tree_count):</span>
<span id="cb2-10"><a href="#cb2-10" aria-hidden="true" tabindex="-1"></a>        <span class="co">#**BAGGING**</span></span>
<span id="cb2-11"><a href="#cb2-11" aria-hidden="true" tabindex="-1"></a>    <span class="co"># We select 60% of the rows from train, sampling with replacement</span></span>
<span id="cb2-12"><a href="#cb2-12" aria-hidden="true" tabindex="-1"></a>    <span class="co"># We set a random state to ensure we&#39;ll be able to replicate our results</span></span>
<span id="cb2-13"><a href="#cb2-13" aria-hidden="true" tabindex="-1"></a>    <span class="co"># We set it to i instead of a fixed value so we don&#39;t get the same sample every time</span></span>
<span id="cb2-14"><a href="#cb2-14" aria-hidden="true" tabindex="-1"></a>    bag <span class="op">=</span> train.sample(<span class="op">**</span>frac<span class="op">=</span>bag_proportion<span class="op">**</span>, replace<span class="op">=</span><span class="va">True</span>, random_state<span class="op">=</span>i)</span>
<span id="cb2-15"><a href="#cb2-15" aria-hidden="true" tabindex="-1"></a>    </span>
<span id="cb2-16"><a href="#cb2-16" aria-hidden="true" tabindex="-1"></a>    <span class="co"># Fit a decision tree model to the &quot;bag&quot;</span></span>
<span id="cb2-17"><a href="#cb2-17" aria-hidden="true" tabindex="-1"></a>        <span class="op">**</span><span class="co">#Random Feature Subsets**</span></span>
<span id="cb2-18"><a href="#cb2-18" aria-hidden="true" tabindex="-1"></a>    clf <span class="op">=</span> DecisionTreeClassifier(random_state<span class="op">=</span><span class="dv">1</span>, min_samples_leaf<span class="op">=</span><span class="dv">2</span>, <span class="op">**</span>splitter<span class="op">=</span><span class="st">&quot;random&quot;</span>, max_features<span class="op">=</span><span class="st">&quot;auto&quot;</span><span class="op">**</span>)</span>
<span id="cb2-19"><a href="#cb2-19" aria-hidden="true" tabindex="-1"></a>    clf.fit(bag[columns], bag[<span class="st">&quot;high_income&quot;</span>])</span>
<span id="cb2-20"><a href="#cb2-20" aria-hidden="true" tabindex="-1"></a>    </span>
<span id="cb2-21"><a href="#cb2-21" aria-hidden="true" tabindex="-1"></a>    <span class="co"># Using the model, make predictions on the test data</span></span>
<span id="cb2-22"><a href="#cb2-22" aria-hidden="true" tabindex="-1"></a>    predictions.append(clf.predict_proba(test[columns])[:,<span class="dv">1</span>])</span>
<span id="cb2-23"><a href="#cb2-23" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb2-24"><a href="#cb2-24" aria-hidden="true" tabindex="-1"></a>combined <span class="op">=</span> numpy.<span class="bu">sum</span>(predictions, axis<span class="op">=</span><span class="dv">0</span>) <span class="op">/</span> <span class="dv">10</span></span>
<span id="cb2-25"><a href="#cb2-25" aria-hidden="true" tabindex="-1"></a>rounded <span class="op">=</span> numpy.<span class="bu">round</span>(combined)</span>
<span id="cb2-26"><a href="#cb2-26" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb2-27"><a href="#cb2-27" aria-hidden="true" tabindex="-1"></a><span class="bu">print</span>(roc_auc_score(test[<span class="st">&quot;high_income&quot;</span>], rounded))</span></code></pre></div>
<p><br/><br/> <u><strong>Useful links</strong></u>: <br/><br/> <a
href="https://blog.clairvoyantsoft.com/entropy-information-gain-and-gini-index-the-crux-of-a-decision-tree-99d0cdc699f4">Entropy,
Information gain and Gini Index; the crux of a Decision Tree</a></p>
<p><a
href="https://medium.datadriveninvestor.com/decision-tree-adventures-2-explanation-of-decision-tree-classifier-parameters-84776f39a28">Decision
Tree Adventures 2 - Explanation of Decision Tree Classifier
Parameters</a></p>
