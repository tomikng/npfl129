### Assignment: thyroid_competition
#### Date: Deadline: Oct 31, 7:59 a.m.
#### Points: 3 points+4 bonus

This assignment is a [competition task](https://ufal.mff.cuni.cz/courses/npfl129/2324-winter#competitions). Your goal
is to perform binary classification – given medical data with 15 binary and
6 real-valued attributes, predict whether thyroid is functioning normally or not.
The train set and test set consist of ~3.5k instances.

The [thyroid_competition.py](https://github.com/ufal/npfl129/tree/master/labs/03/thyroid_competition.py)
template shows how to load training data, downloading it if needed.
Furthermore, it shows how to save a trained estimator and how to load it during
prediction.

The performance of your system is measured using _accuracy_ of correctly
predicted examples and your goal is to achieve at least 96% accuracy.
Note that you can use any number of **generalized linear models from sklearn**
to solve this assignment (but no decision trees, MLPs, …).
