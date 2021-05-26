# -*- coding: utf-8 -*-

import sklearn
import os
import sys
import numpy as np
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, model_selection, preprocessing
from features import extract_features
from util import slidingWindow, reorient, reset_vars
import pickle
import matplotlib.pyplot as plt


# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
data_file = 'letters.csv'
data = np.genfromtxt(data_file, delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Pre-processing
#
# -----------------------------------------------------------------------------

print("Reorienting accelerometer data...")
sys.stdout.flush()
reset_vars()
reoriented = np.asarray([reorient(data[i,1], data[i,2], data[i,3]) for i in range(len(data))])
reoriented_data_with_timestamps = np.append(data[:,0:1],reoriented,axis=1)
data = np.append(reoriented_data_with_timestamps, data[:,-1:], axis=1)

# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

window_size = 20
step_size = 20

n_samples = 1000
time_elapsed_seconds = (data[n_samples,0] - data[0,0]) / 1000
if (time_elapsed_seconds == 0):
    time_elapsed_seconds = 0.0001
sampling_rate = n_samples / time_elapsed_seconds

class_names = ["a", "b", "c", "d", "e"]

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

X = []
Y = []

for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    window = window_with_timestamp_and_label[:,1:-1]   
    feature_names, x = extract_features(window)
    X.append(x)
    Y.append(window_with_timestamp_and_label[10, -1])
    
X = np.asarray(X)
Y = np.asarray(Y)
n_features = len(X)
    
print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(Y)))
print("\n")
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------

cv = sklearn.model_selection.KFold(n_splits=10, random_state=None, shuffle=True)

tree = sklearn.tree.DecisionTreeClassifier(criterion="entropy", max_depth=12)

for train, test in cv.split(X):
    X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]

tree.fit(X_train, y_train)

# Print confusion matrix and Prediction vs Actual Reading graph

predicted_y_test = tree.predict(X_test)
confusion_matrix = metrics.confusion_matrix(y_test, predicted_y_test, labels=[0,1,2,3,4])
print("Activities predicted by the tree \n{}\n".format(predicted_y_test))
print("Activities from the collected data \n{}\n".format(y_test))
print("Confusion matrix \n{}\n".format(confusion_matrix))
predic=np.array(predicted_y_test)
num_a_p=(predic == 0).sum()
num_b_p=(predic == 1).sum()
num_c_p=(predic == 2).sum()
num_d_p=(predic == 3).sum()
num_e_p=(predic == 4).sum()
actual=np.array(y_test)
num_a=(actual == 0).sum()
num_b=(actual == 1).sum()
num_c=(actual == 2).sum()
num_d=(actual == 3).sum()
num_e=(actual == 4).sum()
xAxis=['a','a Pred','b','b Pred','c','c Pred','d','d Pred','e','e Pred']
yAxis=[num_a,num_a_p, num_b,num_b_p, num_c,num_c_p, num_d,num_d_p, num_e,num_e_p,]
plt.bar(xAxis,yAxis)
plt.title('Predicted vs Actual Readings')
plt.xlabel('Category')
plt.ylabel('Count')

plt.show()


# Print the accuracy, precision, and recall of the model
accuracy = np.sum(np.diag(confusion_matrix)) / float(np.sum(confusion_matrix))
print("Average accuracy of the model is {}\n".format(accuracy))
precision = np.asarray(np.nan_to_num(np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1).astype(float)))
print("Precision of the model is {}\n".format(precision))
recall = np.asarray(np.nan_to_num(np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0).astype(float)))
print("Recall of the model is {}\n".format(recall))

# %%---------------------------------------------------------------------------
#
#                Training the Decision Tree Classifier On Entire Dataset
#
# -----------------------------------------------------------------------------

print("Training decision tree classifier on entire dataset...")
tree.fit(X, Y)

total_accuracy = 0.0
total_precision = [0.0, 0.0, 0.0, 0.0]
total_recall = [0.0, 0.0, 0.0, 0.0]

for i, (train_index, test_index) in enumerate(cv.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    print("Fold {} : Training Random Forest classifier over {} points...".format(i, len(y_train)))
    sys.stdout.flush()
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    print("Evaluating classifier over {} points...".format(len(y_test)))
    y_pred = clf.predict(X_test)

    conf = metrics.confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4])

    accuracy = np.sum(np.diag(conf)) / float(np.sum(conf))
    precision = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=1).astype(float))
    recall = np.nan_to_num(np.diag(conf) / np.sum(conf, axis=0).astype(float))


# Print confusion matrix and Prediction vs Actual Reading graph

predicted_y_test = tree.predict(X_test)
confusion_matrix = conf
print("\nActivities predicted by the tree \n{}\n".format(predicted_y_test))
print("Activities from the collected data \n{}\n".format(y_test))
print("Confusion matrix \n{}\n".format(confusion_matrix))
predic=np.array(predicted_y_test)
num_a_p=(predic == 0).sum()
num_b_p=(predic == 1).sum()
num_c_p=(predic == 2).sum()
num_d_p=(predic == 3).sum()
num_e_p=(predic == 4).sum()
actual=np.array(y_test)
num_a=(actual == 0).sum()
num_b=(actual == 1).sum()
num_c=(actual == 2).sum()
num_d=(actual == 3).sum()
num_e=(actual == 4).sum()
xAxis=['a','a Pred','b','b Pred','c','c Pred','d','d Pred','e','e Pred']
yAxis=[num_a,num_a_p, num_b,num_b_p, num_c,num_c_p, num_d,num_d_p, num_e,num_e_p,]
plt.bar(xAxis,yAxis)
plt.title('Predicted vs Actual Readings')
plt.xlabel('Category')
plt.ylabel('Count')

plt.show()

# Print the accuracy, precision, and recall of the model

print("The average accuracy is {}\n".format(accuracy))  
print("The average precision is {}\n".format(precision))    
print("The average recall is {}\n".format(recall))  


# Save the decision tree visualization to disk
export_graphviz(tree, out_file='tree_letters.dot', feature_names = feature_names)

# Save the classifier to disk
with open('classifier.pickle', 'wb') as f:
    pickle.dump(tree, f)

print("activity-classification-train.py executed properly");