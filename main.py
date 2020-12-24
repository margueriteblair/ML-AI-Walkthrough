# This is a sample Python script.
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn

print("Pandas:" + pd.__version__)
print("Numpy:" + np.__version__)
print("Seaborn:" + sns.__version__)
print("Scipy:" + scipy.__version__)
print("Sklearn:" + sklearn.__version__)


data = pd.read_csv('creditcard.csv')

print(data.columns) #this will tell us all of the columns
print(data.shape) #will print out (rows, columns), in this case (transactions, numColumns)
print(data.describe())

data = data.sample(frac=0.1, random_state=1)
#the above code reduces our dataset to a tenth of the size
print(data.shape)

#now we plot a histogram of each param
# data.hist(figsize=(20, 20))
# plt.show()
#created a bunch of historgrams that show the relationship for each of the columns in our dataset
#determine the acctual number of fraud cases indataset
Fraud = data[data['Class'] == 1]
Valid = data[data["Class"] == 0]

outlier_fraction = len(Fraud)/float(len(Valid))
print(outlier_fraction)
print("Fraud cases: {}".format(len(Fraud)))
print("Valid cases: {}".format(len(Valid)))

#we should build a correlation matrix, this will tell us if there's a strong correlation between things in our dataset
# correlation_matrix = data.corr()
# fig = plt.figure(figsize=(12,9))
# sns.heatmap(correlation_matrix, vmax=.8, square=True)
# plt.show()

columns = data.columns.tolist()
#we get all the columns from the dataframe

#filter the columns to remove the data that we don't want:
columns = [c for c in columns if c not in ['Class']]

#Store the variables that we'll be predicting
target = 'Class'

X = data[columns]
Y = data[target]

#then we print the shapes of X and Y
print("shapes: \n")
print(X.shape)
print(Y.shape)

#anomoly detection
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
#these are two common anomoly detection methods ^^

state = 1
#define outlier detection methods
classifiers = {
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,
                                        random_state=state),
    "Local Outlier Factor:": LocalOutlierFactor(
        n_neighbors=20,
        contamination=outlier_fraction
    )
}

#now we need to fit the model
n_outliers = len(Fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fix(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
        #we get 1 for inlier, -1 for outlier

    #reshape the values for current fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == 0] = 1

    n_errors = (y_pred != Y).sum()


    #now run the classification metrocs
    #about 30% of the time we'll detect fraud
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))





