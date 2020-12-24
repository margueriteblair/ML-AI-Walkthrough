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
data.hist(figsize=(20, 20))
plt.show()
