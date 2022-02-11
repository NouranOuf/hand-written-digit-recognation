# libraries
from sklearn import svm
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats


# Load dataset -> huge array each index containing 3 arrays.
# An array that holds the patterns in a 1D array , another that holds the patterns in a 8x8 grid
# The last one contains the real number of the pattern
digits = load_digits(n_class=10)

# Seperating the arrays in different variables to use them easily
real_data = digits['data']
targets = digits['target']

# Descriptive statistics section:

# create a new array and copy the target elements
sorted_dataset = targets.copy()
#sort them to apply statistical methods
sorted_dataset.sort()
# use our descriptive statistics functions
mean = np.mean(sorted_dataset)
mode = stats.mode(sorted_dataset)
median = np.median(sorted_dataset)
sd = np.std(sorted_dataset)
var = np.var(sorted_dataset)

# show the results
print("-----------------------------")
print("The mean of the dataset is", mean)
# print("The mode of the dataset is", mode)
print("The median of the dataset is", median)
print("The standard deviation of the dataset is", sd)
print("The variance of the dataset is", var)
print("-----------------------------")

# display the mode in a Bar-graph
sns.countplot(sorted_dataset)
plt.show()

# Machine learning section


# load svm model

svm_model = svm.SVC(kernel='rbf', gamma=0.001, C=10)
svm_model.fit(real_data[:1440], targets[:1440])

# predict all data

predictions = svm_model.predict(real_data[1441:])
testing_data = targets[1441:]

# a function to let the user enter a number and predict what the number is
while True:
    entered_number = input("Enter a number between 1 and 357: ")
    plt.subplot(321)
    plt.imshow(digits.images[1440 + int(entered_number)], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()
    print("The real number is", targets[1440 + int(entered_number)])
    print("And the predicted number is", predictions[int(entered_number) - int(1)])
    print("--------------------------------------------------------------")
    choice = input("Enter 1 to try another number or 2 to exit: ")
    if int(choice) == int(1):
        continue
    elif int(choice) == int(2):
        break

    # print final report
    # show the final report of the estimations and the accuracy
print("-----------------------------------------------------------")
print("accuracy score is", accuracy_score(testing_data, predictions))
print(classification_report(testing_data, predictions))

# print the heatmap of the predicted and real numbers
cm = confusion_matrix(predictions, targets[1441:])
conf_matrix = pd.DataFrame(data=cm)
plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="YlGnBu")
plt.show()

