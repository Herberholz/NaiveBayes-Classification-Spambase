# Cody Herberholz
# CS445 HW4 Naive Bayes Classification

import pandas as pd
from numpy import std
from sklearn import model_selection
import math

data = pd.read_csv("spambase.data", header = None, index_col = 57)
x_train, x_test, y_train, y_test = model_selection.train_test_split(data, data.index.values, test_size=0.5)

# Create Probabilistic Model
##########################################################################
size = x_train.shape[0]
spam = 0
no_spam = 0
spam_sum = 0
no_spam_sum = 0
spam_mean = []
no_spam_mean = []
spam_std = []
no_spam_std = []
spam_pdf = []
no_spam_pdf = []
predictions = []
conf_matrix = pd.np.zeros((2, 2))

# initialize lists
for i in range(57):
    spam_mean.append(0)
    no_spam_mean.append(0)
    spam_std.append(0)
    no_spam_std.append(0)
    spam_pdf.append(0)
    no_spam_pdf.append(0)

# count how many spam and not spam
for i in range(size):
    predictions.append(0) #initalize prediction list
    if y_train[i] == 1:
        spam += 1
    else:
        no_spam += 1

# Calculates Mean and Standard Deviation on Training set
# Index 0 is feature 1 column, index 1 is feature 2 column, and so on...
for i in range(57):
    spam_mean[i] = sum(x_train[i][1]) / len(x_train[i][1])
    no_spam_mean[i] = sum(x_train[i][0]) / len(x_train[i][0])
    spam_std[i] = std(x_train[i][1])
    no_spam_std[i] = std(x_train[i][0])
    # print("Feature", i, ":  Spam: ", spam_mean[i])
    # print("Feature", i, ":NoSpam: ", no_spam_mean[i])

total_spam_prob = spam / size
total_no_spam_prob = no_spam / size

print("Training size: ", x_train.shape[0])
print("Spam probability: ", total_spam_prob)
print("Probability of no spam: ", total_no_spam_prob)
#####################################################################################

# Construct Gaussian Naive Bayes
########################################################################################
spam_prob = 0
no_spam_prob = 0

for i in range(size):
    feature = x_train.iloc[i]
    for j in range(57):
        if spam_std[j] != 0:
            spam_pdf[j] = 1 / (math.sqrt(2 * math.pi) * spam_std[j])
            spam_pdf[j] *= math.exp(-((feature[j] - spam_mean[j]) ** 2) / 2 * (spam_std[j] ** 2))
            spam_prob += math.log(spam_pdf[j])
        if no_spam_std[j] != 0:
            no_spam_pdf[j] = 1 / (math.sqrt(2 * math.pi) * no_spam_std[j])
            no_spam_pdf[j] *= math.exp(-((feature[j] - no_spam_mean[j]) ** 2) / 2 * (no_spam_std[j] ** 2))
            no_spam_prob += math.log(no_spam_pdf[j])
    spam_class = math.log(total_spam_prob) + spam_prob
    no_spam_class = math.log(total_no_spam_prob) + no_spam_prob

    if spam_class > no_spam_class:
        predictions[i] = 1
    else:
        predictions[i] = 0

    if predictions[i] == y_train[i]:
        if predictions[i] == 1:
            conf_matrix[1][1] += 1
        else:
            conf_matrix[0][0] += 1
    else:
        if predictions[i] == 1:
            conf_matrix[1][0] += 1
        else:
            conf_matrix[0][1] += 1

print(conf_matrix)

#########################################################################################
