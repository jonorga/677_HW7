###
### CS667 Data Science with Python, Homework 6, Jon Organ
###

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math

df = pd.read_csv("cmg_weeks.csv")

def Manhattan(X1, X2):
    return 0.1 * abs(X1[0] - X2[0]) + 0.9*abs(X1[1] - X2[1])

# Question 1.1 ====================================================================================================
print("Question 1.1:")

q11_y = df['Color'][df['Week'] <= 50]
q11_x = df[df['Week'] <= 50]
q11_x = q11_x.drop('Color', axis = 1)
q11_x = q11_x.drop('Week', axis = 1)
q11_x = q11_x.drop('Week_of_Month', axis = 1)
q11_x = q11_x.drop('Close', axis = 1)


q11_x_train, q11_x_test, q11_y_train, q11_y_test = train_test_split(q11_x, q11_y, test_size = 0.5, random_state = 0)

K_vals = [3, 5, 7, 9, 11]
q11_training = []
q11_scores = []

for k in K_vals:
	clf = KNeighborsClassifier(n_neighbors = k)
	clf.fit(q11_x_train, q11_y_train)

	training_score = clf.score(q11_x_train, q11_y_train)
	test_score = clf.score(q11_x_test, q11_y_test)

	q11_training.append(training_score)
	q11_scores.append(test_score)

def Q1GenerateGraph(data):
	fig, ax = plt.subplots()
	ax.plot(data["k_Value"], data["k_Accuracy"])
	ax.set(xlabel='k Value', ylabel='k Accuracy',
	       title='k Accuracy by Value')
	ax.grid()
	print("Saving k Accuracy by Value graph...")
	fig.savefig("results/Q1_k_accuracy.png")

frame_data = [[3, q11_scores[0]], [5, q11_scores[1]], [7, q11_scores[2]], [9, q11_scores[3]], [11, q11_scores[4]]]
results_frame = pd.DataFrame(frame_data, columns=['k_Value', 'k_Accuracy'])
Q1GenerateGraph(results_frame)
print("The optimal K value for year 1 is 3")


print("\n")
# Question 1.2 ====================================================================================================
print("Question 1.2:")

q12_y = df['Color'][(df['Week'] > 50) & (df['Week'] <= 100)]
q12_x = df[(df['Week'] > 50) & (df['Week'] <= 100)]
q12_x = q12_x.drop('Color', axis = 1)
q12_x = q12_x.drop('Week', axis = 1)
q12_x = q12_x.drop('Week_of_Month', axis = 1)
q12_x = q12_x.drop('Close', axis = 1)
q12_x_train, q12_x_test, q12_y_train, q12_y_test = train_test_split(q12_x, q12_y, test_size = 0.5, random_state = 0)


clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(q11_x_train, q11_y_train)

q12_predicted = clf.predict(q12_x_test)

q12_test_score = clf.score(q12_x_test, q12_y_test)
print("Using the k* (3) from year 1, the accuracy for year 2 is: " + str(round(q12_test_score, 2)) + "%")


print("\n")
# Question 1.3 ====================================================================================================
print("Question 1.3:")

# ! - ! - ! - ! TODO: Find out how to access P value and set to 1 for question 1

q13_cm = pd.crosstab(q12_y_test, q12_predicted)





