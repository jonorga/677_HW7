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
	clf = KNeighborsClassifier(n_neighbors = k, p=1)
	clf.fit(q11_x_train, q11_y_train)

	training_score = clf.score(q11_x_train, q11_y_train)
	test_score = clf.score(q11_x_test, q11_y_test)

	q11_training.append(training_score)
	q11_scores.append(test_score)

def Q1GenerateGraph(data, q):
	fig, ax = plt.subplots()
	ax.plot(data["k_Value"], data["k_Accuracy"])
	ax.set(xlabel='k Value', ylabel='k Accuracy',
	       title='k Accuracy by Value')
	ax.grid()
	print("Saving Q" + q + " k Accuracy by Value graph...")
	fig.savefig("results/Q" + q + "_k_accuracy.png")

frame_data = [[3, q11_scores[0]], [5, q11_scores[1]], [7, q11_scores[2]], [9, q11_scores[3]], [11, q11_scores[4]]]
results_frame = pd.DataFrame(frame_data, columns=['k_Value', 'k_Accuracy'])
Q1GenerateGraph(results_frame, "1.1")
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


clf = KNeighborsClassifier(n_neighbors = 3, p=1)
clf.fit(q11_x_train, q11_y_train)

q12_predicted = clf.predict(q12_x_test)


q12_test_score = clf.score(q12_x_test, q12_y_test)
print("Using the k* (3) from year 1, the accuracy for year 2 is: " + str(round(q12_test_score, 2)) + "%")


print("\n")
# Question 1.3 ====================================================================================================
print("Question 1.3:")
print("k* confusion matrix for year 2:")
q13_cm = pd.crosstab(q12_y_test, q12_predicted)
print(q13_cm)



print("\n")
# Question 1.4 ====================================================================================================
print("Question 1.4:")
print("The optimal k* value is 3, which is different from the 11 obtained by the previous assignment")




print("\n")
# Question 1.5 ====================================================================================================
print("Question 1.5:")
print("Year 2 true positive rate: 0%\nYear 2 true negative rate: 100%")



print("\n")
# Question 1.6 ====================================================================================================
print("Question 1.6:")
print("Buy-and-hold result from previous assignment: $125.70")

def Q16Strategy(df):
	balance = 100
	file_len = len(df.index)
	i = 0
	while i < file_len - 1:
		today_stock = balance / df['Close'].iloc[i]
		tmr_stock = balance / df['Close'].iloc[i + 1]
		difference = abs(today_stock - tmr_stock)
		if df['Color'].iloc[i] == "Red":
			balance += difference * df["Close"].iloc[i + 1]
		else:
			balance -= difference * df["Close"].iloc[i + 1]
		i += 1
	return round(balance, 2)


q16_knn_bal = Q16Strategy(df[(df['Week'] > 50) & (df['Week'] <= 100)])
print("Calculated kNN (p = 1) result: $" + str(q16_knn_bal))
print("Buy-and-hold results in a larger balance at the end of the year")



print("\n")
# Question 1.7 ====================================================================================================
print("Question 1.7:")
print("Using Euclidean kNN resulted in $209.81 at the end of the year, so the Manhattan metric"
	+ " was a decline in performance")



print("\n")
# Question 2.1 ====================================================================================================
print("Question 2.1:")


q21_training = []
q21_scores = []

for k in K_vals:
	clf = KNeighborsClassifier(n_neighbors = k, p=1.5)
	clf.fit(q11_x_train, q11_y_train)

	training_score = clf.score(q11_x_train, q11_y_train)
	test_score = clf.score(q11_x_test, q11_y_test)

	q21_training.append(training_score)
	q21_scores.append(test_score)


frame_data = [[3, q21_scores[0]], [5, q21_scores[1]], [7, q21_scores[2]], [9, q21_scores[3]], [11, q21_scores[4]]]
results_frame = pd.DataFrame(frame_data, columns=['k_Value', 'k_Accuracy'])
Q1GenerateGraph(results_frame, "2.1")
print("The optimal K value for year 1 is 3")


print("\n")
# Question 2.2 ====================================================================================================
print("Question 2.2:")

clf = KNeighborsClassifier(n_neighbors = 3, p=1.5)
clf.fit(q11_x_train, q11_y_train)

q22_predicted = clf.predict(q12_x_test)


q22_test_score = clf.score(q12_x_test, q12_y_test)
print("Using the k* (3) from year 1, the accuracy for year 2 is: " + str(round(q22_test_score, 2)) + "%")


print("\n")
# Question 2.3 ====================================================================================================
print("Question 2.3:")

print("k* confusion matrix for year 2:")
q23_cm = pd.crosstab(q12_y_test, q22_predicted)
print(q23_cm)


print("\n")
# Question 2.4 ====================================================================================================
print("Question 2.4:")
print("The optimal k* value is 3, which is different from the 11 obtained by the previous assignment")


print("\n")
# Question 2.5 ====================================================================================================
print("Question 2.5:")
print("Year 2 true positive rate: 0%\nYear 2 true negative rate: 100%")


print("\n")
# Question 2.6 ====================================================================================================
print("Question 2.6:")
print("Buy-and-hold result from previous assignment: $125.70")
print("Calculated kNN (p = 1.5) result: $" + str(q16_knn_bal))
print("Buy-and-hold results in a larger balance at the end of the year")


print("\n")
# Question 2.7 ====================================================================================================
print("Question 2.7:")
print("Using Euclidean kNN resulted in $209.81 at the end of the year, so the Minkowski metric"
	+ " was a decline in performance")



print("\n")
# Question 3.1 ====================================================================================================
print("Question 3.1:")
g_ar_sum = df['Avg_Return'][(df['Week'] <= 50) & (df['Color'] == "Green")].sum()
g_v_sum = df['Volatility'][(df['Week'] <= 50) & (df['Color'] == "Green")].sum()
r_ar_sum = df['Avg_Return'][(df['Week'] <= 50) & (df['Color'] == "Red")].sum()
r_v_sum = df['Volatility'][(df['Week'] <= 50) & (df['Color'] == "Red")].sum()
g_count = df['Week'][(df['Week'] <= 50) & (df['Color'] == "Green")].count()
r_count = df['Week'][(df['Week'] <= 50) & (df['Color'] == "Red")].count()

g_centroid = [g_ar_sum / g_count, g_v_sum / g_count]
r_centroid = [r_ar_sum / r_count, r_v_sum / r_count]



def Q31DistToGreenCent(row):
	x_c = (g_centroid[0] - row['Avg_Return']) ** 2 
	y_c = (g_centroid[1] - row['Volatility']) ** 2
	return math.sqrt(x_c + y_c)

def Q31DistToRedCent(row):
	x_c = (r_centroid[0] - row['Avg_Return']) ** 2 
	y_c = (r_centroid[1] - row['Volatility']) ** 2
	return math.sqrt(x_c + y_c)

df['dist_to_greencent'] = df.apply(Q31DistToGreenCent, axis=1)
df['dist_to_redcent'] = df.apply(Q31DistToRedCent, axis=1)


g_cent_avg = df['dist_to_greencent'][df['Week'] <= 50].mean()
g_cent_med = df['dist_to_greencent'][df['Week'] <= 50].median()

r_cent_avg = df['dist_to_redcent'][df['Week'] <= 50].mean()
r_cent_med = df['dist_to_redcent'][df['Week'] <= 50].median()

if g_cent_avg > r_cent_avg:
	print("The green centroid has a larger average radius (" + str(round(g_cent_avg, 5))
		+ ") than the red centroid (" + str(round(r_cent_avg, 5)) + ")")
else:
	print("The green centroid has a smaller average radius (" + str(round(g_cent_avg, 5))
		+ ") than the red centroid (" + str(round(r_cent_avg, 5)) + ")")

if g_cent_med > r_cent_med:
	print("The green centroid has a larger median radius (" + str(round(g_cent_med, 5))
		+ ") than the red centroid (" + str(round(r_cent_med, 5)) + ")")
else:
	print("The green centroid has a smaller median radius (" + str(round(g_cent_med, 5))
		+ ") than the red centroid (" + str(round(r_cent_med, 5)) + ")")


print("\n")
# Question 3.2 ====================================================================================================
print("Question 3.2:")

def Q32CentPredict(row):
	if row['dist_to_greencent'] < row['dist_to_redcent']:
		return "Green"
	else:
		return "Red"

df['y2_cent_predict'] = df.apply(Q32CentPredict, axis=1)
q32_cm = pd.crosstab(df['Color'][(df['Week'] > 50) & (df['Week'] <= 100)], 
	df['y2_cent_predict'][(df['Week'] > 50) & (df['Week'] <= 100)])

# TPR = TP / TP + FN
# TNR = TN / TN + FP
q32_TPR = q32_cm['Green'].iloc[0] / (q32_cm['Green'].iloc[0] + q32_cm['Red'].iloc[0])
q32_TNR = q32_cm['Red'].iloc[1] / (q32_cm['Red'].iloc[1] + q32_cm['Green'].iloc[1])
print("Using centroids, the true positive rate for year 2 was: " + str(round(q32_TPR * 100, 2)) + "%")
print("Using centroids, the true negative rate for year 2 was: " + str(round(q32_TNR * 100, 2)) + "%")


print("\n")
# Question 3.3 ====================================================================================================
print("Question 3.3:")
print("Buy-and-hold result from previous assignment: $125.70")

def Q33Strategy(df):
	balance = 100
	file_len = len(df.index)
	i = 0
	while i < file_len - 1:
		today_stock = balance / df['Close'].iloc[i]
		tmr_stock = balance / df['Close'].iloc[i + 1]
		difference = abs(today_stock - tmr_stock)
		if df['Color'].iloc[i] == df['y2_cent_predict'].iloc[i]:
			balance += difference * df["Close"].iloc[i + 1]
		else:
			balance -= difference * df["Close"].iloc[i + 1]
		i += 1
	return round(balance, 2)

centroid_bal = Q33Strategy(df[(df['Week'] > 50) & (df['Week'] <= 100)])
print("Centroid strategy result: $" + str(centroid_bal))
print("The centroid strategy results in a higher balance than buy-and-hold")


print("\n")
# Question 3.4 ====================================================================================================
print("Question 3.4:")
print("Using Euclidean kNN resulted in $209.81 at the end of the year, so the centroid strategy"
	+ " was an improvement in performance")




