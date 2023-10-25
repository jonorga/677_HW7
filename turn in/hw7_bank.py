###
### CS667 Data Science with Python, Homework 6, Jon Organ
###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
pd.options.mode.chained_assignment = None  # default='warn'


# Question 1.1 ====================================================================================================
print("Question 1.1:")
df = pd.read_csv("data_banknote_authentication.csv")

def Q1AddColor(row):
	if row['class'] == 0:
		val = "Green"
	elif row['class'] == 1:
		val = "Red"
	return val

df['color'] = df.apply(Q1AddColor, axis=1)
print("CSV file read and column added for color...")

print("\n")
# Question 1.2 ====================================================================================================
print("Question 1.2:")

def Q12GenerateTable(data):
	temp_df = pd.DataFrame(data, columns=['class', 'F1 Mean', 'F1 Volatility', 'F2 Mean', 'F2 Volatility',
		'F3 Mean', 'F3 Volatility', 'F4 Mean', 'F4 Volatility'])
	temp_df = temp_df.round(2)
	fig, ax = plt.subplots()
	fig.patch.set_visible(False)
	ax.axis('off')
	ax.axis('tight')
	ax.table(cellText=temp_df.values, colLabels=temp_df.columns, loc='center').set_fontsize(18)

	print("Saving Q1.2 Table...\n")
	fig.savefig("results/Q1.2_Table.png", dpi = 300) #, dpi=1200


# All data for row one in one list, same for following rows
df_0 = df[df['class'] == 0]
df_1 = df[df['class'] == 1]
q12_data = [ ["0", df_0['variance'].mean(), df_0['variance'].std(), df_0['skewness'].mean(), df_0['skewness'].std(), 
	df_0['curtosis'].mean(), df_0['curtosis'].std(), df_0['entropy'].mean(), df_0['entropy'].std()],
	["1", df_1['variance'].mean(), df_1['variance'].std(), df_1['skewness'].mean(), df_1['skewness'].std(), 
	df_1['curtosis'].mean(), df_1['curtosis'].std(), df_1['entropy'].mean(), df_1['entropy'].std()],
	["all", df['variance'].mean(), df['variance'].std(), df['skewness'].mean(), df['skewness'].std(), 
	df['curtosis'].mean(), df['curtosis'].std(), df['entropy'].mean(), df['entropy'].std()] ]

Q12GenerateTable(q12_data)


print("\n")
# Question 1.3 ====================================================================================================
print("Question 1.3:")
print("It seems that generally if a bank note has a positive value for F1 (variance), then it's green."
	+ " If the F2 (skewness) value is greater than 2, it seems to be green. If the F3 (curtosis) value"
	+ " is greater than 1.4, its green. Finally if the F4 (entropy) value is greater than -1.19, then"
	+ " its green. The volatility doesn't seem to show any siginificant patterns")



print("\n")
# Question 2.1 ====================================================================================================
print("Question 2.1:")

df_train = df[(df.index < 343) | ((df.index >= 686) & (df.index < 1029)) ]
df_test = df[(df.index >= 1029) | ((df.index >= 343) & (df.index < 686)) ]


features = ["variance", "skewness", "curtosis", "entropy"]
pair_plot0 = sns.pairplot(df_train[features][df_train['class'] == 0])
pair_plot1 = sns.pairplot(df_train[features][df_train['class'] == 1])
fig0 = pair_plot0.fig
fig1 = pair_plot1.fig
fig0.savefig("results/good_bills.pdf")
fig1.savefig("results/fake_bills.pdf")
print("Pair plots generated...")


print("\n")
# Question 2.2 ====================================================================================================
print("Question 2.2:")
print("Rules for comparison:\nif F1 > 0 and F2 > 1.9 and F3 > 1.4 then its good, else fake")



print("\n")
# Question 2.3 ====================================================================================================
print("Question 2.3:")
def Q23SimplePredict(row):
	if row['variance'] > 0 and row['skewness'] > 1.9 and row['curtosis'] > 1.4:
		val = "Green"
	else:
		val = "Red"
	return val

df_test['simple_predict'] = df_test.apply(Q23SimplePredict, axis=1)
print("Simple classifier labels predicted...")


print("\n")
# Question 2.4 ====================================================================================================
print("Question 2.4:")
confusion_matrix = pd.crosstab(df_test['color'], df_test['simple_predict'])
simple_TP = confusion_matrix['Green'].iloc[0]
simple_FP = confusion_matrix['Green'].iloc[1]
simple_TN = confusion_matrix['Red'].iloc[1]
simple_FN = confusion_matrix['Red'].iloc[0]
simple_accuracy = round((simple_TP + simple_TN) / len(df_test.index), 2)
simple_TPR = round(simple_TP / (simple_TP + simple_FN), 2)
simple_TNR = round(simple_TN / (simple_TN + simple_FP), 2)

print("Simple classifier labels evaluators computed...")


print("\n")
# Question 2.5 ====================================================================================================
print("Question 2.5:")
print("Simple classifier table:")
print("{:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format('TP', 'FP', 'TN', 'FN', 'accuracy', 'TPR', 'TNR'))
print("{:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(simple_TP, simple_FP, simple_TN, simple_FN,
	simple_accuracy, simple_TPR, simple_TNR))



print("\n")
# Question 2.6 ====================================================================================================
print("Question 2.6:")
print("My simple classifier is perfect at identifying fake bills, and very bad at identifying real bills."
	+ " The overall accuracy is just barely better than flipping a coin (53%)")


print("\n")
# Question 3.1 ====================================================================================================
print("Question 3.1:")
q31_y = df['color']
q31_x = df.drop('color', axis = 1)
q31_x = q31_x.drop('class', axis = 1)

q31_x_train, q31_x_test, q31_y_train, q31_y_test = train_test_split(q31_x, q31_y, test_size = 0.5, random_state = 0)

K_vals = [3, 5, 7, 9, 11]
q31_training = []
q31_scores = []

for k in K_vals:
	clf = KNeighborsClassifier(n_neighbors = k)
	clf.fit(q31_x_train, q31_y_train)

	training_score = clf.score(q31_x_train, q31_y_train)
	test_score = clf.score(q31_x_test, q31_y_test)

	q31_training.append(training_score)
	q31_scores.append(test_score)

print("kNN accuracy computed...")


print("\n")
# Question 3.2 ====================================================================================================
print("Question 3.2:")

def Q32GenerateGraph(data):
	fig, ax = plt.subplots()
	ax.plot(data["k_Value"], data["k_Accuracy"])
	ax.set(xlabel='k Value', ylabel='k Accuracy',
	       title='k Accuracy by Value')
	ax.grid()
	print("Saving k Accuracy by Value graph...")
	fig.savefig("results/Q3.2_k_accuracy.png")

q32_frame_data = [[3, q31_scores[0]], [5, q31_scores[1]], [7, q31_scores[2]], [9, q31_scores[3]], [11, q31_scores[4]]]
q32_df = pd.DataFrame(q32_frame_data, columns=['k_Value', 'k_Accuracy'])
Q32GenerateGraph(q32_df)


print("\n")
# Question 3.3 ====================================================================================================
print("Question 3.3:")
print("kNN classifier table:")
print("{:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format('TP', 'FP', 'TN', 'FN', 'accuracy', 'TPR', 'TNR'))

clf = KNeighborsClassifier(n_neighbors = 7)
clf.fit(q31_x_train, q31_y_train)
q33_predicted = clf.predict(q31_x_test)

q33_cm = pd.crosstab(q31_y_test, q33_predicted)

kNN_TP = q33_cm['Green'].iloc[0]
kNN_FP = q33_cm['Green'].iloc[1]
kNN_TN = q33_cm['Red'].iloc[1]
kNN_FN = q33_cm['Red'].iloc[0]
kNN_accuracy = round(q31_scores[2], 2)
kNN_TPR = round(kNN_TP / (kNN_TP + kNN_FN), 2)
kNN_TNR = round(kNN_TN / (kNN_TN + kNN_FP), 2)
print("{:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(kNN_TP, kNN_FP, kNN_TN, kNN_FN,
	kNN_accuracy, kNN_TPR, kNN_TNR))


print("\n")
# Question 3.4 ====================================================================================================
print("Question 3.4:")
print("My kNN classifier was much more accurate in every way compared to my simple classifier")



print("\n")
# Question 3.5 ====================================================================================================
print("Question 3.5:")
# Last 4 of BUID: 4376
buid_bill = pd.DataFrame([[4, 3, 7, 6]], columns=["variance", "skewness", "curtosis", "entropy"])
q35_simple = Q23SimplePredict(buid_bill.iloc[0])
q35_kNN = clf.predict(buid_bill)	
print("Simple classifier prediction for BUID:", q35_simple, 
	"\nkNN classifier prediction for BUID:", q35_kNN[0])



print("\n")
# Question 4.1 ====================================================================================================
print("Question 4.1:")
# kNN, n = 7, with each of the features missing
q41_y = df['color']
q41_x = df.drop('color', axis = 1)
q41_x = q41_x.drop('class', axis = 1)

def Q41kNNMinusFeat(x_set, y_set, feature):
	x_set = x_set.drop(feature, axis = 1)
	q41_x_train, q41_x_test, q41_y_train, q41_y_test = train_test_split(x_set, y_set, 
		test_size = 0.5, random_state = 0)

	clf = KNeighborsClassifier(n_neighbors = 7)
	clf.fit(q41_x_train, q41_y_train)

	return clf.score(q41_x_test, q41_y_test)


q41_f1_acc = Q41kNNMinusFeat(q41_x, q41_y, "variance")
q41_f2_acc = Q41kNNMinusFeat(q41_x, q41_y, "skewness")
q41_f3_acc = Q41kNNMinusFeat(q41_x, q41_y, "curtosis")
q41_f4_acc = Q41kNNMinusFeat(q41_x, q41_y, "entropy")
print("kNN minus 1 feature accuracy computed for each feature")



print("\n")
# Question 4.2 ====================================================================================================
print("Question 4.2:")
print("kNN minus F1 accuracy: " + str(round(q41_f1_acc * 100, 2)) + "%"
	+ "\nkNN minus F2 accuracy: " + str(round(q41_f2_acc * 100, 2)) + "%"
	+ "\nkNN minus F3 accuracy: " + str(round(q41_f3_acc * 100, 2)) + "%"
	+ "\nkNN minus F4 accuracy: " + str(round(q41_f4_acc * 100, 2)) + "%"
	+ "\nNone of these were more accurate than all 4 working together")



print("\n")
# Question 4.3 ====================================================================================================
print("Question 4.3:")
print("Removing feature F1 (variance) caused the greatest accuracy loss")



print("\n")
# Question 4.4 ====================================================================================================
print("Question 4.4:")
print("Removing feature F4 (entropy) caused the least accuracy loss")



print("\n")
# Question 5.1 ====================================================================================================
print("Question 5.1:")
model = LogisticRegression(solver='liblinear', random_state=0)

q51_y = df['color']
q51_x = df.drop('color', axis = 1)
q51_x = q51_x.drop('class', axis = 1)
q51_x_train, q51_x_test, q51_y_train, q51_y_test = train_test_split(q51_x, q51_y, 
	test_size = 0.5, random_state = 0)
model.fit(q51_x_train, q51_y_train)
q51_acc = round(model.score(q51_x_test, q51_y_test), 2)
print("Logistic regression classifier accuracy: " + str(q51_acc) + "%")


print("\n")
# Question 5.2 ====================================================================================================
print("Question 5.2:")
print("Logistic regression classifier table:")
print("{:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format('TP', 'FP', 'TN', 'FN', 'accuracy', 'TPR', 'TNR'))

q52_predicted = model.predict(q51_x_test)
q52_cm = pd.crosstab(q51_y_test, q52_predicted)

LRC_TP = q52_cm['Green'].iloc[0]
LRC_FP = q52_cm['Green'].iloc[1]
LRC_TN = q52_cm['Red'].iloc[1]
LRC_FN = q52_cm['Red'].iloc[0]
LRC_accuracy = q51_acc
LRC_TPR = round(LRC_TP / (LRC_TP + LRC_FN), 2)
LRC_TNR = round(LRC_TN / (LRC_TN + LRC_FP), 2)
print("{:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(LRC_TP, LRC_FP, LRC_TN, LRC_FN,
	LRC_accuracy, LRC_TPR, LRC_TNR))


print("\n")
# Question 5.3 ====================================================================================================
print("Question 5.3:")
print("My logistic regression classifier was much better than my simple classifier in its accuracy"
	+ " and TPR, they both had the same TNR however")



print("\n")
# Question 5.4 ====================================================================================================
print("Question 5.4:")
print("My logistic regression classifier and my kNN classifier are comparable in terms of their"
	+ " performance. However, the kNN classifier was slightly more accurate as it had no false"
	+ " positives or negatives, whereas the logistic regression had 1 false positive, and 8 false"
	+ " negatives.")


print("\n")
# Question 5.5 ====================================================================================================
print("Question 5.5:")
q55_LRC = model.predict(buid_bill)
print("Logistic regression classifier prediction for BUID:", q55_LRC)
print("This is the same label as predicted by kNN")



print("\n")
# Question 6.1 ====================================================================================================
print("Question 6.1:")
q61_y = df['color']
q61_x = df.drop('color', axis = 1)
q61_x = q61_x.drop('class', axis = 1)

def Q61LRCMinusFeat(x_set, y_set, feature):
	x_set = x_set.drop(feature, axis = 1)
	q61_x_train, q61_x_test, q61_y_train, q61_y_test = train_test_split(x_set, y_set, 
		test_size = 0.5, random_state = 0)

	model = LogisticRegression(solver='liblinear', random_state=0)
	model.fit(q61_x_train, q61_y_train)

	return model.score(q61_x_test, q61_y_test)


q61_f1_acc = Q61LRCMinusFeat(q61_x, q61_y, "variance")
q61_f2_acc = Q61LRCMinusFeat(q61_x, q61_y, "skewness")
q61_f3_acc = Q61LRCMinusFeat(q61_x, q61_y, "curtosis")
q61_f4_acc = Q61LRCMinusFeat(q61_x, q61_y, "entropy")
print("Logistic regression classifier minus 1 feature accuracy computed for each feature")


print("\n")
# Question 6.2 ====================================================================================================
print("Question 6.2:")
print("LRC minus F1 accuracy: " + str(round(q61_f1_acc * 100, 2)) + "%"
	+ "\nLRC minus F2 accuracy: " + str(round(q61_f2_acc * 100, 2)) + "%"
	+ "\nLRC minus F3 accuracy: " + str(round(q61_f3_acc * 100, 2)) + "%"
	+ "\nLRC minus F4 accuracy: " + str(round(q61_f4_acc * 100, 2)) + "%"
	+ "\nNone of these were more accurate than all 4 working together")



print("\n")
# Question 6.3 ====================================================================================================
print("Question 6.3:")
print("For LRC, removing the F1 feature (variance), caused the greatest accuracy loss")



print("\n")
# Question 6.4 ====================================================================================================
print("Question 6.4:")
print("For LRC, removing the F4 feature (entropy), caused the least accuracy loss")



print("\n")
# Question 6.5 ====================================================================================================
print("Question 6.5:")
print("Comparing kNN with LRC after removing each feature one by one shows LRC had a much greater"
	+ " loss of accuracy by removing features F1, F2, and F3 when compared with kNN. The relative"
	+ " significance of each feature is comparable otherwise. For both, removing F4 had little effect")



