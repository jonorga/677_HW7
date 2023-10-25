###
### CS667 Data Science with Python, Homework 6, Jon Organ
###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
	+ "\nIf the F2 (skewness) value is greater than 2, it seems to be green. If the F3 (curtosis) value"
	+ "\nis greater than 1.4, its green. Finally if the F4 (entropy) value is greater than -1.19, then"
	+ "\n its green. The volatility doesn't seem to show any siginificant patterns")



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
print(confusion_matrix)
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
print("{:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format('TP', 'FP', 'TN', 'FN', 'accuracy', 'TPR', 'TNR'))
print("{:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(simple_TP, simple_FP, simple_TN, simple_FN,
	simple_accuracy, simple_TPR, simple_TNR))




