###
### CS667 Data Science with Python, Homework 6, Jon Organ
###

import pandas as pd
import matplotlib.pyplot as plt


# Question 1 ====================================================================================================
print("Question 1:")
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
# Question 2 ====================================================================================================
print("Question 2:")

def Q2GenerateTable(data):
	temp_df = pd.DataFrame(data, columns=['class', 'F1 Mean', 'F1 Volatility', 'F2 Mean', 'F2 Volatility',
		'F3 Mean', 'F3 Volatility', 'F4 Mean', 'F4 Volatility'])
	temp_df = temp_df.round(2)
	fig, ax = plt.subplots()
	fig.patch.set_visible(False)
	ax.axis('off')
	ax.axis('tight')
	ax.table(cellText=temp_df.values, colLabels=temp_df.columns, loc='center').set_fontsize(18)

	print("Saving Q2 Table...\n")
	fig.savefig("Q2_Table.png", dpi = 300) #, dpi=1200


# All data for row one in one list, same for following rows
df_0 = df[df['class'] == 0]
df_1 = df[df['class'] == 1]
q2_data = [ ["0", df_0['variance'].mean(), df_0['variance'].std(), df_0['skewness'].mean(), df_0['skewness'].std(), 
	df_0['curtosis'].mean(), df_0['curtosis'].std(), df_0['entropy'].mean(), df_0['entropy'].std()],
	["1", df_1['variance'].mean(), df_1['variance'].std(), df_1['skewness'].mean(), df_1['skewness'].std(), 
	df_1['curtosis'].mean(), df_1['curtosis'].std(), df_1['entropy'].mean(), df_1['entropy'].std()],
	["all", df['variance'].mean(), df['variance'].std(), df['skewness'].mean(), df['skewness'].std(), 
	df['curtosis'].mean(), df['curtosis'].std(), df['entropy'].mean(), df['entropy'].std()] ]

Q2GenerateTable(q2_data)


print("\n")
# Question 3 ====================================================================================================
print("Question 3:")


