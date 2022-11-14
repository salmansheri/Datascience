from sklearn.datasets import  load_iris
import pandas as pd 
import matplotlib.pyplot  as plt 
import seaborn as sns 
import numpy as np 


iris = load_iris()
x = iris["data"]
y = iris["target"]
df = pd.DataFrame(x)
print(df)
corr_matrix = df.corr()
print(corr_matrix)
plt.figure(figsize=(8,6))
plt.title("correlation Heatmap of iris datasets")
a = sns.heatmap(corr_matrix, square=True, annot = True, fmt = ".2f", linecolor="black")
a.set_xticklabels(a.get_xticklabels(), rotation=30)
a.set_yticklabels(a.get_yticklabels(), rotation=30)
plt.show()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

print(to_drop)
df1 = df.drop(df.columns[to_drop], axis=1)
print(df1)