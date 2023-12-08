import pandas as pd

scores = pd.read_csv("scores.csv",header=None)
scores['avg'] = scores[[1,2,3,4,5]].mean(axis=1)
print(scores['avg'])