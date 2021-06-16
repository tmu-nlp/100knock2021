import numpy as np
import pandas as pd
from scipy.stats import spearmanr

df = pd.read_csv('combined.csv')
#for index, row in df.iterrows():
human = list(df['Human (mean)'])
sim = [model.similarity(row['Word 1'], row['Word 2']) for index, row in df.iterrows()]
correlation, pvalue = spearmanr(human, sim)
print(correlation)