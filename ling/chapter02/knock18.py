import pandas as pd
df = pd.read_csv('popular-names.txt', delimiter='\t', header=None)
new = df[2].sort_values(ascending=False)
print (new)