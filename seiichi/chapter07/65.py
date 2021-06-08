import pandas as pd


df = pd.read_csv("./data/64.csv", header=None)
sem = df[~df[0].str.contains("gram")]
syn = df[df[0].str.contains("gram")]
print("意味的アナロジー：", (sem[4] == sem[5]).sum() / len(sem))
print("文法的アナロジー：", (syn[4] == syn[5]).sum() / len(syn))

"""
意味的アナロジー： 0.7308602999210734
文法的アナロジー： 0.7400468384074942
"""