path = './popular-names.txt'

print(sum(1 for _ in open(path)))

###ANS###
#Q.なぜか答えが同じにならない、1行多く出る？
#A.改行コードがwinとlinuxで違うのが原因！