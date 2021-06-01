from knock30 import analized_dicts as d

noun_chain = []

for i in range(len(d)):
	tmp = ''
	while 'pos' in d[i].keys() and d[i]['pos'] == '名詞':
		tmp += d[i]['surface']
		i += 1
	if tmp != '':
		noun_chain.append(tmp)
print(noun_chain)

###ANS###
#Q.2単語以上で連接なのでは？
#A.そう考えて解いた人が多いです！
#解答が重複してた！iはrangeに入った時に値がもとに戻ってしまう！
# とき直しは以下の通り。
# i = 0
# while i < len(d):
# 	tmp = ''
# 	while 'pos' in d[i].keys() and d[i]['pos'] == '名詞':
# 		tmp += d[i]['surface']
# 		i += 1
# 	if tmp != '':
# 		noun_chain.append(tmp)
# 	i += 1
# print(noun_chain)

