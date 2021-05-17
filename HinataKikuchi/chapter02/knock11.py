path = './popular-names.txt'

with open(path) as file:
	buf = file.read()
	print(buf.replace('\t',' '))

###ANS###
# 今回の問題は書き込みなのでprintじゃないです
#write_linesやwriteで書きましょう