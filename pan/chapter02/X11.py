#Replace every occurrence of a tab character into a space. Confirm the result by using sed, tr, or expand command.
file = open('/users/kcnco/github/100knock2021/pan/chapter02/popular-names.txt','r')
filelines = file.read().replace('\t',' ')
file = open('/users/kcnco/github/100knock2021/pan/chapter02/popular-names.txt','w')
file.write(filelines)
