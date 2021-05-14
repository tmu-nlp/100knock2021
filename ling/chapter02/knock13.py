import sys
tmp=open("col3.txt",'w')
a=open('col1.txt','w')
b=open('col2.txt','w')

for line1,line2 in a,b:
    tmp.write(line1+"   "+line2)
tmp.close()
a.close()
b.close()