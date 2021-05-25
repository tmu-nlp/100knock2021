import sys
n=3
count=2780#行数
f=open('./popular-names.txt','r')

#print(type(f.readlines()))
'''
for line in f:
    count+=1
'''

linenum=count//n#各ファイルの行数
rest=count%n#余った行の数

for i in range(0,n):
    if(i!=n-1):
        a=open("./"+str(i),'w')
        for x in range (0,linenum):
            s=f.readline()
            a.write(s)
        a.close()    
    else:
        #最後のファイルに余った行を書き込む
        a=open("./"+str(i),'w')
        for x in range (0,linenum+rest):
            s=f.readline()
            a.write(s)
        a.close()
        
f.close()
