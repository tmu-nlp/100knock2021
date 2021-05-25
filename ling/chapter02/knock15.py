def tail(file,n):
    x=0
    for line in file:
        if x>=2780-n:
            print(line)
        x+=1
        
if __name__=='__main__':
    file=open('popular-names.txt','r')
    n=3
    tail(file,n)
    file.close()
