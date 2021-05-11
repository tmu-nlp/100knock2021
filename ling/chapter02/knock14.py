
def head(file,n):
    for i in range(0,n):
        print(file.readline())
        
if __name__=='__main__':
    file=open('popular-names.txt','r')
    n=3
    head(file,n)
    file.close()
