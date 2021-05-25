string6="Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
s=string6.split()
dict04=dict()
    #take 1 character
dict04[s[0][0]]=1
for i in range(4,9):
    dict04[s[i][0]]=i+1
dict04[s[14][0]]=15
dict04[s[15][0]]=16
dict04[s[18][0]]=19
    #take 2 characters
for i in range(1,4):
    dict04[s[i][0:2]]=i+1
for i in range(9,14):
    dict04[s[i][0:2]]=i+1
dict04[s[16][0:2]]=17
dict04[s[17][0:2]]=18
dict04[s[19][0:2]]=20
print("[04]. "+str(dict04)+ "\n")