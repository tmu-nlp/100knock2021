#「パトカー」＋「タクシー」の文字を先頭から交互に連結して文字列「パタトクカシーー」を得よ．

Input1 = "パトカー"
Input2 = "タクシー"
ans = ""

for i in range(len(Input1)):
    ans += Input1[i]
    ans += Input2[i]

print(ans)