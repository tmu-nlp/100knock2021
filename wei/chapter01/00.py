str = 'stressed'
print(str[::-1])


# str[i:j:s]是一种字符串切片处理语法，其中i代表切片对象str的起始位置，j表示取到j-1位置处的字符为止，s表示步长。
# 缺省值为0，字符串长度，1.
# 当s<0时，i缺省时，默认为-1;j缺省时，默认为-len(str)-1;
# 因此，str[::-1]表示从str最后一位开始，倒着往前走一步，一直取到str的第一个元素，相当于将str倒叙排列