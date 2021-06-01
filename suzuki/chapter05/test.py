class myclass:
    h = []
    print("ああーん")
    def __init__(self):
        print("コンストラクタが呼び出されました")
        self.h.append("パイン")
        self.h.append("万華鏡")
        self.h.append("青梗菜")
    
    def __del__(self):
        print("いやーん")

    def sex(self):
        print("いくー")
        print(self.h)

inst = myclass()
print(inst.h)
inst.sex()