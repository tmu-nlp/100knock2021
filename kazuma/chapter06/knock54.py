from knock52 import train

def knock54():
    lr, data, vectorizer= train()
    print("train", lr.score(data[0][0],data[0][1]))
    print("vaild", lr.score(data[1][0], data[1][1]))
    


if __name__ == "__main__":
    knock54()