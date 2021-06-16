import pickle
#モデル読み込み
#model_file = 'model.sav'
#loaded_model = pickle.load(open(model_file, 'rb'))

if __name__ == '__main__':
    print(model.most_similar('United_States', topn=10))