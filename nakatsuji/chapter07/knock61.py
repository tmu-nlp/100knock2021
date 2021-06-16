import pickle
import numpy as np 

#モデル読み込み
#model_file = 'model.sav'
#oaded_model = pickle.load(open(model_file, 'rb'))

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

if __name__ == '__main__':

    v1 = np.array(model['United_States'])
    v2 = np.array(model['U.S.'])

    sim_cos = cos_sim(v1, v2)
    print(sim_cos)

    #0.7310775