from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

print(model.most_similar("United_States"))

'''
出力結果
[('Unites_States', 0.7877248525619507),
 ('Untied_States', 0.7541370987892151), 
 ('United_Sates', 0.7400724291801453), 
 ('U.S.', 0.7310774326324463), 
 ('theUnited_States', 0.6404393911361694), 
 ('America', 0.6178410053253174), 
 ('UnitedStates', 0.6167312264442444), 
 ('Europe', 0.6132988929748535), 
 ('countries', 0.6044804453849792), 
 ('Canada', 0.601906955242157)]
'''