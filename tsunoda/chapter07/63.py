#chapter63
vec = model['Spain'] - model['madrid'] + model['Athens'] 
model.most_similar(positive=['Spain', 'Athens'], negative=['Madrid'], topn=10)
