import pickle
import numpy as np

# def load_final_dico(path='../dataset/final_dico.pkl'):
#     with open(path,'rb') as f:
#         final_dico = pickle.load(f)
#     print('\nFinal dictionnary successfully loaded.')
#     return final_dico

def load_dico(filename):
    with open('../dataset/'+filename,'rb') as f:
        dico = pickle.load(f)
    print('\nDictionnary successfully loaded: ', filename)
    return dico

def load_emb_weights(filename):
    emb_matrix = np.load('../dataset/'+filename)
    print('\nPre-trained embedding weight matrix successfully loaded: ', filename)
    print()
    return emb_matrix
