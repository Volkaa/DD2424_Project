import numpy as np
import torch
from extract_data import get_train_list, split_sentence
import math as m
from utils import load_dico
from sklearn.model_selection import train_test_split
from data_loader_aspect import *

def get_all_labels(categories):
    matrix = np.zeros((len(categories),), dtype=np.int)
    for k, liste_sentence in enumerate(categories):
        polarities = []
        for category in liste_sentence:
            polarities.append(category[1])
        matrix[k] = 1+np.sign(polarities.count('positive')-polarities.count('negative'))
    return matrix

if __name__ == "__main__":
    texts_list, _, categories = get_train_list()
    labels = get_all_labels(categories)
    print(labels)
    print((labels==0).sum(), (labels==1).sum(), (labels==2).sum()) #results

    texts, categories, labels = get_all_combinations(texts_list, categories)
    print(labels)
    print((labels==0).sum(), (labels==1).sum(), (labels==2).sum()) #results
