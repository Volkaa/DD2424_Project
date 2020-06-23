import numpy as np
import pandas as pd
import sys
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pickle

def get_train_list(verbose=0):
    tree = ET.parse('../dataset/train_raw.xml')
    root = tree.getroot()

    texts = []
    aspect_terms = []
    aspect_categories = []

    first = True
    state = 0
    for elem in root.iter():
        if elem.tag == 'text':
            texts.append(elem.text)

        if elem.tag == 'sentence':
            if first:
                term = []
                category = []
                first = False

            else :
                aspect_terms.append(term)
                aspect_categories.append(category)
                term = []
                category = []
                state = 0

        elif elem.tag == 'aspectTerm':
            state = 1
            attributes = elem.attrib
            attrib_term = attributes['term']
            attrib_polarity = attributes['polarity']
            term.append([attrib_term, attrib_polarity])

        elif elem.tag == 'aspectCategory':
            attributes = elem.attrib
            category.append([attributes['category'], attributes['polarity']])
            if state != 1:
                term = ['None']

    # Final terms and categories parameters are not appended in the lists yet
    aspect_terms.append(term)
    aspect_categories.append(category)

    # Veryfying data extraction
    if verbose !=0:
        for k in range(len(texts)):
            print('\n')
            print('Text : ', texts[k])
            print('Terms : ', aspect_terms[k])
            print('Categories : ', aspect_categories[k])

    return np.asarray(texts), np.asarray(aspect_terms), np.asarray(aspect_categories)

def get_test_list(verbose=0):
    tree = ET.parse('../dataset/test_raw.xml')
    root = tree.getroot()

    texts = []
    aspect_terms = []
    aspect_categories = []

    first = True
    state = 0
    for elem in root.iter():
        if elem.tag == 'text':
            texts.append(elem.text)

        if elem.tag == 'sentence':
            if first:
                term = []
                category = []
                first = False

            else :
                aspect_terms.append(term)
                aspect_categories.append(category)
                term = []
                category = []
                state = 0

        elif elem.tag == 'aspectTerm':
            state = 1
            attributes = elem.attrib
            attrib_term = attributes['term']
            attrib_polarity = attributes['polarity']
            term.append([attrib_term, attrib_polarity])

        elif elem.tag == 'aspectCategory':
            attributes = elem.attrib
            category.append([attributes['category'], attributes['polarity']])
            if state != 1:
                term = ['None']

    # Final terms and categories parameters are not appended in the lists yet
    aspect_terms.append(term)
    aspect_categories.append(category)

    # Veryfying data extraction
    if verbose !=0:
        for k in range(len(texts)):
            print('\n')
            print('Text : ', texts[k])
            print('Terms : ', aspect_terms[k])
            print('Categories : ', aspect_categories[k])

    return np.asarray(texts), np.asarray(aspect_terms), np.asarray(aspect_categories)

def split_sentence(sentence):
    return sentence.replace('!', ' ! ').replace(':', ' : ').replace('(', ' ( ').replace(')', ' ) ').replace('?', ' ? ').replace('[', ' [ ').replace(']', ' ] ').replace('.', ' . ').replace(',', ' , ').replace('/', ' / ').replace('-', ' - ').split()

def create_dico(filename):
    corpus = np.concatenate((get_train_list()[0],get_test_list()[0]),axis=-1)
    word_corpus = ['anecdotes']
    for sentence in corpus:
        temp_splitted = split_sentence(sentence)
        for word in temp_splitted:
            word_corpus.append(word.lower())

    uniques = list(set(word_corpus))
    vocab_size = len(uniques)

    # Creates and intializes the dictionnary and the embedding matrix
    emb_matrix = np.zeros((vocab_size+1, 300), dtype=np.float)
    emb_matrix[0:] = np.random.uniform(low=-0.01, high=0.01, size=300)
    dico = dict([])
    dico['<pad>']=0

    with open('../dataset/'+filename, 'rb') as f:
        idx = 1
        for k, l in tqdm(enumerate(f), total=2196017):
            splitted = l.decode().split(' ')
            word = splitted[0].lower()
            if word in uniques :
                emb_matrix[idx,:] = np.asarray(splitted[1:], dtype=np.float)
                dico[word]=idx
                idx+=1
                uniques.pop(uniques.index(word))

    print(f'{len(uniques)/vocab_size} of the words in the data is not in the '+filename +' corpus')

    for word in uniques:
        dico[word]=idx
        emb_matrix[idx:] = np.random.uniform(low=-0.01, high=0.01, size=300)
        idx+=1

    print(f'Veryfying that the size of the dico matches the size of the matrix: {vocab_size+1==len(dico)}.')
    print(f'\nVeryfying the 5 categories are in the dictionnary :')
    print(f"\tservice: {dico['service']}")
    print(f"\tfood: {dico['food']}")
    print(f"\tambience: {dico['ambience']}")
    print(f"\tanecdotes: {dico['anecdotes']}")
    print(f"\tprice: {dico['price']}")

    with open('../dataset/dico_'+filename[:-4]+'.pkl','wb') as f:
        pickle.dump(dico,f)
    print('Dictionnary successfully saved.')

    np.save('../dataset/weights_'+filename[:-4]+'.npy', emb_matrix)
    print('weight matrix successfully saved.')

    # # Check the saved files are the good ones
    print("\n\nChecking the written files are correct.")
    with open('../dataset/dico_'+filename[:-4]+'.pkl','rb') as f:
        print(pickle.load(f))
    return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please enter what you want to do : either 'train' or 'test'\n")
        print("You can also enter 'dico' to create the dictionnary related to the dataset.\n")
        quit()
    if sys.argv[1] == 'train' :
        texts, aspect_terms, aspect_categories = get_train_list()
    elif sys.argv[1]=='test' :
        texts, aspect_terms, aspect_categories = get_test_list()
    elif sys.argv[1] == 'dico':
        if len(sys.argv) < 3:
            print("Please enter the name of the file you want to do build the embedding dictionnary from.\n")
            quit()
        create_dico(sys.argv[2])
    else :
        print("Please enter what you want to do : either 'train' or 'test'\n")
        print("You can also enter 'dico' to create the dictionnary related to the dataset.\n")
        quit()
