import numpy as np
import torch
from extract_data import get_train_list, get_test_list, split_sentence
from torch.nn.utils.rnn import pad_sequence
import math as m
from torch.utils.data import Dataset, DataLoader
from utils import load_dico
from sklearn.model_selection import train_test_split

def toDevice(data):
    return (torch.from_numpy(data)).cuda()

def get_all_labels(categories):
    matrix = np.zeros((len(categories),), dtype=np.int)
    for k, liste_sentence in enumerate(categories):
        polarities = []
        for category in liste_sentence:
            polarities.append(category[1])
        matrix[k] = 1+np.sign(polarities.count('positive')-polarities.count('negative'))
    return matrix

def make_balanced_split(labels, validation_ratio=0.2, verbose=False):
    sort_indexes = labels.argsort()
    count0 = np.count_nonzero(labels==0)
    count1 = np.count_nonzero(labels==1)
    if verbose:
        print('Result should be 661, 669, 1711: ',count0, count1, labels.shape[0]-count0-count1)

    length = m.floor(validation_ratio*labels.shape[0])
    length = int(length-length%3)
    if verbose:
        print()
        print('Length should be a 3*x so the rest by /3 is 0: ', length%3)

    size = int(length/3)
    indexes0 = np.random.choice(sort_indexes[:count0], size, replace=False)
    indexes1 = np.random.choice(sort_indexes[count0:count1+count0], size, replace=False)
    indexes2 = np.random.choice(sort_indexes[count1+count0:], size, replace=False)
    indexes = np.concatenate((indexes0, indexes1, indexes2), axis=-1)

    if verbose:
        print('The mean should be 1 since it is balanced between 0,1,2: ', labels[indexes].mean())
        print()

    valid_indexes = indexes
    train_indexes = np.asarray([idx for idx in sort_indexes if idx not in valid_indexes], dtype=np.int)
    if verbose:
        print('The validation indexes should have length close to : ', m.floor(validation_ratio*labels.shape[0]), valid_indexes.shape[0])
        print('The training indexes should have length : ', labels.shape[0]-valid_indexes.shape[0], train_indexes.shape[0])
    return train_indexes, valid_indexes

def custom_padding(batch, pad_mode='start'):
    max_len = max([len(sentence) for sentence in batch])
    matrix = np.zeros((len(batch), max_len), dtype=np.int)
    for k, sentence in enumerate(batch):
        nb_zeros = max_len-len(sentence)
        if pad_mode=='start':
            matrix[k,:] = np.asarray([0]*nb_zeros+sentence, dtype=np.int)
        elif pad_mode=='end':
            matrix[k,:] = np.asarray(sentence+[0]*nb_zeros, dtype=np.int)
    return toDevice(matrix)

class TrainDataset(Dataset):
    def __init__(self, texts, labels, dico, batch_size=32, pad_mode='start'):
        self.batch_size=batch_size
        self.pad_mode = pad_mode
        self.dico=dico
        self.embedded_text=self.get_embedded_text(texts)

        self.idx0, self.idx1, self.idx2 = self.split_labels(labels)
        self.text0 = self.embedded_text[self.idx0]
        self.text1 = self.embedded_text[self.idx1]
        self.text2 = self.embedded_text[self.idx2]

        self.tb = self.batch_size//3 # tb stands for third of a batch
        self.n = self.text2.shape[0]//self.tb

        self.atext0, self.atext1 = self.reproduce()

    def reproduce(self):
        n0 = m.ceil(self.text2.shape[0]/self.text0.shape[0])
        n1 = m.ceil(self.text2.shape[0]/self.text1.shape[0])
        temp0 = [self.text0]*n0
        temp1 = [self.text1]*n1
        return np.concatenate(temp0,axis=-1), np.concatenate(temp1, axis=-1)


    def get_embedded_text(self, texts):
        embedded_text = []
        for sentence in texts:
            embedded_sentence = []
            for word in split_sentence(sentence):
                embedded_sentence.append(self.dico[word.lower()])
            embedded_text.append(embedded_sentence)
        return np.asarray(embedded_text)

    def split_labels(self,labels):
        return np.where(labels==0), np.where(labels==1), np.where(labels==2)

    def shuffle(self):
        self.text0 = np.random.choice(self.text0, self.text0.shape[0], replace=False)
        self.text1 = np.random.choice(self.text1, self.text1.shape[0], replace=False)
        self.text2 = np.random.choice(self.text2, self.text2.shape[0], replace=False)

        self.atext0, self.atext1 = self.reproduce()
        return None

    def get_batch(self, index):
        temp_text0 = self.atext0[index*self.tb:(index+1)*self.tb]
        temp_text1 = self.atext1[index*self.tb:(index+1)*self.tb]
        temp_text2 = self.text2[index*self.tb:(index+1)*self.tb]
        temp_text = np.concatenate((temp_text0, temp_text1, temp_text2), axis=-1)
        return custom_padding(temp_text, pad_mode=self.pad_mode)

    def get_labels(self, index):
        temp_label0 = np.zeros((self.tb,))
        temp_label1 = np.ones((self.tb,))
        temp_label2 = 2*np.ones((self.batch_size-2*self.tb, ))
        return toDevice(np.concatenate((temp_label0, temp_label1, temp_label2), axis=-1))

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.get_batch(index), self.get_labels(index)

class TestDataset(Dataset):
    def __init__(self, texts, labels, dico, batch_size=32, pad_mode='start'):
        self.batch_size=batch_size
        self.pad_mode = pad_mode
        self.texts = texts
        self.len_texts = len(self.texts)
        self.n = m.floor(self.len_texts/self.batch_size)
        self.dico=dico
        self.labels=labels
        self.embedded_text=[]
        for sentence in self.texts:
            embedded_sentence = []
            for word in split_sentence(sentence):
                embedded_sentence.append(self.dico[word.lower()])
            self.embedded_text.append(embedded_sentence)

    def get_batch(self, index):
        return custom_padding(self.embedded_text[index*self.batch_size: (index+1)*self.batch_size], pad_mode=self.pad_mode)

    def get_labels(self, index):
        return toDevice(self.labels[index*self.batch_size: (index+1)*self.batch_size])

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.get_batch(index), self.get_labels(index)

def get_keys(dico):
    matrix = np.zeros((len(dico),), dtype='object')
    for key in dico.keys():
        matrix[dico[key]]=key
    return matrix

def control(dico, sentences):
    sentences_npy = sentences.cpu().numpy()
    keys = get_keys(dico)
    sentences_final = []
    for sentence in sentences_npy:
        temp_sentence = []
        for embedding in sentence:
            temp_sentence.append(keys[embedding])
        sentences_final.append(temp_sentence)
    print(sentences_final)
    return None


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.get_device_name(device=device))
    torch.cuda.init()

    texts_list, _, categories = get_train_list()
    print('Testing the make_balanced_split function:')
    labels = get_all_labels(categories)
    idx_train, idx_valid = make_balanced_split(labels, verbose=True)

    # idx_train, idx_valid = train_test_split([k for k in range(len(texts_list))], train_size=0.7)
    texts_train, texts_valid = texts_list[idx_train], texts_list[idx_valid]

    labels_train, labels_valid = labels[idx_train], labels[idx_valid]

    dico = load_dico()
    train_dataset = TrainDataset(texts_train, labels_train, dico, batch_size=9)
    train_loader = DataLoader(train_dataset)
    valid_dataset = TestDataset(texts_valid, labels_valid, dico, batch_size=9)
    valid_loader = DataLoader(valid_dataset)

    print(f'expected {len(train_loader)} training batch')
    for k, (sentences, targets) in enumerate(train_loader):
        if k%1000 == 0:
            print(f'\nTraining batch number {k}')
            print(sentences)
            control(dico, sentences)
            print(targets)
    print(k+1)

    print(f'expected {len(valid_loader)} validation batch')
    for k, (sentences, targets) in enumerate(valid_loader):
        if k%10 == 0:
            print(f'\nValidation batch number {k}')
            # print(sentences)
            # control(dico, sentences)
            print(targets)
    print(k+1)
