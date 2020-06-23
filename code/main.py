import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import pickle
import model as mod
import random
import torch.optim as optim
from data_loader import *
from extract_data import get_train_list, get_test_list, split_sentence
from sklearn.model_selection import train_test_split
from utils import load_dico, load_emb_weights
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from train import *
from train import update_learning_rate, early_stopping

if __name__ == "__main__":
    # Initialize gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.get_device_name(device=device))
    torch.cuda.init()
    # Define hyperparameters
    BATCH_SIZE = 30
    embedding_dim = 300
    num_hidden_nodes = 300
    num_output_nodes = 3
    num_layers = 1
    bidirectional = False
    dropout = 0
    N_EPOCHS = 50
    pad_mode = 'start'
    dico = load_dico('dico_glove.pkl')
    embed_weights = load_emb_weights('weights_glove.npy')
    scores = []
    L2=0
    LR = 1e-2
    for config in tqdm(range(3)):
        if config==0:
            glove=False
            init=False
        if config==1:
            glove=False
            init=True
        if config==2:
            glove=True
            init=True

        for k in range(20):
            lr = LR
            # Loading data
            texts_list, _, categories = get_train_list()
            labels = get_all_labels(categories)
            idx_train, idx_valid = make_balanced_split(labels)

            # idx_train, idx_valid = train_test_split([k for k in range(len(texts_list))], train_size=0.7)
            texts_train, texts_valid = texts_list[idx_train], texts_list[idx_valid]
            labels_train, labels_valid = labels[idx_train], labels[idx_valid]

            train_dataset = TrainDataset(texts_train, labels_train, dico, batch_size=BATCH_SIZE, pad_mode=pad_mode)
            train_loader = DataLoader(train_dataset)
            valid_dataset = TestDataset(texts_valid, labels_valid, dico, batch_size=BATCH_SIZE, pad_mode=pad_mode)
            valid_loader = DataLoader(valid_dataset)


            # Instantiate the model
            model = mod.classifier(embedding_dim, num_hidden_nodes,num_output_nodes, num_layers, init=init, glove=glove, embed_weights=embed_weights, bidirectional=bidirectional, dropout=dropout)
            model.to(device)

            # Visualizing architecture
            print(model)

            #define optimizer and loss
            optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=L2) #Momentum seems not to be in pytorch adagrad ...

            criterion = nn.CrossEntropyLoss()

            # Push criterion to cuda if available
            criterion = criterion.to(device)

            best_valid_acc = 0

            #Initialize the history of the losses for the early stopping and the lr reducer
            valid_accuracies = []

            # Initialize cooldown for learning rate reducer
            cooldown = 0

            for epoch in range(N_EPOCHS):
                #train the model
                train_loss, train_acc, train_f1 = train(model, train_loader, optimizer, criterion)

                #evaluate the model
                valid_loss, valid_acc, valid_f1 = evaluate(model, valid_loader, criterion)

                #save the best model
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    torch.save(model.state_dict(), '../models_checkpoints/saved.pt')

                # Append the valid_loss to keep track for the following features
                valid_accuracies.append(valid_acc)

                # Shuffling datasets
                train_dataset.shuffle()

                # # early stopping if no improvements in the last patience epochs
                es = early_stopping(valid_accuracies, epoch, patience=10)
                if es:
                    break

                #reduce learning rate if no improvement in the last patience epochs
                lr, cooldown = update_learning_rate(valid_accuracies, epoch, lr, cooldown, patience=3)
                optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=L2)

            # Testing the best model
            model.load_state_dict(torch.load('../models_checkpoints/saved.pt'))
            model.eval()
            test_texts_list, _, test_categories = get_test_list()
            labels_test = get_all_labels(test_categories)
            test_dataset = TestDataset(test_texts_list, labels_test, dico, batch_size=BATCH_SIZE, pad_mode=pad_mode)
            test_loader = DataLoader(test_dataset)
            test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion)
            print(f'\t Final Loss: {test_loss:.3f} |  Final Acc: {test_acc*100:.2f}% |  Final F1: {test_f1*100:.2f}%')
            scores.append([test_loss, test_acc, test_f1])

    np.save('../models_performances/scores-base.npy', np.asarray(scores, dtype=np.float))
