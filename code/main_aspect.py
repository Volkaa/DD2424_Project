import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import pickle
import model as mod
import random
import torch.optim as optim
from data_loader_aspect import *
from extract_data import get_train_list, get_test_list, split_sentence
from sklearn.model_selection import train_test_split
from utils import load_dico, load_emb_weights
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from train_aspect import *

if __name__ == "__main__":
    # Initialize gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.get_device_name(device=device))
    torch.cuda.init()
    # Define hyperparameters
    BATCH_SIZE = 30
    embedding_dim = 300
    num_hidden_nodes = 300
    aspect_embedding_dim = 300
    hidden_dim = 300
    num_output_nodes = 3
    num_layers = 1
    bidirectional = False
    dropout = 0
    N_EPOCHS = 80
    pad_mode = 'start'
    L2_values = [0, 1e-8,1e-7,1e-6,1e-5,1e-4,1e-3]
    dico = load_dico('dico_glove.pkl')
    embed_weights = load_emb_weights('weights_glove.npy')

    ###########################################################################################################################################################
    # AE-LSTM
    scores = []
    LR = 1e-3
    AT = False
    AE = True
    L2_values = [0, 1e-9,1e-8,1e-7,1e-6]
    for LR in [1e-2, 1e-3]:
        for i, L2 in tqdm(enumerate(L2_values), total=len(L2_values)):
                for k in range(10):
                    lr = LR
                    # Loading data
                    texts_list, _, categories_list = get_train_list()
                    texts, categories, labels = get_all_combinations(texts_list, categories_list)

                    idx_train, idx_valid = make_balanced_split(labels)

                    texts_train, texts_valid = texts[idx_train], texts[idx_valid]
                    cat_train, cat_valid = categories[idx_train], categories[idx_valid]
                    labels_train, labels_valid = labels[idx_train], labels[idx_valid]

                    train_dataset = TrainDataset(texts_train, cat_train, labels_train, dico, batch_size=BATCH_SIZE)
                    train_loader = DataLoader(train_dataset)
                    valid_dataset = TestDataset(texts_valid, cat_valid, labels_valid, dico, batch_size=BATCH_SIZE)
                    valid_loader = DataLoader(valid_dataset)


                    # Instantiate the model
                    model = mod.AT_LSTM(embedding_dim, aspect_embedding_dim, hidden_dim,num_output_nodes, num_layers, embed_weights=embed_weights, at=AT, ae=AE, dropout=dropout)
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
                    test_texts_list, _, test_categories_list = get_test_list()
                    texts, categories, labels = get_all_combinations(test_texts_list, test_categories_list)
                    test_dataset = TestDataset(texts_valid, cat_valid, labels_valid, dico, batch_size=BATCH_SIZE)
                    test_loader = DataLoader(test_dataset)
                    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion)
                    print('\n\Final evaluation on the test set :')
                    print(f'\t Final Loss: {test_loss:.3f} |  Final Acc: {test_acc*100:.2f}% |  Final F1: {test_f1*100:.2f}%')
                    scores.append([test_loss, test_acc, test_f1])

    np.save('../models_performances/test_params_AE.npy', np.asarray(scores, dtype=np.float))


###########################################################################################################################################################
    # AT-LSTM
    scores = []
    LR = 1e-3
    AT = True
    AE = False
    L2_values = [0, 1e-9,1e-8,1e-7,1e-6]
    for LR in [1e-2, 1e-3]:
        for i, L2 in tqdm(enumerate(L2_values), total=len(L2_values)):
            for k in range(10):
                lr = LR
                # Loading data
                texts_list, _, categories_list = get_train_list()
                texts, categories, labels = get_all_combinations(texts_list, categories_list)

                idx_train, idx_valid = make_balanced_split(labels)

                texts_train, texts_valid = texts[idx_train], texts[idx_valid]
                cat_train, cat_valid = categories[idx_train], categories[idx_valid]
                labels_train, labels_valid = labels[idx_train], labels[idx_valid]

                train_dataset = TrainDataset(texts_train, cat_train, labels_train, dico, batch_size=BATCH_SIZE)
                train_loader = DataLoader(train_dataset)
                valid_dataset = TestDataset(texts_valid, cat_valid, labels_valid, dico, batch_size=BATCH_SIZE)
                valid_loader = DataLoader(valid_dataset)


                # Instantiate the model
                model = mod.AT_LSTM(embedding_dim, aspect_embedding_dim, hidden_dim,num_output_nodes, num_layers, embed_weights=embed_weights, at=AT, ae=AE, dropout=dropout)
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
                test_texts_list, _, test_categories_list = get_test_list()
                texts, categories, labels = get_all_combinations(test_texts_list, test_categories_list)
                test_dataset = TestDataset(texts_valid, cat_valid, labels_valid, dico, batch_size=BATCH_SIZE)
                test_loader = DataLoader(test_dataset)
                test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion)
                print('\n\Final evaluation on the test set :')
                print(f'\t Final Loss: {test_loss:.3f} |  Final Acc: {test_acc*100:.2f}% |  Final F1: {test_f1*100:.2f}%')
                scores.append([test_loss, test_acc, test_f1])

    np.save('../models_performances/test_params_AT.npy', np.asarray(scores, dtype=np.float))


###########################################################################################################################################################
    # ATAE-LSTM
    scores = []
    LR = 1e-3
    AT = True
    AE = True
    L2_values = [0, 1e-9,1e-8,1e-7,1e-6]
    for LR in [1e-2, 1e-3]:
        for i, L2 in tqdm(enumerate(L2_values), total=len(L2_values)):
            for k in range(10):
                lr = LR
                # Loading data
                texts_list, _, categories_list = get_train_list()
                texts, categories, labels = get_all_combinations(texts_list, categories_list)

                idx_train, idx_valid = make_balanced_split(labels)

                texts_train, texts_valid = texts[idx_train], texts[idx_valid]
                cat_train, cat_valid = categories[idx_train], categories[idx_valid]
                labels_train, labels_valid = labels[idx_train], labels[idx_valid]

                train_dataset = TrainDataset(texts_train, cat_train, labels_train, dico, batch_size=BATCH_SIZE)
                train_loader = DataLoader(train_dataset)
                valid_dataset = TestDataset(texts_valid, cat_valid, labels_valid, dico, batch_size=BATCH_SIZE)
                valid_loader = DataLoader(valid_dataset)


                # Instantiate the model
                model = mod.AT_LSTM(embedding_dim, aspect_embedding_dim, hidden_dim,num_output_nodes, num_layers, embed_weights=embed_weights, at=AT, ae=AE, dropout=dropout)
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
                test_texts_list, _, test_categories_list = get_test_list()
                texts, categories, labels = get_all_combinations(test_texts_list, test_categories_list)
                test_dataset = TestDataset(texts_valid, cat_valid, labels_valid, dico, batch_size=BATCH_SIZE)
                test_loader = DataLoader(test_dataset)
                test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion)
                print('\n\Final evaluation on the test set :')
                print(f'\t Final Loss: {test_loss:.3f} |  Final Acc: {test_acc*100:.2f}% |  Final F1: {test_f1*100:.2f}%')
                scores.append([test_loss, test_acc, test_f1])

    np.save('../models_performances/test_params_ATAE.npy', np.asarray(scores, dtype=np.float))
