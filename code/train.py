import torch
from torch import nn
import numpy as np
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

def count_parameters(model): #No. of trainable parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_accuracy(preds, y): #define metric
    top_class = preds.argmax(dim = -1)
    equals = top_class == y.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor))
    return accuracy

def train(model, iterator, optimizer, criterion):
	 #initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    #set the model in training phase
    model.train()
    y_true = []
    y_pred = []
    for k, (text, targets) in enumerate(iterator):
        #resets the gradients after every batch
        optimizer.zero_grad()

        #convert to 1D tensor
        predictions = model(text).squeeze()
        targets = targets.view(targets.size()[1]).long()

        #compute the loss
        loss = criterion(predictions, targets)

        #compute the binary accuracy
        acc = compute_accuracy(predictions, targets)

        #backpropage the loss and compute the gradients
        loss.backward()

        #update the weights
        optimizer.step()

        #loss and accuracy
        epoch_loss += loss.item()
        epoch_acc += acc.item()

        #saves predictions and targets for f1_score computation
        y_true+=targets.cpu().numpy().tolist()
        y_pred+=predictions.argmax(dim = -1).cpu().numpy().tolist()

    y_true=np.asarray(y_true)
    y_pred=np.asarray(y_pred)

    return epoch_loss / len(iterator), epoch_acc / len(iterator), f1_score(y_true, y_pred, average='weighted')


def evaluate(model, iterator, criterion):
    #initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    #deactivating dropout layers
    model.eval()

    #deactivates autograd
    with torch.no_grad():
        y_true = []
        y_pred = []
        for k, (text, targets) in enumerate(iterator):
            #convert to 1D tensor
            predictions = model(text).squeeze()
            targets = targets.view(targets.size()[1]).long()
            #compute the loss
            loss = criterion(predictions, targets)
            acc = compute_accuracy(predictions, targets)

            #keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()

            #saves predictions and targets for f1_score computation
            y_true+=targets.cpu().numpy().tolist()
            y_pred+=predictions.argmax(dim = -1).cpu().numpy().tolist()

        y_true=np.asarray(y_true)
        y_pred=np.asarray(y_pred)

    return epoch_loss / len(iterator), epoch_acc / len(iterator), f1_score(y_true, y_pred, average='weighted')

def update_learning_rate(accuracies, epoch, lr, cooldown, factor=np.sqrt(10), patience=3, min_lr=1e-5):
    idx_max = accuracies.index(max(accuracies))
    if epoch>=patience and cooldown>=patience and idx_max in [k for k in range(len(accuracies)-patience)]:
        new_lr = lr/factor
        if new_lr < min_lr:
            print(f'\nReducing learning rate to the minimum {min_lr} at epoch nb {epoch}.')
            return min_lr, 0
        else:
            print(f'\nReducing learning rate at epoch nb {epoch} to {new_lr}.')
            return new_lr, 0
    else :
        return lr, cooldown+1

def early_stopping(accuracies, epoch, patience=5):
    if epoch>=patience and accuracies.index(max(accuracies)) in [k for k in range(len(accuracies)-patience)]:
        print(f'\nTraining stopped due to early_stopping at epoch nb {epoch}.')
        return True
    else:
        return False


if __name__ == "__main__":
    # Initialize gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.get_device_name(device=device))
    torch.cuda.init()
    # Define hyperparameters
    BATCH_SIZE = 60
    embedding_dim = 300
    num_hidden_nodes = 300
    num_output_nodes = 3
    num_layers = 1
    bidirectional = False
    dropout = 0
    N_EPOCHS = 200
    pad_mode = 'start'
    lr = 1e-2
    L2 = 0
    es=True

    # Loading data
    texts_list, _, categories = get_train_list()
    labels = get_all_labels(categories)
    idx_train, idx_valid = make_balanced_split(labels)

    # idx_train, idx_valid = train_test_split([k for k in range(len(texts_list))], train_size=0.7)
    texts_train, texts_valid = texts_list[idx_train], texts_list[idx_valid]
    labels_train, labels_valid = labels[idx_train], labels[idx_valid]

    dico = load_dico('dico_glove.pkl')
    embed_weights = load_emb_weights('weights_glove.npy')

    train_dataset = TrainDataset(texts_train, labels_train, dico, batch_size=BATCH_SIZE, pad_mode=pad_mode)
    train_loader = DataLoader(train_dataset)
    valid_dataset = TestDataset(texts_valid, labels_valid, dico, batch_size=BATCH_SIZE, pad_mode=pad_mode)
    valid_loader = DataLoader(valid_dataset)


    # Instantiate the model
    model = mod.classifier(embedding_dim, num_hidden_nodes,num_output_nodes, num_layers, embed_weights=embed_weights, bidirectional=bidirectional, dropout=dropout)
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
        print(f'\epoch number : {epoch}')
        #train the model
        train_loss, train_acc, train_f1 = train(model, train_loader, optimizer, criterion)

        #evaluate the model
        valid_loss, valid_acc, valid_f1 = evaluate(model, valid_loader, criterion)

        #save the best model
        if train_acc > best_valid_acc:
            best_valid_acc = train_acc
            torch.save(model.state_dict(), '../models_checkpoints/saved.pt')

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% |  Val. F1: {valid_f1*100:.2f}%')

        # Append the valid_loss to keep track for the following features
        valid_accuracies.append(valid_acc)

        # Shuffling datasets
        train_dataset.shuffle()

        # # early stopping if no improvements in the last patience epochs
        if es:
            if early_stopping(valid_accuracies, epoch, patience=10):
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
    print('\n\Final evaluation on the test set :')
    print(f'\t Final Loss: {test_loss:.3f} |  Final Acc: {test_acc*100:.2f}% |  Final F1: {test_f1*100:.2f}%')
