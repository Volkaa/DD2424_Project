from custom_layers import custom_word_embedding
from custom_layers import Attention
from utils import load_emb_weights
import torch
from torch import nn

class classifier(nn.Module):

    #define all the layers used in model
    def __init__(self, embedding_dim, hidden_dim, output_dim, n_layers, embed_weights,
                 bidirectional=False, glove=True, init=True, dropout=0):

        #Constructor
        super().__init__()
        self.bidirectional = bidirectional

        if glove:
            # Embedding layer using GloVe
            self.embedding = custom_word_embedding(embed_weights)
        else:
            # Embedding layer without GloVe
            self.embedding = nn.Embedding(embed_weights.shape[0], embed_weights.shape[1])

        # LSTM layer and initialization
        self.lstm = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)
        if init:
            for name, param in self.lstm.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)

        # Dense layer with initialization
        if self.bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim * 1, output_dim)
        if init:
            nn.init.xavier_normal_(self.fc.weight)
        #activation function
        #self.act = nn.Sigmoid()
        self.act = nn.Softmax(dim = 1)

    def forward(self, text, text_lengths=None):
        #text = [batch size,sent_length]
        text = text.view(text.size()[1], -1) # Remove the useless 1st axis
        embedded = self.embedding(text.long())
        #embedded = [batch size, sent_len, emb dim]
        embedded = embedded.float().cuda()
        #packed sequence
        #packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths,batch_first=True)
        #si = embedded.size()
        #embedded = embedded.view(si[1],si[2],si[3])
        packed_output, (hidden, cell) = self.lstm(embedded)

        #hidden = [batch size, num layers * num directions,hid dim]
        #cell = [batch size, num layers * num directions,hid dim]

        #concat the final forward and backward hidden state
        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)

        #hidden = [batch size, hid dim * num directions]
        dense_outputs=self.fc(hidden)

        #Final activation function
        outputs=self.act(dense_outputs)

        return outputs



class AT_LSTM(nn.Module):

    #define all the layers used in model
    def __init__(self, embedding_dim, aspect_embedding_dim, hidden_dim,
                 output_dim, n_layers, embed_weights, at=True, ae=False, dropout=0):

        #Constructor
        super().__init__()
        # ATAE ?
        self.ae = ae
        self.at = at
        self.embedding_dim= embedding_dim
        # Embedding layer using GloVe or fasttext
        self.embedding = custom_word_embedding(embed_weights)

        # Embedding layer using Glove for aspects
        self.aspects_embedding = custom_word_embedding(embed_weights)

        # Embedding layer without GloVe
        # self.embedding = nn.Embedding(emb_mat.shape[0], emb_mat.shape[1])

        # LSTM layer and initialization
        if self.ae:
            self.lstm = nn.LSTM(embedding_dim*2,
                               hidden_dim,
                               num_layers=n_layers,
                               bidirectional=False,
                               dropout=dropout,
                               batch_first=True)
        else:
            self.lstm = nn.LSTM(embedding_dim,
                                hidden_dim,
                                num_layers=n_layers,
                                bidirectional=False,
                                dropout=dropout,
                                batch_first=True)

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        # Attention layer with initialization
        if self.at:
            self.attention = Attention(aspect_embedding_dim, hidden_dim)
            self.attention.xavier_init()

        # Final dense layer with initialization
        self.fc = nn.Linear(embedding_dim, output_dim)
        nn.init.xavier_normal_(self.fc.weight)

        #activation function
        #self.act = nn.Sigmoid()
        self.act = nn.Softmax(dim = 1)

    def forward(self, inp, text_lengths=None):

        text = inp[0].view(inp[0].size()[1], -1) # Remove the useless 1st axis
        #text = [batch_size, sent_length]
        categories = inp[1].view(inp[1].size()[1]).long()       #categories = [batch_size]

        embedded = self.embedding(text.long())

        # ATAE
        if self.ae:
            embedded_input_aspect = self.aspects_embedding(categories)
            embedded_input_aspect = embedded_input_aspect.view(embedded_input_aspect.size()[0],1,self.embedding_dim)
            embedded_input_aspect = embedded_input_aspect.repeat(1,embedded.size()[1],1)
            embedded = torch.cat((embedded, embedded_input_aspect), -1)

        #packed sequence
        #packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths,batch_first=True)
        #si = embedded.size()
        #embedded = embedded.view(si[1],si[2],si[3])
        embedded = embedded.float().cuda()

        packed_output, (hidden, cell) = self.lstm(embedded)
        #packed_output = [batch_size, sent_length, hid_dim]
        #hidden = [batch size, num layers * num directions,hid dim]
        #cell = [batch size, num layers * num directions,hid dim]
        embedded_aspects = self.aspects_embedding(categories)
        embedded_aspects = embedded_aspects.float().cuda()
        #embedded_aspects = [batch_size, aspect_embedding_dim]

        if self.at:
            final_hidden = self.attention(embedded, embedded_aspects, packed_output)
        else:
            final_hidden = hidden
        #hidden = [batch size, hid dim * num directions]
        dense_outputs=self.fc(final_hidden)

        #Final activation function
        outputs=self.act(dense_outputs)

        return outputs
