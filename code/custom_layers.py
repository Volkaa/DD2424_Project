import torch
from torch import nn, from_numpy

def custom_word_embedding(weights):
    embedding = nn.Embedding.from_pretrained(from_numpy(weights).cuda(), freeze=False)
    return embedding

class Attention(nn.Module):

    def __init__(self, aspect_embedding_dim, hidden_dim):

        super(Attention, self).__init__()

        #W_h = [hidden_dim, hidden_dim]
        #W_v = [aspect_embedding_dim, aspect_embedding_dim]
        #W_p = [hidden_dim, hidden_dim]
        #W_x = [hidden_dim, hidden_dim]
        #w = [1, hidden_dim+aspect_embedding_dim]
        self.W_h = torch.nn.Parameter(data=torch.Tensor(hidden_dim, hidden_dim),requires_grad=True)
        self.W_v = torch.nn.Parameter(data=torch.Tensor(aspect_embedding_dim, aspect_embedding_dim),requires_grad=True)
        self.W_p = torch.nn.Parameter(data=torch.Tensor(hidden_dim, hidden_dim),requires_grad=True)
        self.W_x = torch.nn.Parameter(data=torch.Tensor(hidden_dim, hidden_dim),requires_grad=True)
        self.w = torch.nn.Parameter(data=torch.Tensor(1, hidden_dim + aspect_embedding_dim), requires_grad=True)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, text, embedded_aspects, packed_hidden_vectors):
        #text = [batch_size, sent_length, hidden_dim]
        #embedded_aspects = [batch_size, aspect_embedding_dim]
        #packed_hidden_vectors = [batch_size, sent_length, hidden_dim]

        batch_size = packed_hidden_vectors.size()[0]
        lin_hidden_vectors = torch.bmm(self.W_h.repeat(batch_size, 1, 1),
                                       torch.transpose(packed_hidden_vectors, 1, 2))
        #lin_hidden_vectors = [batch_size, hidden_dim, sent_length]

        lin_aspects = torch.bmm(self.W_v.repeat(batch_size, 1, 1),
                                embedded_aspects.view(embedded_aspects.size()[0], embedded_aspects.size()[1], 1))

        repeat_lin_aspects = lin_aspects.view(lin_aspects.size()[0],lin_aspects.size()[1]).repeat(lin_hidden_vectors.size()[2], 1, 1)
        #repeat_lin_aspects = [sent_length, batch_size, aspect_embedding_dim]

        M = torch.tanh(torch.cat((lin_hidden_vectors, repeat_lin_aspects.permute(1,2,0)), 1))
        #M = [batch_size, hidden_dim + aspect_embedding_dim, sent_length]

        input_mul = torch.bmm(self.w.repeat(batch_size, 1, 1), M)
        #input_mul = [batch_size, 1, sent_length]

        alpha = self.softmax(input_mul)
        #alpha = [batch_size, 1, sent_length]

        r = torch.bmm(alpha, packed_hidden_vectors).transpose(1,2)
        #r = [batch_size, hidden_dim, 1]

        lin_r = torch.bmm(self.W_p.repeat(batch_size, 1, 1),
                          r)
        #lin_r = [batch_size, hidden_dim, 1]

        lin_h_n = torch.bmm(self.W_x.repeat(batch_size, 1, 1),
                            packed_hidden_vectors[:,-1,:].view(batch_size, packed_hidden_vectors.size()[2], 1))
        #lin_h_n = [batch_size, hidden_dim, 1]

        h_star = torch.tanh(lin_r + lin_h_n)
        #h_start = [batch_size, hidden_dim, 1]

        return h_star.view(batch_size, h_star.size()[1])

    def xavier_init(self):
        nn.init.xavier_normal_(self.W_h)
        nn.init.xavier_normal_(self.W_v)
        nn.init.xavier_normal_(self.W_p)
        nn.init.xavier_normal_(self.W_x)
        nn.init.xavier_normal_(self.w)
