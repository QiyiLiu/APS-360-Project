import torch
import torch.nn as nn
import torchvision.models as models

'''class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(RNN, self).__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, img, caption):
        # Look up the embedding
        self.length= len(caption)+1
        cap_emb = self.emb(caption)
        embedding= torch.cat((img, cap_emb), 0)

        # Set an initial hidden state and cell state
        #h0 = torch.zeros(1, img_emb.size(0), self.hidden_size)
        #c0 = torch.zeros(1, img_emb.size(0), self.hidden_size)

        # Forward propagate the LSTM
        lstm_out, _ = self.lstm(embedding.unsqueeze(1))

        # Pass the output of the last time step to the classifier
        out=self.fc(lstm_out.view(self.length, -1))

        return out

    def search(self, cnn_emb , length=30):
        ids = []
        inputs=cnn_emb
        for i in range(length):
            lstm_out, _ = self.lstm(inputs.unsqueeze(1), None)  # hiddens: (batch_size, 1, hidden_size)

            # one word at a time
            outputs = self.fc(lstm_out.squeeze(1))  # outputs:  (batch_size, vocab_size)
            cap = outputs.max(1)[1]  # predicted: (batch_size)
            ids.append(cap)
            inputs = self.emb(cap)  # inputs: (batch_size, embed_size)
            #inputs = inputs.unsqueeze(1)  # inputs: (batch_size, 1, embed_size)
        #sampled_ids = torch.stack(ids, 1)  # sampled_ids: (batch_size, max_seq_length)
        return ids

'''
import torch
import torch.nn as nn
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.word_embeddings.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, caption):
        seq_length = len(caption) + 1
        embeds = self.word_embeddings(caption)
        embeds = torch.cat((features, embeds), 0)
        lstm_out, _ = self.lstm(embeds.unsqueeze(1))
        out = self.linear(lstm_out.view(seq_length, -1))
        return out

    def search(self, cnn_out, seq_len=20):
        ip = cnn_out
        hidden = None
        ids_list = []
        for t in range(seq_len):
            lstm_out, hidden = self.lstm(ip.unsqueeze(1), hidden)
            # generating single word at a time
            linear_out = self.linear(lstm_out.squeeze(1))
            word_caption = linear_out.max(dim=1)[1]
            ids_list.append(word_caption)
            ip = self.word_embeddings(word_caption)
        return ids_list