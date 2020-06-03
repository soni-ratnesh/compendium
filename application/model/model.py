import torch
from torch import nn
import dill
import random
import torch.nn.functional as F
import math
import spacy

class Encoder(nn.Module):

    def __init__(self, vocab, embeding_dim, encoder_hidden_dim, decoder_hidden_dim, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab, embeding_dim)
        self.rnn = nn.GRU(embeding_dim, encoder_hidden_dim, bidirectional=True)
        self.fc = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, text, text_len):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_len)
        packed_outputs, hidden = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        return outputs, hidden