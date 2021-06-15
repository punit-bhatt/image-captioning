import torch
import torch.nn as nn
import torch.nn.init as I
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        # Removing the FC layer.
        modules = list(resnet.children())[:-1]
        
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)        
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)

        # The linear layer that maps from hidden state space back to vocab size
        # (plus loglikelihood or softmax)
        self.hidden2vocab = nn.Linear(hidden_size, vocab_size)
        self.__weights_init(self.hidden2vocab)
        
    def forward(self, features, captions):
        
        # training a lstm model - sequence info is just used for cell and hidden state.
        # input provided is the ith word in the sequence during time step i.
        
        batch_size = features.shape[0]
        
        # Removing end token.
        captions = captions[:, :-1]
        
        # Embedding the words in the caption -> batch_size, caption_len - 1, embed_size.
        captions_embed = self.word_embeddings(captions)
        
        # Appending features as the first input.
        inputs = torch.cat((features.view(batch_size, 1, -1), captions_embed), dim = 1)
                    
        out, hidden_f = self.lstm(inputs)
        
        out = self.hidden2vocab(out) 
        
        return out

    def sample(self, features, states=None, max_len=20, end_token=1):
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        
        batch_size = features.shape[0]
        outputs = []
        
        inputs = features
        hidden = (torch.zeros(1, batch_size, self.hidden_size).to(device), \
                  torch.zeros(1, batch_size, self.hidden_size).to(device))
        length = 0
        
        while length <= max_len:
            
            inputs, hidden = self.lstm(inputs, hidden)
            out = self.hidden2vocab(inputs)
            
            # Softmax scores
            out = F.softmax(out, dim = 2)
            
            out = torch.argmax(out, dim=2)
            
            for i in range(batch_size):
                word = torch.squeeze(out)
                outputs.append(word.tolist())
                
            length += 1
            
            if all([x[0] == end_token for x in out]):
                return outputs
            
            # Converting the word integer back to input (embed) for next timestep
            inputs = self.word_embeddings(out)
            
        return outputs

    def __weights_init(self, layer):
        
        I.xavier_uniform_(layer.weight)