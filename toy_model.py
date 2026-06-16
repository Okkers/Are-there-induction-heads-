import torch
import torch.nn as nn

class ToyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 50, d_model)) 

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        x = self.embed(x) + self.pos_embed[:, :T, :]
        x = x.transpose(0,1) 
        x = self.transformer(x)
        x = x[-1] 
        return self.fc(x)