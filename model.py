import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class BERT4Rec(nn.Module):
    def __init__(
        self,
        num_items: int,
        hidden_size: int = 512,
        num_heads: int = 4,     
        num_layers: int = 6,
        max_seq_len: int = 20,
        dropout: float = 0.1
    ):
        super().__init__()
        self.pad_token = 0
        self.mask_token = num_items + 1
        self.vocab_size = num_items + 2
        self.hidden_size = hidden_size

        # Embeddings
        self.item_embeddings = nn.Embedding(
            self.vocab_size, 
            hidden_size, 
            padding_idx=self.pad_token
        )
        
        # Sinusoidal positional embeddings for sequences
        self.register_buffer(
            'position_embeddings',
            self._get_sinusoidal_encoding(max_seq_len, hidden_size)
        )

        # Custom Transformer with Gelu
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output Layer
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_layer = nn.Linear(hidden_size, self.vocab_size)

    def _get_sinusoidal_encoding(self, max_len, d_model):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()

        item_emb = self.item_embeddings(input_ids)
        pos_emb = self.position_embeddings[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        x = item_emb + pos_emb

        # Transformer
        pad_mask = input_ids.eq(self.pad_token)
        x = self.transformer(x, src_key_padding_mask=pad_mask)

        # Output
        x = self.layer_norm(x)
        return self.output_layer(x)

if __name__ == '__main__':
    # DEBUG Check: Sanity check
    num_items = 1000
    model = BERT4Rec(num_items=num_items)
    sample = torch.randint(0, num_items, (2, 20))
    out = model(sample)
    print(out.shape)  # Expected: (2, 20, num_items+2(1002))
