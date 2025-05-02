import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class BERT4Rec(nn.Module):
  def __init__(self, num_items, hidden_size=128, num_heads=4, num_layers=2, max_seq_length=20, dropout=0.1):
      super().__init__()
      self.num_items = num_items
      self.hidden_size = hidden_size
      self.max_seq_length = max_seq_length

      # 1. Item + Positional Embeddings
      # +2 for [PAD]=0 and [MASK]=num_items+1
      self.item_embeddings = nn.Embedding(num_items + 2, hidden_size, padding_idx=0)
      self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)

      # 2. Transformer Encoder
      encoder_layer = TransformerEncoderLayer(
          d_model=hidden_size,
          nhead=num_heads,
          dim_feedforward=hidden_size * 4,
          dropout=dropout,
          activation='gelu',
          batch_first=True  
      )
      self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)

      # 3. LayerNorm + Dropout
      self.layer_norm = nn.LayerNorm(hidden_size)
      self.dropout = nn.Dropout(dropout)

      # 4. Output layer: predict item logits
      self.output_layer = nn.Linear(hidden_size, num_items + 2)  # Same as embedding size

  def forward(self, input_sequences):
      """
      input_sequences: (batch_size, seq_length)
      returns: logits of shape (batch_size, seq_length, num_items + 2)
      """
      batch_size, seq_length = input_sequences.shape

      positions = torch.arange(seq_length, device=input_sequences.device).unsqueeze(0).expand(batch_size, -1)

      # Embeddings
      item_embeds = self.item_embeddings(input_sequences)
      pos_embeds = self.position_embeddings(positions)
      embeddings = item_embeds + pos_embeds

      embeddings = self.layer_norm(embeddings)
      embeddings = self.dropout(embeddings)

      padding_mask = input_sequences.eq(0) 

      transformer_output = self.transformer_encoder(embeddings, src_key_padding_mask=padding_mask)

      logits = self.output_layer(transformer_output)
      return logits