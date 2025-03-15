import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_dim, dropout_rate=0.1):
        super(TransformerLayer, self).__init__()
        
        self.multihead_attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout_rate)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_hidden_dim, embed_size)
        )
        
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        """
        x: Input tensor of shape (seq_len, batch_size, embed_size)
        mask: Optional mask tensor (seq_len, seq_len)
        """
        attn_output, _ = self.multihead_attention(x, x, x, attn_mask=mask)
        x = self.layer_norm_1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.layer_norm_2(x + self.dropout(ff_output))
        
        return x