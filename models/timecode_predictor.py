import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_dim, dropout_rate=0.1):
        super(TransformerLayer, self).__init__()
        
        # Multi-Head Attention
        self.multihead_attention = nn.MultiheadAttention(embed_size, num_heads, dropout=dropout_rate)
        
        # Feed Forward Neural Network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_hidden_dim, embed_size)
        )
        
        # Layer Normalization
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        """
        x: Input tensor of shape (seq_len, batch_size, embed_size)
        mask: Optional mask tensor (seq_len, seq_len)
        """
        # Multi-Head Attention Layer
        attn_output, _ = self.multihead_attention(x, x, x, attn_mask=mask)
        
        # Add & Normalize
        x = self.layer_norm_1(x + self.dropout(attn_output))
        
        # Feed Forward Neural Network
        ff_output = self.feed_forward(x)
        
        # Add & Normalize
        x = self.layer_norm_2(x + self.dropout(ff_output))
        
        return x