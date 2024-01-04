import torch
from torch import nn
import torch.nn.functional as F
import math

class Config():
    def __init__(self):
        self.ctx_length = 10 # Number of context tokens
        self.ctx_width = 80 # Number of spectograms
        
        # Attention layers configuration
        self.attn_heads = 6
        self.attn_layers = 4
        self.attn_features = 384

        # Encodings
        self.encoding_max_timescale = 10000

        # Internals
        self.bias = False
        self.dropout = 0.1
    

class PositionalEncoding(nn.Module):

    def __init__(self, config):
        super().__init__()

        # Positions [0, 1, 2, ..., config.ctx_length - 1]
        position = torch.arange(config.ctx_length).unsqueeze(1)
        
        # Divisor
        div_term = torch.exp(torch.arange(0, config.attn_features, 2) * (-math.log(config.encoding_max_timescale) / config.attn_features))

        # Sin/Cos matrix
        pe = torch.zeros(config.ctx_length, config.attn_features)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]
    

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Query/Key/Value in single tensor for speedup
        self.attention = nn.Linear(config.attn_features, 3 * config.attn_features, bias=config.bias)
        torch.nn.init.normal_(self.attention.weight, mean=0.0, std=0.02)

        # Output 
        self.output = nn.Linear(config.attn_features, config.attn_features, bias=config.bias)
        torch.nn.init.normal_(self.output.weight, mean=0.0, std=0.02)

        # Dropouts
        self.output_dropout = nn.Dropout(config.dropout)
    

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, context width

        # Calculate Query, Key and Value
        q, k, v  = self.attention(x).split(self.config.attn_features, dim=2)
        k = k.view(B, T, self.config.attn_heads, C // self.config.attn_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.config.attn_heads, C // self.config.attn_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.config.attn_heads, C // self.config.attn_heads).transpose(1, 2) # (B, nh, T, hs)

        # Dot product attention
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.config.dropout if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # Output
        y = self.output_dropout(self.output(y))
        
        return y

class AttentionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input = nn.Linear(config.attn_features, config.attn_features * 4, bias=config.bias)
        self.output = nn.Linear(config.attn_features * 4, config.attn_features, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        y = self.input(x)
        y = F.gelu(y)
        y = self.output(y)
        y = self.dropout(y)
        return y
        

class AttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Attention
        self.attention_ln = nn.LayerNorm(config.attn_features, bias=config.bias)
        self.attention = MultiHeadAttention(config)

        # MLP
        self.mlp_ln = nn.LayerNorm(config.attn_features, bias=config.bias)
        self.mlp = AttentionMLP(config)

    def forward(self, x):
        x = x + self.attention(self.attention_ln(x))
        x = x + self.mlp(self.mlp_ln(x))
        return x
    
class SuperVAD(nn.Module):
    def __init__(self, config = None):
        super().__init__()

        # Default config
        if config is None:
            config = Config()

        # Input normalization
        self.input_ln = nn.LayerNorm((config.ctx_width, config.ctx_length * 2), bias=False)

        # Convolutions
        self.conv1 = nn.Conv1d(config.ctx_width, config.attn_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(config.attn_features, config.attn_features, kernel_size=3, stride=2, padding=1)

        # Embedding
        self.encoding = PositionalEncoding(config)
        self.encoding_dropout = nn.Dropout(config.dropout)

        # Self Attention
        self.blocks_ln = nn.LayerNorm(config.attn_features, bias=config.bias)
        self.blocks = nn.ModuleList(
            [AttentionBlock(config) for _ in range(config.attn_layers)]
        )

        # Probability
        self.output = nn.Linear(config.ctx_length * config.attn_features, 1, bias=False)
        torch.nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        self.activation = torch.nn.Sigmoid()

        # Init weights for linear_layers
        self.apply(self._init_linear_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for block in self.blocks:
            torch.nn.init.normal_(block.mlp.output.weight, mean=0.0, std=0.02/math.sqrt(2 * config.attn_layers))        

    def _init_linear_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
    
    def tokenize(self, x):
        # Normalize input
        y = self.input_ln(x)

        # Convolutions
        y = F.gelu(self.conv1(y))
        y = F.gelu(self.conv2(y))
        y = y.permute(0, 2, 1) # LogMel has (batch, mels, ctx) instead of (batch, ctx, mels). But why not to do so before convolutions?

        return y

    def predict(self, x):
        # Apply positional embedding
        y = self.encoding(x)
        y = self.encoding_dropout(y)

        # Attention
        for block in self.blocks:
            y = block(y)
        y = self.blocks_ln(y)
        
        # Flatten features
        y = torch.flatten(y, start_dim=1)

        # Regression
        y = self.output(y)
        y = self.activation(y)

        # Return probability
        return y

    def forward(self, x):

        # Normalize input
        y = self.tokenize(x)

        # Predict
        y = self.predict(y)

        # Return probability
        return y