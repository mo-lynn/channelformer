import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self, attention, c_attn_layer, c_n_heads, seq_len, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.c_attn_layer = c_attn_layer
        self.c_n_heads    = c_n_heads
        self.seq_len      = seq_len
        self.d_model      = d_model

        self.channel_projection = nn.Linear(c_n_heads * d_model, seq_len)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout    = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    
    def channel_scores(self, x):
        x = x.permute(0, 2, 1)
        # x: [batch_size, dimension/variate, seq_len/time]

        # attn: [batch_size, n_heads, dimension/variate, dimension/variate]
        attn = self.c_attn_layer(x, x, x, attn_mask=None)[1]
        
        '''
            attn: [B, H, D, D]
            split       -> ([B, 1, D, D], [B, 1, D, D], ... , [B, 1, D, D]) : A tuple with H elements
            cat         -> [B, 1, H*D, D]
            squeeze     -> [B, H*D, D]
            permute     -> [B, D, H*D]
            projection  -> [B, D, seq_len] -> [B, seq_len, D]
        '''
        return self.channel_projection(torch.cat(torch.split(attn, 1, dim=1), dim=2).squeeze(1).permute(0,2,1)).transpose(1,2)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)

        # add channel scores to y
        y = x = y + self.channel_scores(y) 

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm_layer = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        return x, attns
