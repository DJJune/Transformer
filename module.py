import torch.nn as nn
import torch as t
import torch.nn.functional as F
import math
import hyperparameters as hp
import numpy as np
import copy


class MultiheadAttention(nn.Module):
    """
    Multihead attention mechanism (dot attention)
    """

    def __init__(self, num_hidden_k):
        """
        :param num_hidden_k: dimension of hidden 
        """
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query, mask=None, query_mask=None):
        # Get attention score
        attn = t.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)

        # Masking to ignore padding (key side)
        if mask is not None:
            attn = attn.masked_fill(mask, -2 ** 32 + 1)
            attn = t.softmax(attn, dim=-1)
        else:
            attn = t.softmax(attn, dim=-1)

        # Masking to ignore padding (query side)
        if query_mask is not None:
            attn = attn * query_mask

        # Dropout
        # attn = self.attn_dropout(attn)

        # Get Context Vector
        result = t.bmm(attn, value)

        return result, attn


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=True, w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention(nn.Module):
    """
    Attention Network
    """

    def __init__(self, num_hidden, h=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads 
        """
        super(Attention, self).__init__()

        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h

        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)

        self.multihead = MultiheadAttention(self.num_hidden_per_attn)

        self.residual_dropout = nn.Dropout(p=0.1)

        self.final_linear = Linear(num_hidden * 2, num_hidden)

        self.layer_norm_1 = nn.LayerNorm(num_hidden)

    def forward(self, memory, decoder_input, mask=None, query_mask=None):

        batch_size = memory.size(0)
        seq_k = memory.size(1)
        seq_q = decoder_input.size(1)

        # Repeat masks h times
        if query_mask is not None:
            query_mask = query_mask.unsqueeze(-1).repeat(1, 1, seq_k)
            query_mask = query_mask.repeat(self.h, 1, 1)
        if mask is not None:
            mask = mask.repeat(self.h, 1, 1)

        # Make multihead
        key = self.key(memory).view(batch_size, seq_k,
                                    self.h, self.num_hidden_per_attn)
        value = self.value(memory).view(batch_size, seq_k,
                                        self.h, self.num_hidden_per_attn)
        query = self.query(decoder_input).view(
            batch_size, seq_q, self.h, self.num_hidden_per_attn)

        key = key.permute(2, 0, 1, 3).contiguous().view(-1,
                                                        seq_k, self.num_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).contiguous(
        ).view(-1, seq_k, self.num_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).contiguous(
        ).view(-1, seq_q, self.num_hidden_per_attn)

        # Get context vector
        result, attns = self.multihead(
            key, value, query, mask=mask, query_mask=query_mask)

        # Concatenate all multihead context vector
        result = result.view(self.h, batch_size, seq_q,
                             self.num_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).contiguous().view(
            batch_size, seq_q, -1)

        # Concatenate context vector with input (most important)
        result = t.cat([decoder_input, result], dim=-1)

        # Final linear
        result = self.final_linear(result)

        # Residual dropout & connection
        result = result + decoder_input

        # result = self.residual_dropout(result)

        # Layer normalization
        result = self.layer_norm_1(result)

        return result, attns


class FFN(nn.Module):
    """
    Positionwise Feed-Forward Network
    """

    def __init__(self, num_hidden):
        """
        :param num_hidden: dimension of hidden 
        """
        super(FFN, self).__init__()
        self.w_1 = Conv(num_hidden, num_hidden * 4,
                        kernel_size=1, w_init='relu')
        self.w_2 = Conv(num_hidden * 4, num_hidden, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, input_):
        # FFN Network
        x = input_.transpose(1, 2)
        x = self.w_2(t.relu(self.w_1(x)))
        x = x.transpose(1, 2)

        # residual connection
        x = x + input_

        # dropout
        # x = self.dropout(x)

        # layer normalization
        x = self.layer_norm(x)

        return x


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)
