from module import Linear, clones, FFN, Attention
from utils import get_positional_table, get_sinusoid_encoding_table
import hyperparameters as hp
import copy
from symbols import vocab
import torch.nn.functional as F
import torch.nn as nn
import torch as t


class Encoder(nn.Module):
    """
    Encoder Network
    """

    def __init__(self, embedding_size, num_block, heads=4, pretrain_embedding_model=None):
        """
        Args:
            embedding_size: dimension of embedding
            num_block:  number of blocks
            heads: number of heads ("embedding_size must be divisible by heads")
            pretrain_embedding_model(FloatTensor) : pretrain embedding model
        """
        super(Encoder, self).__init__()

        if pretrain_embedding_model:
            self.embed = nnn.Embedding.from_pretrained(
                pretrain_embedding_model, freeze=True)
        else:
            self.embed = nn.Embedding(
                len(vocab), embedding_size, padding_idx=0)

        # the max length of sequence should be less than 1024
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(1024, embedding_size, padding_idx=0),
                                                    freeze=True)
        self.pos_dropout = nn.Dropout(p=0.1)

        self.layers = clones(Attention(embedding_size, h=heads), num_block)
        self.ffns = clones(FFN(embedding_size), num_block)

    def forward(self, x, pos):

        # Get character mask
        if self.training:
            c_mask = pos.ne(0).type(t.float)
            mask = pos.eq(0).unsqueeze(1).repeat(1, x.size(1), 1)

        else:
            c_mask, mask = None, None

        # Encoder embedding layer
        x = self.embed(x)

        # Get positional embedding and add
        pos = self.pos_emb(pos)
        x = pos + x

        # Positional dropout
        x = self.pos_dropout(x)

        # Attention encoder-encoder
        attns = list()
        for layer, ffn in zip(self.layers, self.ffns):
            x, attn = layer(x, x, mask=mask, query_mask=c_mask)
            x = ffn(x)
            attns.append(attn)

        return x, c_mask, attns


class Decoder(nn.Module):
    """
    Decoder Network
    """

    def __init__(self, embedding_size, num_block, heads=4, pretrain_embedding_model=None):
        """
        Args:
            embedding_size: dimension of embedding
            num_block:  number of blocks
            heads: number of heads ("embedding_size must be divisible by heads")
            pretrain_embedding_model(FloatTensor) : pretrain embedding model
        """
        super(Decoder, self).__init__()

        if pretrain_embedding_model:
            self.embed = nnn.Embedding.from_pretrained(
                pretrain_embedding_model, freeze=True)
        else:
            self.embed = nn.Embedding(
                len(vocab), hp.embedding_size, padding_idx=0)

        # the max length of sequence should be less than 1024
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(1024, embedding_size, padding_idx=0),
                                                    freeze=True)
        self.pos_dropout = nn.Dropout(p=0.1)
        self.norm = Linear(embedding_size, embedding_size)

        self.selfattn_layers = clones(
            Attention(embedding_size, h=heads), num_block)
        self.dotattn_layers = clones(
            Attention(embedding_size, h=heads), num_block)
        self.ffns = clones(FFN(embedding_size), num_block)
        self.linear = Linear(embedding_size, hp.output_size)


    def forward(self, memory, decoder_input, c_mask, pos):
        batch_size = memory.size(0)
        decoder_len = decoder_input.size(1)

        # get decoder mask with triangular matrix
        if self.training:
            m_mask = pos.ne(0).type(t.float)
            mask = m_mask.eq(0).unsqueeze(1).repeat(1, decoder_len, 1)
            if next(self.parameters()).is_cuda:
                mask = mask + t.triu(t.ones(decoder_len, decoder_len).cuda(),
                                     diagonal=1).repeat(batch_size, 1, 1).byte()
            else:
                mask = mask + t.triu(t.ones(decoder_len, decoder_len),
                                     diagonal=1).repeat(batch_size, 1, 1).byte()
            mask = mask.gt(0)
            zero_mask = c_mask.eq(0).unsqueeze(-1).repeat(1, 1, decoder_len)
            zero_mask = zero_mask.transpose(1, 2)
        else:
            if next(self.parameters()).is_cuda:
                mask = t.triu(t.ones(decoder_len, decoder_len).cuda(),
                              diagonal=1).repeat(batch_size, 1, 1).byte()
            else:
                mask = t.triu(t.ones(decoder_len, decoder_len),
                              diagonal=1).repeat(batch_size, 1, 1).byte()
            mask = mask.gt(0)
            m_mask, zero_mask = None, None

        decoder_input = self.embed(decoder_input)

        # Centered position
        decoder_input = self.norm(decoder_input)

        # Get positional embedding and add
        pos = self.pos_emb(pos)
        decoder_input = pos + decoder_input

        # Positional dropout
        decoder_input = self.pos_dropout(decoder_input)

        # Attention decoder-decoder, encoder-decoder
        attn_dot_list = list()
        attn_dec_list = list()

        for selfattn, dotattn, ffn in zip(self.selfattn_layers, self.dotattn_layers, self.ffns):
            decoder_input, attn_dec = selfattn(
                decoder_input, decoder_input, mask=mask, query_mask=m_mask)
            decoder_input, attn_dot = dotattn(
                memory, decoder_input, mask=zero_mask, query_mask=m_mask)
            decoder_input = ffn(decoder_input)
            attn_dot_list.append(attn_dot)
            attn_dec_list.append(attn_dec)

        # linear projection towards vocab
        out = self.linear(decoder_input)

        return out, attn_dot_list, attn_dec_list


class Transformer(nn.Module):
    """
    Transformer Network
    """

    def __init__(self):
        super(Transformer, self).__init__()
        # the default number of attention heads is 4
        self.encoder = Encoder(hp.embedding_size, hp.num_block)
        self.decoder = Decoder(hp.embedding_size, hp.num_block)

    def forward(self, source_seq, target_seq_input, source_pos, target_pos):
        """
        Args:
            source_seq : vocab ids of source sequence
                         The padding part is zero 
                         The end character is 1
                         e.g.  [2,32,46,75,1,0,0,0,0]

            source_pos : postion information of source sequence.
                         The padding part is zero 
                         e.g. `np.arange(1, source_seq + 1)`
                              [1,2,3,4,5,0,0,0,0]

            target_seq_input : right-shifted vocab ids of target sequence. 
                               e.g. [0,2,32,46,75,1,0,0,0,0]
                               ** right-shift : np.concatenate([np.zeros([1,embedding_size], np.float32), targe[:-1,:]], axis=0)

            target_pos : postion information of target sequence. 
                         The padding part is zero 
                         e.g. `np.arange(1, target_seq + 1)`
                              [1,2,3,4,5,0,0,0,0]
        """
        memory, c_mask, attns_enc = self.encoder.forward(
            source_seq, pos=source_pos)

        # the memory is the contextual information from encoder
        output, attn_probs, attns_dec = self.decoder.forward(
            memory, target_seq_input, c_mask, pos=target_pos)

        return output, attn_probs, attns_enc, attns_dec
