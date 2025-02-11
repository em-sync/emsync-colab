import torch
import math as m
import numpy as np
import math
import torch.nn.functional as F


"""
MUSIC GENERATOR
"""

def set_dropout(model, rate):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = rate
        set_dropout(child, rate)
    return model

def build_model(args, load_config_dict=None):

    if load_config_dict is not None:
        args = load_config_dict
       
    config = {
        "vocab_size": args["vocab_size"], 
        "num_layer": args["n_layer"], 
        "num_head": args["n_head"], 
        "embedding_dim": args["d_model"], 
        "d_inner": args["d_inner"],
        "dropout": args["dropout"],
        "d_condition": args["d_condition"],
        "max_seq": args["context_len"],
        # "max_seq": 2048,
        "pad_token": 0,
        "attn_type": args["attn_type"],
        "threshold_n_instruments": args["threshold_n_instruments"],
        "conditioning": args["conditioning"],
        "use_chords": True,
        "chord_insertion": args["chord_insertion"],
        # "add_chord_to_every": args["add_chord_to_every"],
    }

    config['n_conditions'] = 2 if args["conditioning"] else 0
    from .midi_model \
            import MusicTransformerContinuousToken as MusicTransformer
    del config["d_condition"]

    model = MusicTransformer(**config)
    if load_config_dict is not None and args is not None:
        if args["overwrite_dropout"]:
            model = set_dropout(model, args["dropout"])
            rate = args["dropout"]
            print(f"Dropout rate changed to {rate}")
    return model, args

def generate_mask(x, pad_token=None, batch_first=True):

    batch_size = x.size(0)
    seq_len = x.size(1)

    subsequent_mask = torch.logical_not(torch.triu(torch.ones(seq_len, seq_len, device=x.device)).t()).unsqueeze(
        -1).repeat(1, 1, batch_size)
    pad_mask = x == pad_token
    if batch_first:
        pad_mask = pad_mask.t()
    mask = torch.logical_or(subsequent_mask, pad_mask)
    if batch_first:
        mask = mask.permute(2, 0, 1)
    return mask


class MusicTransformerContinuousToken(torch.nn.Module):
    def __init__(self, embedding_dim=None, d_inner=None, vocab_size=None, num_layer=None, num_head=None,
                 max_seq=None, dropout=None, pad_token=None, has_start_token=True, n_conditions=2,
                 attn_type=None, use_chords=False, chord_insertion="add", **kwargs):
        super().__init__()



        self.max_seq = max_seq
        self.num_layer = num_layer
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self.pad_token = pad_token
        self.has_start_token = has_start_token
        self.n_conditions = n_conditions
        self.use_chords = use_chords
        self.chord_insertion = chord_insertion


        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, 
                                            embedding_dim=self.embedding_dim,
                                            padding_idx=pad_token)

        # two vectors for two types of emotion (valence, energy/tempo)
        # just like token embedding
        
        self.fc_conditions = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(1, self.embedding_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.embedding_dim // 2, self.embedding_dim),
                torch.nn.Dropout(dropout)
            ) for _ in range(self.n_conditions)])
        
        self.null_conditions = torch.nn.ParameterList([torch.nn.Parameter(torch.rand((1, self.embedding_dim))) \
                                                        for _ in range(self.n_conditions)])
        
        if use_chords:
            if chord_insertion == "concat":
                # self.ff_times_to_chord = torch.nn.Linear(1, embedding_dim // 2)
                self.ff_times_to_chord = torch.nn.Sequential(
                    torch.nn.Linear(1, embedding_dim // 4),
                    torch.nn.ReLU(),
                    torch.nn.Linear(embedding_dim // 4, embedding_dim // 2),
                ) 
                self.positional_encoding = torch.nn.Parameter(torch.randn(1, max_seq, embedding_dim // 2))
            elif chord_insertion == "add":
                self.ff_times_to_chord = torch.nn.Linear(1, d_inner)
                self.positional_encoding = torch.nn.Parameter(torch.randn(1, max_seq, embedding_dim))
            else:
                self.positional_encoding = torch.nn.Parameter(torch.randn(1, max_seq, embedding_dim))

        self.enc_layers = torch.nn.ModuleList(
            [EncoderLayer(embedding_dim, d_inner, dropout, h=num_head, additional=False, max_seq=max_seq)
             for _ in range(num_layer)])
        self.dropout = torch.nn.Dropout(dropout)

        self.fc = torch.nn.Linear(self.embedding_dim, self.vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
        # self.pos_encoding.positional_embedding.data.uniform_(-initrange, initrange)
        self.positional_encoding.data.uniform_(-initrange, initrange)
        for i in range(len(self.fc_conditions)):
            for layer in self.fc_conditions[i]:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.data.uniform_(-initrange, initrange)
                    layer.bias.data.zero_()
            self.null_conditions[i].data.uniform_(-initrange, initrange)
        
        if self.use_chords:
            for layer in self.ff_times_to_chord:
                if isinstance(layer, torch.nn.Linear):
                    layer.weight.data.uniform_(-initrange, initrange)
                    layer.bias.data.zero_()
            
    def forward(self, x_tokens, conditions, times_to_chord):
        # takes batch first
        # x.shape = [batch_size, sequence_length]

        # embed input
        x = self.embedding(x_tokens)  # (batch_size, input_seq_len, d_model)
        x *= math.sqrt(self.embedding_dim)

        # pad input sequence to represent continuous emotion vectors
        # Padding to set the length. Not padding with zeros because we don't want to mask conditions
        x_tokens_padded = torch.nn.functional.pad(x_tokens, (self.n_conditions, 0), value=-1) 
        mask = generate_mask(x_tokens_padded, self.pad_token)

        # embed conditions one by one, using different linear layers,
        # just like token embedding
        conditions_encoded = []
        for i in range(self.n_conditions):
            condition = conditions[:, [i]]
            # Mask out NaN values from input tensors before passing them through the encoding layers
            nan_mask = condition.isnan()
            # Replace NaNs with zero to avoid NaNs in gradients
            masked = torch.where(nan_mask, torch.zeros_like(condition), condition)
            # Forward pass with masked values
            condition = self.fc_conditions[i](masked)   # encode
            # Then, insert the encoded null vectors into where NaNs were
            condition = torch.where(nan_mask.expand_as(condition), self.null_conditions[i], condition)
            conditions_encoded.append(condition)
        if self.n_conditions > 0:
            conditions_encoded = torch.stack(conditions_encoded, dim=1)
            # concatenate with conditions
            x = torch.cat((conditions_encoded, x), dim=1)
            # need to pad times_to_chord as well, to match sizes ("same" padding)
            padding = times_to_chord[:, [0]].repeat(1, 2)   
            times_to_chord = torch.cat((padding, times_to_chord), dim=1)

        batch_size = x.size(0)
        seq_len = x.size(1)
        positional_encoding = self.positional_encoding[:, :seq_len, :].repeat(batch_size, 1, 1)

        chord_to_encoder = None
        if self.use_chords:
            # Chord anticipation encoding
            chord_offset_encoding = self.ff_times_to_chord(8 - times_to_chord.unsqueeze(-1))
            if self.chord_insertion == "concat":
                positional_encoding = torch.cat((positional_encoding, chord_offset_encoding), 2)
            elif self.chord_insertion == "add_to_position":
                positional_encoding += chord_offset_encoding
            else:
                chord_to_encoder = chord_offset_encoding

        x += positional_encoding

        x = self.dropout(x)
        for i in range(len(self.enc_layers)):
            x = self.enc_layers[i](x, mask, chord_grid=chord_to_encoder)

        x = self.fc(x)
        return x

class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, d_inner, rate=0.1, h=16, additional=False, max_seq=2048):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.rga = RelativeGlobalAttention(h=h, d=d_model, max_seq=max_seq, add_emb=additional)

        self.FFN_pre = torch.nn.Linear(self.d_model, d_inner)
        self.FFN_suf = torch.nn.Linear(d_inner, self.d_model)

        self.layernorm1 = torch.nn.LayerNorm(self.d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(self.d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)

    def forward(self, x, mask=None, chord_grid=None, **kwargs):

        attn_out = self.rga([x,x,x], mask)

        attn_out = self.dropout1(attn_out)
        out1 = self.layernorm1(attn_out+x)

        ffn_out = F.relu(self.FFN_pre(out1))

        if chord_grid is not None:
            ffn_out = ffn_out + chord_grid

        ffn_out = self.FFN_suf(ffn_out)
        ffn_out = self.dropout2(ffn_out)
        out2 = self.layernorm2(out1+ffn_out)
        return out2


class RelativeGlobalAttention(torch.nn.Module):
    """
    from Music Transformer ( Huang et al, 2018 )
    [paper link](https://arxiv.org/pdf/1809.04281.pdf)
    """
    def __init__(self, h=4, d=256, add_emb=False, max_seq=2048, **kwargs):
        super().__init__()
        self.len_k = None
        self.max_seq = max_seq
        self.E = None
        self.h = h
        self.d = d
        self.dh = d // h
        self.Wq = torch.nn.Linear(self.d, self.d)
        self.Wk = torch.nn.Linear(self.d, self.d)
        self.Wv = torch.nn.Linear(self.d, self.d)
        self.fc = torch.nn.Linear(d, d)
        self.additional = add_emb
        self.E = torch.nn.Parameter(torch.randn([self.max_seq, int(self.dh)]))
        if self.additional:
            self.Radd = None

    def forward(self, inputs, mask=None, chord_grid=None, **kwargs):
        """
        :param inputs: a list of tensors. i.e) [Q, K, V]
        :param mask: mask tensor
        :param kwargs:
        :return: final tensor ( output of attention )
        """
        q = inputs[0]
        q = self.Wq(q)
        q = torch.reshape(q, (q.size(0), q.size(1), self.h, -1))
        q = q.permute(0, 2, 1, 3)  # batch, h, seq, dh

        k = inputs[1]
        k = self.Wk(k)
        if chord_grid != None:
            k += chord_grid
        k = torch.reshape(k, (k.size(0), k.size(1), self.h, -1))
        k = k.permute(0, 2, 1, 3)

        v = inputs[2]
        v = self.Wv(v)
        v = torch.reshape(v, (v.size(0), v.size(1), self.h, -1))
        v = v.permute(0, 2, 1, 3)

        self.len_k = k.size(2)
        self.len_q = q.size(2)

        E = self._get_left_embedding(self.len_q, self.len_k).to(q.device)
        QE = torch.einsum('bhld,md->bhlm', [q, E])
        QE = self._qe_masking(QE)
        Srel = self._skewing(QE)

        Kt = k.permute(0, 1, 3, 2)
        QKt = torch.matmul(q, Kt)
        logits = QKt + Srel
        logits = logits / math.sqrt(self.dh)

        if mask is not None:
            mask = mask.unsqueeze(1)
            new_mask = torch.zeros_like(mask, dtype=torch.float)
            new_mask.masked_fill_(mask, float("-inf"))
            mask = new_mask
            logits += mask

        attention_weights = F.softmax(logits, -1)
        attention = torch.matmul(attention_weights, v)

        out = attention.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), -1, self.d))

        out = self.fc(out)
        return out

    def _get_left_embedding(self, len_q, len_k):
        starting_point = max(0,self.max_seq-len_q)
        e = self.E[starting_point:,:]
        return e

    def _skewing(self, tensor: torch.Tensor):
        padded = F.pad(tensor, [1, 0, 0, 0, 0, 0, 0, 0])
        reshaped = torch.reshape(padded, shape=[padded.size(0), padded.size(1), padded.size(-1), padded.size(-2)])
        Srel = reshaped[:, :, 1:, :]
        if self.len_k > self.len_q:
            Srel = F.pad(Srel, [0, 0, 0, 0, 0, 0, 0, self.len_k-self.len_q])
        elif self.len_k < self.len_q:
            Srel = Srel[:, :, :, :self.len_k]

        return Srel

    @staticmethod
    def _qe_masking(qe):
        mask = sequence_mask(
            torch.arange(qe.size()[-1] - 1, qe.size()[-1] - qe.size()[-2] - 1, -1).to(qe.device),
            qe.size()[-1])
        mask = ~mask.to(mask.device)
        return mask.to(qe.dtype) * qe

def sequence_mask(length, max_length=None):
    """Tensorflow의 sequence_mask를 구현"""
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

