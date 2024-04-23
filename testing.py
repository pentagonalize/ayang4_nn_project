import torch
import math
import random
import encoder
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--train_length', dest='train_length', type=int, default=100)
ap.add_argument('--test_length', dest='test_length', type=int, default=100)
ap.add_argument('--epochs', dest='epochs', type=int, default=100)
ap.add_argument('--steps', dest='steps', type=int, default=100)
ap.add_argument('--big', dest='big', type=float, default=1.)
ap.add_argument('--perturb', dest='perturb', type=float, default=0, help='randomly perturb parameters')
ap.add_argument('--train', dest='train', action='store_true', default=False)
args = ap.parse_args()

log_sigmoid = torch.nn.LogSigmoid()

class PositionEncoding(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, n):
        zero = torch.zeros(n)
        pos = torch.arange(0, n).to(torch.float)
        pe = torch.stack([zero]*3 +
                         [pos / n,
                          torch.cos(pos*math.pi)] +
                         [zero]*5,
                         dim=1)
        return pe

class FirstLayer(torch.nn.TransformerEncoderLayer):
    def __init__(self):
        super().__init__(10, 2, 3, dropout=0.)
        self.self_attn.in_proj_weight = torch.nn.Parameter(torch.tensor(
            # First head attends to all symbols,
            # second head does nothing.
            # W^Q
            [[0]*10]*10 +
            # W^K
            [[0]*10]*10 +
            # W^V
            [[0,1,0,0,0,0,0,0,0,0],   # count 1s  (k)
             [0,0,1,0,0,0,0,0,0,0]]+   # count CLS (1)
            [[0]*10]*8,
            dtype=torch.float))

        self.self_attn.in_proj_bias = torch.nn.Parameter(torch.zeros(30))

        self.self_attn.out_proj.weight = torch.nn.Parameter(torch.tensor(
            # W^O
            [[0]*10]*5 +
            [[1,0,0,0,0,0,0,0,0,0],   # put new values into dims 5-6
             [0,1,0,0,0,0,0,0,0,0]] +
            [[0]*10]*3,
            dtype=torch.float))
        self.self_attn.out_proj.bias = torch.nn.Parameter(torch.zeros(10))

        self.linear1.weight = torch.nn.Parameter(torch.tensor([
            [0,0,0,-1,0,1,-1,0,0,0],  # k-i-1
            [0,0,0,-1,0,1, 0,0,0,0],  # k-i
            [0,0,0,-1,0,1, 1,0,0,0],  # k-i+1
        ], dtype=torch.float))
        self.linear1.bias = torch.nn.Parameter(torch.zeros(3))
        self.linear2.weight = torch.nn.Parameter(torch.tensor(
            [[0, 0, 0]]*7 +
            [[1,-2, 1],  # put I[i=c1] in dim 7
             [0, 0, 0],
             [0, 0, 0]], 
            dtype=torch.float))
        self.linear2.bias = torch.nn.Parameter(torch.zeros(10))
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src

class SecondLayer(torch.nn.TransformerEncoderLayer):
    def __init__(self):
        super().__init__(10, 2, 3, dropout=0.)
        self.self_attn.in_proj_weight = torch.nn.Parameter(torch.tensor(
            # W^Q
            # Heads 1 and 2 attend from CLS
            [[0,0,args.big,0,0,0,0,0,0,0]] +
            [[0]*10]*4 +
            [[0,0,args.big,0,0,0,0,0,0,0]] +
            [[0]*10]*4 +
            # W^K
            # Head 1 attends to odd positions
            [[0,0,0,0, 1,0,0,0,0,0]] +
            [[0]*10]*4 +
            # Head 2 attends to even positions
            [[0,0,0,0,-1,0,0,0,0,0]] +
            [[0]*10]*4 +
            # W^V
            # Heads 1 and 2 average dim 7
            [[0,0,0,0,0,0,0,1,0,0]] +
            [[0]*10]*4 +
            [[0,0,0,0,0,0,0,1,0,0]] +
            [[0]*10]*4,
            dtype=torch.float))

        self.self_attn.in_proj_bias = torch.nn.Parameter(torch.zeros(30))

        self.self_attn.out_proj.weight = torch.nn.Parameter(torch.tensor(
            # W^O
            # Even positions minus odd positions
            # Place in dim 8
            [[0]*10]*8 +
            [[-1,0,0,0,0,1,0,0,0,0],
             [0,0,0,0,0,0,0,0,0,0]],
            dtype=torch.float))
        self.self_attn.out_proj.bias = torch.nn.Parameter(torch.zeros(10))

        self.linear1.weight = torch.nn.Parameter(torch.zeros(3,10))
        self.linear1.bias = torch.nn.Parameter(torch.zeros(3))
        self.linear2.weight = torch.nn.Parameter(torch.zeros(10,3))
        self.linear2.bias = torch.nn.Parameter(torch.zeros(10))

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=None):
        q = src
        v = src
        src2 = self.self_attn(q, src, v, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src

class MyTransformerEncoder(torch.nn.TransformerEncoder):
    def __init__(self):
        torch.nn.Module.__init__(self)

        self.layers = torch.nn.ModuleList([
            FirstLayer(),
            SecondLayer(),
        ])
        self.num_layers = len(self.layers)
        self.norm = None

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embedding = torch.eye(3, 10)
        self.pos_encoding = PositionEncoding()
        self.encoder = MyTransformerEncoder()
        self.output_layer = torch.nn.Linear(10, 1)
        self.output_layer.weight = torch.nn.Parameter(torch.tensor(
            [[0,0,0,0,0,0,0,0,1,0]], dtype=torch.float))
        self.output_layer.bias = torch.nn.Parameter(torch.tensor([0.]))

    def forward(self, w):
        x = self.word_embedding[w] + self.pos_encoding(len(w))
        y = self.encoder(x.unsqueeze(1)).squeeze(1)
        z = self.output_layer(y[-1])
        return z

model = Model()
optim = torch.optim.Adam(model.parameters(), lr=3e-4)

w = torch.tensor([random.randrange(2) for i in range(10)]+[2])
output = model(w)
print(w, output)
