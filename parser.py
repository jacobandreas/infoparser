#!/usr/bin/env python3

from corpus import load_english_treebank

import numpy as np
import torch
from torch import nn, optim
from torchdec.seq import batch_seqs, Encoder, Decoder

SEED = 0

N_EMBED = 64
N_HIDDEN = 512
N_LAYERS = 1
N_BATCH = 256

random = np.random.RandomState(SEED)

class WordPredictor(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.encoder = Encoder(vocab, N_EMBED, N_HIDDEN, N_LAYERS, bidirectional=False)
        self.decoder = Decoder(vocab, N_EMBED, N_HIDDEN, N_LAYERS)
        self.score = nn.CrossEntropyLoss(ignore_index=vocab.pad(), reduction='none')

    def forward(self, context, target):
        ref = target[:-1, ...]
        tgt = target[1:, ...]
        n_seq, n_batch = ref.shape[:2]
        if context is None:
            state = [torch.zeros(N_LAYERS, n_batch, N_HIDDEN) for _ in range(2)]
        else:
            _, state = self.encoder(context)
        pred, _, _, _ = self.decoder(state, ref.shape[0], ref)
        pred = pred.view(n_seq * n_batch, -1)
        tgt = tgt.view(n_seq * n_batch)
        scores = self.score(pred, tgt).view(n_seq, n_batch)
        return scores.sum(dim=0)

class SplitScorer(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.forward_pred = WordPredictor(vocab)
        self.backward_pred = WordPredictor(vocab)

    def forward(self, left_forward, right_forward, left_backward, right_backward):
        forward_cond = self.forward_pred(left_forward, right_forward)
        forward_uncond = self.forward_pred(None, right_forward)
        backward_cond = self.backward_pred(right_backward, left_backward)
        backward_uncond = self.backward_pred(None, left_backward)
        return forward_cond + backward_cond, forward_uncond + backward_uncond

def sample_batch(corpus):
    strings = corpus.train.strings
    max_len = max(len(s) for s in strings)
    batch_len = random.randint(2, max_len+1)
    batch_split = random.randint(1, batch_len)
    data = []
    candidates = [s for s in strings if len(s) >= batch_len]
    while len(data) < N_BATCH:
        string = candidates[random.randint(len(candidates))]
        assert len(string) >= batch_len
        offset = random.randint(len(string) - batch_len + 1)
        left = string[offset:offset+batch_split]
        right = string[offset+batch_split:offset+batch_len]
        left_rev = list(reversed(left))
        right_rev = list(reversed(right))
        data.append((
            left, 
            [left[-1]] + right,
            [right[0]] + left_rev,
            right_rev
        ))
    left, right, left_rev, right_rev = (batch_seqs(s) for s in zip(*data))
    return left, right, left_rev, right_rev

def parser_batch(string, offset_left=0, offset_right=0):
    data = []
    for i in range(1 + offset_left, len(string) - offset_right):
        left = string[:i]
        right = string[i:]
        left_rev = list(reversed(left))
        right_rev = list(reversed(right))
        data.append((
            left,
            [left[-1]] + right,
            [right[0]] + left_rev,
            right_rev
        ))
    left, right, left_rev, right_rev = (batch_seqs(s) for s in zip(*data))
    return left, right, left_rev, right_rev

def parse(string, scorer, depth):
    offset_l = 1 if string[0] == scorer.vocab.sos() else 0
    offset_r = 1 if string[-1] == scorer.vocab.eos() else 0
    if depth == 0 or len(string) - offset_l - offset_r == 1:
        return " ".join(scorer.vocab.decode(string))
    batch = parser_batch(string, offset_l, offset_r)
    conditional, unconditional = scorer(*batch)
    scores = torch.log_softmax(conditional, dim=0) - torch.log_softmax(unconditional, dim=0)
    split = 1 + offset_l + torch.argmin(scores)
    left, right = string[:split], string[split:]
    return (parse(left, scorer, depth - 1), parse(right, scorer, depth - 1))
    return (scorer.vocab.decode(string[:split]), scorer.vocab.decode(string[split:]))

def pp(tree):
    if isinstance(tree, str):
        return tree
    return "(%s)" % " ".join(pp(t) for t in tree)

def validate(corpus, scorer):
    for i in range(3):
        string = corpus.val.strings[random.randint(len(corpus.val.strings))]
        print(pp(parse(string, scorer, depth=5)))

def main():
    corpus = load_english_treebank(max_length=40)
    scorer = SplitScorer(corpus.vocab)
    opt = optim.Adam(scorer.parameters(), lr=0.001)
    for i_epoch in range(100):
        epoch_loss = 0
        for i_batch in range(10):
            batch = sample_batch(corpus)
            conditional, unconditional = scorer(*batch)
            loss = (conditional + unconditional).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        print(epoch_loss / 10)
        validate(corpus, scorer)

if __name__ == "__main__":
    main()
