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
#device = torch.device("cuda:0")
device=  torch.device("cpu:0")

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
            state = [s.to(target.device) for s in state]
        else:
            _, state = self.encoder(context)
        pred, _, _, _ = self.decoder(state, ref.shape[0], ref)
        pred = pred.view(n_seq * n_batch, -1)
        tgt = tgt.view(n_seq * n_batch)
        scores = self.score(pred, tgt).view(n_seq, n_batch)
        return -scores.sum(dim=0)

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
    left, right, left_rev, right_rev = (batch_seqs(s).to(device) for s in zip(*data))
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
    left, right, left_rev, right_rev = (batch_seqs(s).to(device) for s in zip(*data))
    return left, right, left_rev, right_rev

def right_branching(string):
    if len(string) == 1:
        return tuple(string)
    return ((string[0],), right_branching(string[1:]))

def parse(string, scorer, depth):
    assert string[0] != scorer.vocab.sos()
    assert string[-1] != scorer.vocab.eos()
    offset_l = 1 if string[0] == scorer.vocab.sos() else 0
    offset_r = 1 if string[-1] == scorer.vocab.eos() else 0
    if depth == 0 or len(string) - offset_l - offset_r == 1:
        #return " ".join(scorer.vocab.decode(string))
        return right_branching(string)
    batch = parser_batch(string, offset_l, offset_r)
    conditional, unconditional = scorer(*batch)
    scores = conditional - unconditional
    split = 1 + offset_l + torch.argmin(scores)
    left, right = string[:split], string[split:]
    return (parse(left, scorer, depth - 1), parse(right, scorer, depth - 1))

def pp(tree):
    if not isinstance(tree, tuple):
        return str(tree)
    return "(%s)" % " ".join(pp(t) for t in tree)

def evaluate(pred_tree, gold_tree):
    def spans(tree):
        found = []
        def traverse(subtree, start):
            if not isinstance(subtree, tuple):
                found.append((start, start + 1))
                return start + 1
            end = start
            for t in subtree:
                end = traverse(t, end)
            found.append((start, end))
            return end
        traverse(tree, 0)
        found = [s for s in found if s[1] > s[0] + 1]
        return set(found)
    pred_spans = spans(pred_tree)
    gold_spans = spans(gold_tree)
    tp = len(pred_spans & gold_spans)
    fp = len(pred_spans - gold_spans)
    fn = len(gold_spans - pred_spans)
    return tp, fp, fn
    #fp = len(t for t in pred_spans if t not in gold_spans)
    #fn = len(t for t in gold_spans if t not in pred_spans)

def validate(corpus, scorer):
    #for i in range(3):
    #    string = corpus.train.strings[random.randint(len(corpus.val.strings))]
    #    print(pp(parse(string, scorer, depth=2)))
    tps = []
    fps = []
    fns = []
    f1s = []
    for string, tree in zip(corpus.train.strings, corpus.train.trees):
        pred_tree = parse(string, scorer, depth=2)
        tp, fp, fn = evaluate(pred_tree, tree)
        if tp == fp == fn == 0:
            continue
        tps.append(tp)
        fps.append(fp)
        fns.append(fp)
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1s.append(2 * p * r / (p + r))
    macro_f1 = np.mean(f1s)
    p = sum(tps) / (sum(tps) + sum(fps))
    r = sum(tps) / (sum(tps) + sum(fns))
    micro_f1 = 2 * p * r / (p + r)
    print("macro_f1", macro_f1)
    print("micro_f1", micro_f1)

def main():
    corpus = load_english_treebank(max_length=40, strip_punct=True)
    scorer = SplitScorer(corpus.vocab).to(device)
    opt = optim.Adam(scorer.parameters(), lr=0.001)
    for i_epoch in range(100):
        epoch_loss = 0
        for i_batch in range(50):
            batch = sample_batch(corpus)
            conditional, unconditional = scorer(*batch)
            loss = -(conditional + unconditional).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        print(epoch_loss / 50)
        validate(corpus, scorer)

if __name__ == "__main__":
    main()
