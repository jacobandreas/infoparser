#!/usr/bin/env python3

from collections import namedtuple
import os
import sexpdata
from torchdec.vocab import Vocab

Corpus = namedtuple("Corpus", "train val test vocab")
Fold = namedtuple("Fold", "strings trees")

def tree_flatten(tree):
    if isinstance(tree, tuple):
        return sum([tree_flatten(t) for t in tree], ())
    return tree

def tree_yield(tree):
    if len(tree) == 2 and all(isinstance(t, str) for t in tree):
        return (tree[1].lower(),)
    return sum([tree_yield(t) for t in tree[1:]], ())

def tree_map(fn, tree):
    if isinstance(tree, tuple):
        return tuple(tree_map(t) for t in tree)
    return fn(tree)

def sexp_clean(parsed):
    def helper(tree):
        if isinstance(tree, list):
            rec = tuple(helper(t) for t in tree)
            return tuple(r for r in rec if r[0] != "-NONE-")
        elif isinstance(tree, sexpdata.Symbol):
            value = tree.value()
        elif isinstance(tree, int) or isinstance(tree, float):
            value = str(tree)
        else:
            assert False
        return value.replace("-LQUOT-", "'").replace("-RQUOT-", "`")
    tree, = parsed
    return helper(tree)

def read_ptb_tree(sexp):
    escaped = sexp.replace("'", "-LQUOT-").replace("`", "-RQUOT-")
    parsed = sexpdata.loads(escaped)
    cleaned = sexp_clean(parsed)
    return cleaned

def load_ptb(filename, max_length=None):
    out = []
    with open(filename) as f:
        for line in f:
            #if len(out) >= 1000:
            #    break
            try:
                tree = read_ptb_tree(line.strip())
                if max_length is not None and len(tree_yield(tree)) >= max_length:
                    continue
                out.append(tree)
            except:
                pass
    return out

def load_english_treebank(max_length=None):
    DATA_DIR = "/Users/jaandrea/data/english_treebank"
    train_trees = load_ptb(
        os.path.join(DATA_DIR, "alltrees_train_2to21.mrg.oneline"),
        max_length=max_length
    )
    val_trees = load_ptb(
        os.path.join(DATA_DIR, "alltrees_dev.mrg.oneline"),
        max_length=max_length
    )
    test_trees = load_ptb(
        os.path.join(DATA_DIR, "alltrees_test.mrg.oneline"),
        max_length=max_length
    )
    train_strings = [tree_yield(t) for t in train_trees]
    vocab = Vocab()
    for string in train_strings:
        for word in string:
            vocab.add(word)
    folds = {}
    for fold, trees in [
            ("train", train_trees), ("val", val_trees), ("test", test_trees)
    ]:
        strings = [tree_yield(t) for t in trees]
        #trees_i = [tree_map(vocab.index, t) for t in trees]
        strings_i = [vocab.encode(s, unk=True) for s in strings]
        folds[fold] = Fold(strings_i, None)
    return Corpus(folds["train"], folds["val"], folds["test"], vocab)

if __name__ == "__main__":
    load_english_treebank()
