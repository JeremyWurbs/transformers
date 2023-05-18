import os
import sys


class ShakespeareTokenizer(object):
    def __init__(self):
        with open(os.path.join(os.path.dirname(sys.modules[__name__].__file__), '..', 'data', 'shakespeare.txt'), 'r', encoding='utf-8') as f:
            text = f.read()

            chars = sorted(list(set(text)))
            self.vocab_size = len(chars)
            self._stoi = {ch: i for i, ch in enumerate(chars)}
            self._itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, x):
        """Encode a list of sentences into an array"""
        x = x if isinstance(x, list) else [x]
        return [[self._stoi[char] for char in sent] for sent in x]

    def decode(self, x):
        """Decode a list of sentences into a list of strings"""
        x = x if isinstance(x, list) else [x]
        return [[''.join([self._itos[token] for token in sent])] for sent in x]

    def __call__(self, x):
        return self.encode(x)
