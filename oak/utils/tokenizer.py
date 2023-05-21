import tiktoken


class Tokenizer(object):
    def __init__(self, encoding='gpt2'):
        self.tokenizer = tiktoken.get_encoding(encoding)

    def encode(self, x):
        """Encode a list of sentences into an array"""
        x = x if isinstance(x, list) else [x]
        return self.tokenizer.encode_batch(x)

    def decode(self, x):
        """Decode a list of sentences into a list of strings"""
        x = x if isinstance(x, list) else [x]
        return self.tokenizer.decode_batch(x)

    def __call__(self, x):
        return self.encode(x)

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab
