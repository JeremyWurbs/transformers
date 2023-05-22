from whisper.tokenizer import get_tokenizer


class Tokenizer(object):
    def __init__(self, language='en', multilingual=None, task=None):
        if multilingual:
            task = 'translate'
        self.tokenizer = get_tokenizer(multilingual=multilingual, language=language, task=task)

    def encode(self, x):
        """Encode a list of sentences into an array"""
        x = x if isinstance(x, list) else [x]
        return self.tokenizer.encoding.encode_batch(x)

    def decode(self, x):
        """Decode a list of sentences into a list of strings"""
        x = x if isinstance(x, list) else [x]
        return self.tokenizer.encoding.decode_batch(x)

    def __call__(self, x):
        return self.encode(x)

    @property
    def vocab_size(self):
        return self.tokenizer.encoding.n_vocab
