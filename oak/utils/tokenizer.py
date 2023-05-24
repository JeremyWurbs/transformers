from whisper.tokenizer import get_tokenizer


class Tokenizer(object):
    def __init__(self, language='en', multilingual=None, task=None):
        if multilingual:
            task = 'translate'
        self.tokenizer = get_tokenizer(multilingual=multilingual, language=language, task=task)

        # Special tokens (we just select some to reuse for our purposes)
        self._bos = '<|startoftranscript|>'
        self._eos = '<|endoftext|>'
        self._pad = '<|nospeech|>'

    def encode(self, x):
        """Encode a list of sentences into an array"""
        x = x if isinstance(x, list) else [x]
        return self.tokenizer.encoding.encode_batch(x, allowed_special={self._eos})

    def decode(self, x):
        """Decode a list of sentences into a list of strings"""
        x = x if isinstance(x, list) else [x]
        return self.tokenizer.encoding.decode_batch(x)

    def __call__(self, x):
        return self.encode(x)

    @property
    def vocab_size(self):
        return self.tokenizer.encoding.n_vocab

    @property
    def BOS(self):
        return self.tokenizer.encode(self._bos, allowed_special={self._bos})[0]

    @property
    def EOS(self):
        return self.tokenizer.encode(self._eos, allowed_special={self._eos})[0]

    @property
    def PAD(self):
        return self.tokenizer.encode(self._pad, allowed_special={self._pad})[0]
