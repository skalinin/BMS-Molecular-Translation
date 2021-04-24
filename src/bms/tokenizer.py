from rdkit import Chem

from bms.model_config import model_config


def make_smile_from_inchi(inchi):
    return Chem.MolToSmiles(Chem.MolFromInchi(inchi))


def split_InChI_to_tokens(raw_text):
    """Split InChI-string to separate tokens."""
    return " ".join(raw_text)


class Tokenizer:
    """Text tokenizer.

    All raw text should be preprocess to separate tokens by spaces.
    """

    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}

    def __len__(self):
        return len(self.token2idx)

    def fit_on_texts(self, texts):
        """Create vocab of tokens from list of texts."""
        vocab = set()
        for text in texts:
            if text:
                vocab.update(text)
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        for idx, token in enumerate(vocab):
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    def text_to_sequence(self, text):
        """Convert text to sequence of token indexes."""
        sequence = []
        sequence.append(self.token2idx['<sos>'])
        if text:
            for s in text.split(' '):
                sequence.append(self.token2idx[s])
        sequence.append(self.token2idx['<eos>'])
        return sequence

    def predict_caption(self, sequence):
        caption = ''
        for i in sequence:
            if i == self.token2idx['<eos>']:
                break
            caption += self.idx2token[i]
        return caption

    def predict_captions(self, sequences):
        captions = []
        for sequence in sequences:
            caption = self.predict_caption(sequence)
            captions.append(caption)
        return captions
