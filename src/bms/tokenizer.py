from rdkit import Chem
import re



def make_smile_from_inchi(inchi):
    return Chem.MolToSmiles(Chem.MolFromInchi(inchi))


def atomwise_tokenizer(smi, exclusive_tokens=None):
    """
    Get from https://github.com/XinhaoLi74/SmilesPE/blob/master/SmilesPE/pretokenizer.py

    Tokenize a SMILES molecule at atom-level:
        (1) 'Br' and 'Cl' are two-character tokens
        (2) Symbols with bracket are considered as tokens
    exclusive_tokens: A list of specifical symbols with bracket you want to keep. e.g., ['[C@@H]', '[nH]'].
    Other symbols with bracket will be replaced by '[UNK]'. default is `None`.
    """
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]

    if exclusive_tokens:
        for i, tok in enumerate(tokens):
            if tok.startswith('['):
                if tok not in exclusive_tokens:
                    tokens[i] = '[UNK]'
                    print('Unknown tokenizer!!!')
    return tokens


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
            for s in text:
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
