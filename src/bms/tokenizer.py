from rdkit import Chem
import re


def inchi2smile(inchi):
    """Convers inchi string to smile."""
    return Chem.MolToSmiles(Chem.MolFromInchi(inchi))


# Source https://github.com/XinhaoLi74/SmilesPE/blob/master/SmilesPE
def atomwise_tokenizer(smi, exclusive_tokens=None):
    """
    Tokenize a SMILES molecule at atom-level:
        (1) 'Br' and 'Cl' are two-character tokens
        (2) Symbols with bracket are considered as tokens

    Args:
    smi (str): Smile string.
    exclusive_tokens (list of str, optional): A list of specifical symbols with
        bracket you want to keep. e.g., ['[C@@H]', '[nH]']. Other symbols with
        bracket will be replaced by '[UNK]'. default is `None`.
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
    """Text tokenizer."""

    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}
        self.vocab = set()

    def __len__(self):
        return len(self.token2idx)

    def add_texts(self, texts):
        """Create vocab of tokens from list of texts."""
        for text in texts:
            self.vocab.update(text)

    def fit_on_texts(self):
        self.vocab = sorted(self.vocab)
        self.vocab.append('<sos>')
        self.vocab.append('<eos>')
        for idx, token in enumerate(self.vocab):
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    def text_to_sequence(self, text):
        """Convert text to sequence of token indexes."""
        sequence = []
        sequence.append(self.token2idx['<sos>'])
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
