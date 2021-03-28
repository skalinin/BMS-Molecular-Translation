

def is_put_space(prev_char, curr_char):
    """Cases to put space in string."""

    # split numbers from letters
    if (
        curr_char.isdigit()
        and not prev_char.isdigit()
    ):
        return True

    # split letters from numbers
    if (
        curr_char.isalpha()
        and prev_char.isdigit()
    ):
        return True

    # split upper letters and leave clued lower
    # chars with upper ones (e.g. "Br").
    if (
        curr_char.isalpha()
        and curr_char.isupper()
    ):
        return True

    # split non-letters symbols
    if (
        not curr_char.isalpha()
        and not curr_char.isdigit()
    ):
        return True

    return False


def split_InChI_to_tokens(raw_text):
    """Split InChI-string to separate tokens."""

    # remove constant "InChI=1S/" from text
    raw_text = '/'.join(raw_text.split('/')[1:])

    splitted_text = ''
    prev_char = ''
    for char in raw_text:
        if is_put_space(prev_char, char):
            splitted_text += ' '
        splitted_text += char
        prev_char = char
    return splitted_text.lstrip(' ')


class Tokenizer(object):
    """Text tokenizer.

    All raw text should be preprocess to separate tokens by spaces.
    """

    def __init__(self):
        self.token_to_idx = {}
        self.idx_to_token = {}

    def __len__(self):
        return len(self.stoi)

    def fit_on_texts(self, texts):
        """Create vocab of tokens from list of texts."""

        vocab = set()
        for text in texts:
            vocab.update(text.split(' '))
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        for idx, token in enumerate(vocab):
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

    def text_to_sequence(self, text):
        """Convert text to sequence of token indexes."""

        sequence = []
        sequence.append(self.token_to_idx['<sos>'])
        for s in text.split(' '):
            sequence.append(self.token_to_idx[s])
        sequence.append(self.token_to_idx['<eos>'])
        return sequence
