

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


def remove_InChI_prefix(text):
    """Remove constant "InChI=1S/" from text."""

    return '/'.join(text.split('/')[1:])


def split_InChI_to_tokens(raw_text):
    """Split InChI-string to separate tokens."""

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
        self.token2idx = {}
        self.idx2token = {}

    def __len__(self):
        return len(self.token2idx)

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
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    def text_to_sequence(self, text):
        """Convert text to sequence of token indexes."""

        sequence = []
        sequence.append(self.token2idx['<sos>'])
        for s in text.split(' '):
            sequence.append(self.token2idx[s])
        sequence.append(self.token2idx['<eos>'])
        return sequence

    def predict_caption(self, sequence):
        caption = ''
        for i in sequence:
            if i == self.token2idx['<eos>'] or i == self.token2idx['<pad>']:
                break
            caption += self.idx2token[i]
        return caption

    def predict_captions(self, sequences):
        captions = []
        for sequence in sequences:
            caption = self.predict_caption(sequence)
            captions.append(caption)
        return captions
