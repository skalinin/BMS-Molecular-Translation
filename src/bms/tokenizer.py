from bms.model_config import model_config


def make_inchi_from_chem_dict(chem_dict, chem_to_take):
    inchi_text = ''
    for chem in chem_to_take:
        chem_text = chem_dict[chem]
        if len(chem_text) > 0:
            inchi_text += (chem + chem_text)
    return inchi_text


def split_InChI_to_chem_groups(InChI_text, chem_tokens):
    chem_groups_dict = {}
    # all tokens must be in chem_groups_dict
    for chem_token in chem_tokens:
        chem_groups_dict[chem_token] = ''

    curr_chem_token = None
    processed_tokens = []
    for i in range(len(InChI_text)):
        for chem_token in chem_tokens:
            if (
                InChI_text[i:].startswith(chem_token) and
                chem_token not in processed_tokens
            ):
                curr_chem_token = chem_token
                processed_tokens.append(chem_token)
        chem_groups_dict[curr_chem_token] += InChI_text[i]

    # remove start tokens from the begining of the strings
    for chem_token in chem_tokens:
        if chem_groups_dict[chem_token]:
            chem_groups_dict[chem_token] = \
                chem_groups_dict[chem_token][len(chem_token):]

    return chem_groups_dict


def get_chem_text_from_dict(chem_groups_dict, key):
    return chem_groups_dict.get(key)


def text_to_sequence(InChI_tokens, tokenizer, InChI_chem_group):
    sequence = tokenizer.text_to_sequence(
        InChI_tokens, model_config['chem2start_token'][InChI_chem_group])
    return sequence


def is_put_space(prev_char, curr_char):
    """Cases to put space in string."""

    # split numbers from anything
    if (
        curr_char.isdigit()
        # and not prev_char.isdigit()
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

    splitted_text = ''
    prev_char = ''
    for char in raw_text:
        if is_put_space(prev_char, char):
            splitted_text += ' '
        splitted_text += char
        prev_char = char
    return splitted_text.lstrip(' ')


class Tokenizer:
    """Text tokenizer.

    All raw text should be preprocess to separate tokens by spaces.
    """

    def __init__(self, start_tokens):
        self.token2idx = {}
        self.idx2token = {}
        self.start_tokens = start_tokens

    def __len__(self):
        return len(self.token2idx)

    def fit_on_texts(self, texts):
        """Create vocab of tokens from list of texts."""
        vocab = set()
        for text in texts:
            if text:
                vocab.update(text.split(' '))
        vocab = sorted(vocab)
        for start_token in self.start_tokens:
            vocab.append(start_token)
        vocab.append('<eos>')
        for idx, token in enumerate(vocab):
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    def text_to_sequence(self, text, start_token):
        """Convert text to sequence of token indexes."""
        sequence = []
        sequence.append(self.token2idx[start_token])
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
