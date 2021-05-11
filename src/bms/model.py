import timm
import torch
import torch.nn as nn


class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = timm.create_model('efficientnet_b3', pretrained=True)

    def forward(self, x):
        features = self.cnn.forward_features(x)
        features = features.permute(0, 2, 3, 1)
        return features


# Source from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
class Attention(nn.Module):
    """Attention Network."""

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        Args:
            encoder_dim: Feature size of encoded images.
            decoder_dim: Size of decoder's RNN.
            attention_dim: Size of the attention network.
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """Forward propagation.

        Args:
            encoder_out: Encoded images, a tensor of dimension (batch_size,
                num_pixels, encoder_dim).
            decoder_hidden: Previous decoder output, a tensor of dimension
                (batch_size, decoder_dim).
        """
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = \
            (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha


# Source from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
class DecoderWithAttention(nn.Module):
    """Decoder network with attention network used for training."""

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size,
                 device, encoder_dim=1536, dropout=0.5):
        """
        Args:
            attention_dim: Input size of attention network.
            embed_dim: Input size of embedding network.
            decoder_dim: Input size of decoder network.
            vocab_size: Total number of characters used in training.
            device: Device.
            encoder_dim: Input size of encoder network.
            dropout: Dropout rate.
        """
        super(DecoderWithAttention, self).__init__()
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.device = device
        self.dropout = dropout
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim,
                                       bias=True)
        self.dropout = nn.Dropout(p=self.dropout)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Args:
            encoder_out: Output of encoder network.
            encoded_captions: Transformed sequence from character to integer.
            caption_lengths: Length of transformed sequence.
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)

        embeddings = self.embedding(encoded_captions)
        h, c = self.init_hidden_state(encoder_out)
        # We won't decode at the <end> position, since we've finished
        # generating as soon as we generate <end>. So, decoding lengths are
        # actual lengths - 1 (we don't pass <end> to the lstm as we don't want
        # to teach it to predict after <end> token).
        decode_lengths = (caption_lengths - 1).tolist()
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size)
        predictions = predictions.to(self.device)
        # predict sequence
        for t in range(max(decode_lengths)):
            # get from batch samples with lengths more than t (samples shoud
            # have been sorted by length)
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = \
                self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            rnn_input = torch.cat([embeddings[:batch_size_t, t, :],
                                  attention_weighted_encoding], dim=1)
            h, c = self.decode_step(rnn_input,
                                    (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
        return predictions, decode_lengths

    def predict(self, encoder_out, decode_lengths, start_token_index):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        # embed start token for LSTM input
        start_tockens = torch.ones(batch_size, dtype=torch.long).to(self.device)
        start_tockens = start_tockens * start_token_index
        embeddings = self.embedding(start_tockens)
        # initialize hidden state and cell state of LSTM cell
        h, c = self.init_hidden_state(encoder_out)
        predictions = torch.zeros(batch_size, decode_lengths, vocab_size)
        predictions = predictions.to(self.device)
        for t in range(decode_lengths):
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            rnn_input = torch.cat([embeddings, attention_weighted_encoding],
                                  dim=1)
            h, c = self.decode_step(rnn_input, (h, c))
            preds = self.fc(self.dropout(h))
            predictions[:, t, :] = preds
            embeddings = self.embedding(torch.argmax(preds, -1))
        return predictions
