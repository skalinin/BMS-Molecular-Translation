# Solution for Bristol-Myers Squibb – Molecular Translation

![header](data/header.jpeg)


## Solution

The main goal is to translate chemical structure images into [InChI](https://en.wikipedia.org/wiki/International_Chemical_Identifier) transcription - machine-readable format. This is image captioning task and solved as CNN+RNN architecture.

![task example](data/image_captioning.png)

Key points:
* EfficientNet as encoder and LSTM+Attention as decoder.
* Adaptive batchsampler (the higher sample loss, the higher probability to add the sample in a batch).
* Predict [Smile](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) notation and than convert it to InChI (smile is much more simpler to predict - shorter notation and less token-classes).

What other experiments could be done:
* Synthetic images generation
* Split InChI to 8 indepenpent layers
