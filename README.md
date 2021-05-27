# Solution for Bristol-Myers Squibb – Molecular Translation

![header](data/header.jpeg)


## Solution

The main goal is to translate chemical structure images into [InChI](https://en.wikipedia.org/wiki/International_Chemical_Identifier) transcription - machine-readable format. This is image captioning task and solved as CNN+RNN architecture.

![task example](data/image_captioning.png)

#### Key points:

* EfficientNet as encoder, and LSTM+Attention as decoder.
* Adaptive batchsampler (the higher sample loss, the higher probability to add the sample in a batch).
* Predict [Smile](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system) notation and than convert it to InChI (Smiles is much more simpler to predict - shorter notation and less token-classes).
* Freeze the encoder and train the lstm-decoder separately.

#### What other experiments could be done:

* More experiments with synthetic images generation.
* Split InChI up to 8 indepenpent string layers (separated by the "/" notation: "/b", "/t", "/m", "/s", etc.) and train the separate models for each layer. Each layer in the InChI describes different information about the molecule, and several models trained on separate inchi-layers can get good results.
* Add rotation transoform to different angles.
* Add beam search.
* Try to replace LSTM to Transformer.
