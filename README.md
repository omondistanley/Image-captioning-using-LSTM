# Conditioned LSTM Language Model for Image Captioning

This project implements an image captioning system using a conditioned LSTM language model in PyTorch. The model generates natural language descriptions for images from the Flickr8k dataset by combining deep visual features (from a pretrained ResNet-18) with a sequence model trained on human-written captions.


## Overview

The goal of this project is to generate captions for images by training an LSTM-based language model that is conditioned on image features. The workflow includes:

1. **Image Encoding:** Extracting 512-dimensional feature vectors from images using a pretrained ResNet-18.
2. **Caption Processing:** Tokenizing and preparing caption data, including special tokens and padding.
3. **Language Modeling:** Training an LSTM to generate captions, first unconditioned, then conditioned on image features.
4. **Decoding:** Implementing greedy, sampling, and beam search decoders for caption generation.

## Project Structure


'''

├── lstm_flickr_caption_generator_torch.ipynb   # Main Jupyter notebook
├── encoded_images_train.pt                     # Precomputed image encodings (train)
├── data/
│   ├── Flickr_8k.trainImages.txt
│   ├── Flickr_8k.devImages.txt
│   ├── Flickr_8k.testImages.txt
│   ├── Flickr8k.token.txt
│   ├── Flickr8k_Dataset/
│   └── Flickr8k_text/
├── data.zip / data.tar.


'''
## Setup & Requirements

- Python 3.9+
- PyTorch
- torchvision
- PIL (Python Image Library)


## Data Preparation

**Download the Flickr8k dataset.**
   - You can use the provided Google Cloud Storage link in the notebook or request access from the [official site](https://forms.illinois.edu/sec/1713398).
   - Unzip the data into the `data/` directory.


## Model Architecture

- **Image Encoder:** Pretrained ResNet-18 (torchvision), outputting a 512-dim feature vector per image.
- **Caption Encoder:** Tokenized captions with `<START>`, `<EOS>`, and `<PAD>` tokens.
- **LSTM Decoder:** 
  - Embedding size: 512
  - Hidden size: 512
  - Input: Concatenation of word embedding and image encoding (1024-dim per timestep)
  - Output: Vocabulary-sized logits per timestep

## Training

- **Loss:** Cross-entropy, ignoring `<PAD>` tokens.
- **Optimizer:** AdamW
- **Batch size:** 16
- **Epochs:** Train until accuracy exceeds 0.5 (typically ~5 epochs for this dataset).

## Decoding Strategies

- **Greedy Decoder:** Selects the most probable word at each timestep.
- **Sampling Decoder:** Samples the next word from the predicted distribution.
- **Beam Search Decoder:** Maintains the top-n most probable sequences at each step, expanding and pruning the beam.

## References

- [Flickr8k Dataset](https://forms.illinois.edu/sec/1713398)
- [ResNet-18 Paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
- PyTorch and torchvision documentation

---

