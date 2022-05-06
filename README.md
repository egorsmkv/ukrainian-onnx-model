# An ONNX model for speech recognition of the Ukrainian language

## Overview

This repository contains an ONNX model for speech recognition of the Ukrainian language exported from [wav2vec2 1b model](https://huggingface.co/Yehor/wav2vec2-xls-r-1b-uk-with-lm).

If you want to export own ONNX model, follow this [Google Colab](https://colab.research.google.com/drive/1bvkwrdLl6rgbWdF2fYe0Tt0-CkG2vvBD?usp=sharing).

## Installation

Download [**onnx-uk-1b.zip**](https://www.dropbox.com/s/03qh8u10lkyfntz/onnx-uk-1b.zip?dl=0) file and unpack it in the repository folder.

Install Python dependencies:

```bash
pip install onnxruntime numpy scipy
```

## Running

```bash
python recognize.py
```
