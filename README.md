# Multimodal Emotion Detection

A deep learning system that detects emotion from video by jointly analysing facial expressions and speech audio, with GradCAM-based visual explanations.

## Overview

The model fuses two modalities:

- **Audio** — mel spectrogram processed through a CNN and Transformer encoder
- **Video** — facial frames processed through a residual CNN and Transformer encoder

The two streams are combined via a **Bidirectional Cross-Attention** fusion layer that allows each modality to inform the other before final classification.

**Detectable emotions:** Angry · Happy · Neutral · Sad

**Dataset:** CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)

## Explainability

GradCAM heatmaps are generated for both modalities after each prediction:

- **Video GradCAM** — highlights face regions (eyes, mouth, brow) that most influenced the decision
- **Audio GradCAM** — highlights time–frequency bands in the mel spectrogram that carried the strongest emotional signal

## Demo

Upload any short video clip (3–10 seconds with a visible face and speech) to get:

1. Predicted emotion with per-class confidence scores
2. GradCAM overlay on sampled video frames
3. GradCAM overlay on the audio spectrogram

## Requirements

- Python 3.10+
- PyTorch, torchvision
- librosa, opencv-python
- streamlit, imageio-ffmpeg, matplotlib

## Usage

```bash
streamlit run app.py
```

---

*DATA 255 · Deep Learning · San José State University · Spring 2026*
