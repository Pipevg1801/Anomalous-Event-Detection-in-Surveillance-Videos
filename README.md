# Anomalous Event Detection in Surveillance Videos

Detects **when** an anomalous event happens in a surveillance video and **what kind** of event it is (robbery, assault or fighting), and **who** is involved. The output is an MP4 where every person is drawn in a green (normal) or red (involved) box with the event class.

![Pipeline](Images/pipeline.png)

## How it works

| Stage | What it does | Trained? | Based on |
| --- | --- | --- | --- |
| **A · People** | Detects and tracks each person with an ID | No | YOLO11s + ByteTrack |
| **B · Events** | Scores anomaly per 16-frame segment and classifies the event | Yes | Frozen CLIP ViT-B/16 + 2-layer Transformer (VadCLIP-style, MIL top-k) |
| **C2 · Per-person fusion** | Decides which people are involved | Calibrated only | Global gate + CLIP crops, motion and interaction signals |
| **D · Visualization** | Smooths states over time and renders the video | No | OpenCV + ffmpeg |

**Classes:** Normal, Robo (Robbery), Asalto (Assault) and Agresión (Fighting + Abuse).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        
pip install -r requirements.txt
```
## Data

The datasets are **not included** in this repository. Download them from their original sources:

| Data | Source | Required for |
| --- | --- | --- |
| VadCLIP CLIP features for UCF-Crime (`UCFClipFeatures/`) | Download links in the [VadCLIP README](https://github.com/nwpu-zxr/VadCLIP) | Training and evaluating stage B |
| Official lists (`list/` folder) | [VadCLIP repo](https://github.com/nwpu-zxr/VadCLIP) (downloaded automatically if not set) | Splits and test ground truth |
| UCA annotations (`UCFCrime_*.json`) | [UCA repo](https://github.com/Xuange923/Surveillance-Video-Understanding) (downloaded automatically if not set) | Auxiliary supervision |
| UCF-Crime videos | Official UCF-Crime release | Tracking, annotation and demo only |

You don't need all the UCF-Crime videos (~100 GB): only the ~40 videos used for person annotation and the demo. Training stage B uses only the features.


## License

This code is released under the **AGPL-3.0** license, since it depends on [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0).

The datasets keep their own terms and are intended for **academic and research use only**. UCF-Crime contains real footage of people; please respect their privacy when sharing results.

## Citations

If you use this work, please cite the resources it builds on:

- **UCF-Crime**: W. Sultani, C. Chen, M. Shah. *Real-world Anomaly Detection in Surveillance Videos*. CVPR 2018.
- **UCA**: T. Yuan et al. *Towards Surveillance Video-and-Language Understanding: New Dataset, Baselines, and Challenges*. CVPR 2024.
- **VadCLIP**: P. Wu et al. *VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection*. AAAI 2024.
- **CLIP**: A. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision*. ICML 2021.
- **ByteTrack**: Y. Zhang et al. *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. ECCV 2022.
- **YOLO11**: G. Jocher, J. Qiu. *Ultralytics YOLO11*. 2024. https://github.com/ultralytics/ultralytics

## Acknowledgements

Stage B follows the design of [VadCLIP](https://github.com/nwpu-zxr/VadCLIP), and its MIL training is inspired by [UMIL](https://github.com/ktr-hubrt/UMIL).
