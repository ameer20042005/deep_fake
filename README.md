# Temporal Modeling for Deepfake Video Detection

A controlled comparison of three temporal architectures for video-level deepfake detection, sharing an identical ResNet-18 spatial backbone, evaluated on **Celeb-DF-v2**. Includes a desktop application for inference on arbitrary videos.

Undergraduate thesis, Electronics and Communication Engineering, Al-Muthanna University (April 2026).
Supervisor: Assist. Prof. Dr. Ahmed Saaudi.

---

## Problem

Frame-level classifiers detect manipulation artifacts in individual frames, but modern face-swap methods leave weak per-frame traces and stronger **temporal** inconsistencies — flicker at blend boundaries, unnatural expression dynamics, irregular blink patterns.

The usual response is to stack a temporal model on a spatial feature extractor. But *which* temporal model is typically chosen by convention rather than measurement, and the common candidates are rarely compared under identical conditions.

This work fixes the backbone, the preprocessing, and the training protocol, and varies only the temporal head.

---

## Setup

| Component | Choice |
|---|---|
| Spatial backbone | ResNet-18, ImageNet-pretrained (identical across all variants) |
| Temporal heads | Bi-LSTM · Transformer encoder · vanilla RNN |
| Dataset | Celeb-DF-v2 |
| Test set | 2,108 videos — 1,130 fake (53.6%), 978 real (46.4%) |
| Face detection | MTCNN, with margin retained to preserve blend-boundary artifacts |
| Input | 224×224, ImageNet normalization |

**Training:** AdamW, lr 8e-4, batch size 8, weight decay 1e-4, BCE loss, 12 epochs max, `ReduceLROnPlateau`, early stopping (patience 6), AMP enabled.
**Transfer learning:** backbone frozen for epochs 1–4, unfrozen from epoch 5.
**Split:** 80:10:10, performed at video level (not frame level) to avoid frame leakage across splits.

Before supervised training, PCA and K-Means were applied to the frame embeddings to inspect class separability. The projection showed substantial overlap between real and fake, indicating that Celeb-DF-v2 manipulations are not linearly separable from spatial features alone.

---

## Results

| Temporal head | Accuracy | F1 | Recall (fake) | Recall (real) | FP | FN | FPR |
|---|---|---|---|---|---|---|---|
| ResNet-18 + Bi-LSTM | 97.63% | 97.63% | 99.12% | 95.91% | 40 | 10 | 4.09% |
| ResNet-18 + Transformer | 97.87% | 97.86% | 98.50% | 97.14% | 28 | 17 | 2.86% |
| ResNet-18 + RNN | **97.96%** | **97.96%** | 98.85% | 96.93% | 30 | 13 | 3.07% |

### What the numbers actually say

**The three heads are indistinguishable on accuracy.** The spread is 0.33 percentage points across all three — roughly 7 videos out of 2,108. The simplest head (vanilla RNN) scores highest. No claim that one architecture is *more accurate* than another survives this data.

What does differ is the **error profile**:

- **Bi-LSTM** is the most sensitive detector (misses 10 of 1,130 fakes, 0.88%) and the noisiest (40 false alarms).
- **Transformer** trades sensitivity for precision on the real class — 28 false alarms, the lowest of the three, at the cost of 7 more missed fakes than the Bi-LSTM.
- **RNN** sits between them at a fraction of the parameter count.

The practical reading: on Celeb-DF-v2, the ResNet-18 backbone appears to carry most of the discriminative signal, and the choice of temporal head is an operating-point decision — how you want to spend your error budget between misses and false alarms — rather than an accuracy decision. For a moderation pipeline where false alarms cost user trust, the Transformer's operating point is preferable. For forensic triage where a missed fake is the expensive error, the Bi-LSTM's is.

<!-- TODO: add AUC for all three. AUC is threshold-independent and is what this field
     reports; accuracy at a fixed threshold cannot distinguish these models.
     sklearn.metrics.roc_auc_score on the saved probability outputs. -->

<!-- TODO: run McNemar's test between each pair. With ~2,108 samples a 0.09pp gap is
     about 2 videos; the test will almost certainly return p >> 0.05, and saying so
     explicitly is stronger than leaving the reader to wonder. -->

---

## Desktop application

GUI for running a trained model on a video file: frame extraction, MTCNN face cropping, per-frame scoring, and an aggregated video-level verdict with a confidence score.

<!-- TODO: add a screenshot -->

---

## Repository structure

```
├── data/           # Preprocessing, MTCNN face extraction, frame sampling
├── models/         # ResNet-18 backbone and the three temporal heads
├── train/          # Training loop and configuration
├── eval/           # Metrics, confusion matrices
└── app/            # Desktop inference application
```

## Reproducing

```bash
git clone https://github.com/ameer20042005/<repo-name>
cd <repo-name>
pip install -r requirements.txt

python train/train.py --head transformer    # or: bilstm | rnn
python eval/evaluate.py --checkpoint <path>
```

Celeb-DF-v2 is **not redistributed here**. Request access from the dataset authors: https://github.com/yuezunli/celeb-deepfakeforensics

---

## Limitations

- **Single-dataset evaluation.** Training and testing both use Celeb-DF-v2. Cross-dataset generalization — the property that determines real-world usefulness — is not measured.
- **Identity overlap is not controlled.** The split is video-disjoint but not identity-disjoint. Celeb-DF-v2 draws its real videos from 59 celebrities, so the same identity can appear in both train and test, and a model can score well by learning identity-specific cues rather than manipulation artifacts. An identity-disjoint re-split is needed to rule this out.
- **Threshold-dependent metrics only.** No AUC, no calibration analysis, no significance testing.
- **No compression robustness testing.** Deployed video is re-encoded; detectors typically degrade sharply under compression.
- **Narrow generation family.** Celeb-DF-v2 predates current diffusion-based face synthesis.

## Planned work

1. **Identity-disjoint re-split** of Celeb-DF-v2, and re-reporting all three models against it.
2. **Cross-dataset evaluation** — train on Celeb-DF-v2, test on FaceForensics++ and DFDC without retraining, to test whether any temporal head generalizes better than the others. In-dataset gains in this field frequently fail to survive a dataset shift, and that result is the one worth reporting.
3. AUC and McNemar significance testing across all three heads.

---

## Publication

An Arabic-language description of this work appeared in the NURAI journal. An English write-up is in preparation.

<!-- TODO: add citation and link -->

## Team

Undergraduate thesis by **Ameer Wisam Abdulsattar**, **Ali Jawad Gata**, and **Mohammed Taleb Sabbar**.

<!-- TODO: add one line naming which components you personally built — supervisors and
     admissions committees ask, and answering it before being asked reads well. -->

## Author

[GitHub](https://github.com/ameer20042005) · [Hugging Face](https://huggingface.co/ameer4wisam) · [LinkedIn](https://www.linkedin.com/in/ameer-wisam-b15027317)

## License

<!-- TODO: MIT -->
