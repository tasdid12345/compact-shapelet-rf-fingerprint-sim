# Multi-length I/Q shapelets + gradient boosting for RF fingerprinting under domain shift

## Motivation
The paper "Generalizable and interpretable RF fingerprinting with shapelet-enhanced large language models" (2026) combines variable-length two-dimensional I/Q shapelets with a frozen pre-trained LLM backbone. This experiment asks how far a compact shapelet front end can go without any LLM, using a data-adaptive shapelet choice and gradient boosting.

## Question
Can a compact, data-adaptive multi-length I/Q shapelet model with gradient boosting, and no LLM, beat a fixed-window random-shapelet baseline at identifying simulated wireless devices under a domain shift?

## Dataset
Self-generated, seeded numpy simulation (no download) of QPSK I/Q frames (256 samples) from 10 devices, each with its own IQ gain/phase imbalance, DC offset, carrier-frequency offset and PA nonlinearity. The training domain has high SNR and a mild channel. The held-out test domain has lower SNR, new random multipath channels and new phase offsets. This is **not** the paper's data (it stands in for its six non-identified datasets).

## Baseline vs improved
- **Baseline** (not the paper's method): random fixed-length 2D I/Q shapelets cut from training frames, min-distance features, and an sklearn MLP.
- **Improved**: the same distance code and shapelet budget, but variable-length shapelets chosen by k-means on training windows at several scales, plus a few phase-invariant statistics, and a class-balanced LightGBM tuned by cross-validation on the training set.

The main metric is accuracy on the shifted test domain. Secondary metrics are macro-F1 and in-domain accuracy, averaged over seeds 42, 43 and 44.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> T Zhao, J Zhang, H Xu, X Sun, J Dai (2026). *Generalizable and interpretable RF fingerprinting with shapelet-enhanced large language models*. arXiv preprint arXiv. https://arxiv.org/abs/2602.03035

Paper author: Prof. Jun Dai, Worcester Polytechnic Institute (jdai@wpi.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
