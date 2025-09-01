# Entropy-Driven Curriculum for Multi-Task Training in Human Mobility Prediction

[[arxiv]()]

Human mobility prediction faces two key challenges: diverse trajectory complexity impedes efficient training, and focusing solely on next-location prediction ignores auxiliary mobility characteristics. This paper proposes an entropy-driven curriculum learning framework combined with multi-task training. The approach uses Lempel-Ziv compression to quantify trajectory predictability and organizes training from simple to complex patterns for faster convergence. Multi-task learning simultaneously predicts locations, movement distances, and directions to capture realistic mobility patterns. Experiments on the [HuMob Challenge](https://connection.mit.edu/humob-challenge-2023/) dataset show a GEO-BLEU score of 0.354 and a DTW distance of 26.15, with up to 2.92x faster convergence compared to conventional training.

## Usage

```bash
git clone --recursive https://github.com/tum-bgd/2025-ICML-1DPathConv.git
```

Detailed steps will be updated.

## Citation

```
tba
```

## License

Licensed under Apache-2.0 license (LICENSE or https://opensource.org/licenses/Apache-2.0)

