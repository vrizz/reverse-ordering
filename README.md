
# Reverse Ordering Techniques for Attention-Based Channel Prediction

🚀 Welcome to the repository for the paper "Reverse Ordering Techniques for Attention-Based Channel Prediction"! This repository contains the code to reproduce the main results of our research work.

## Abstract

In our study, we introduce two models for predicting time-varying channels: Transformer-RPE and Seq2Seq-attn-R. Both outperform existing methods in channel prediction accuracy across different noise levels and generalize to unseen sequence lengths.

## Table of Contents

- [Dataset Download](#dataset-download)
- [Usage](#usage)
- [Citation](#citation)

## Dataset Download

The channel dataset can be downloaded [here](https://drive.google.com/file/d/1gTkx6_WYjz9-9l5IUdxURpZ_4DNo4pqi/view?usp=sharing) as a `.zip` file.

After you have downloaded and extracted the dataset, you have put it in the same folder of the source code.

## Usage

You can reproduce the results with these commands:

```
python main.py transformer-rpe
```
or
```
python main.py seq2seq-attn-r
```

You will see this on your terminal for the Transformer-RPE:
<details>
  <summary>Click to expand test results</summary>

<div style="height: 200px; overflow: auto;">

```console
Testing l=16 and delta=4 (same as training): SNR=-5dB   NMSE=0.4747
Testing l=8 and delta=2: SNR=-5dB   NMSE=0.5138
Testing l=14 and delta=6: SNR=-5dB   NMSE=0.574
---------------------------------------------------------------------------
Testing l=16 and delta=4 (same as training): SNR=0dB   NMSE=0.2915
Testing l=8 and delta=2: SNR=0dB   NMSE=0.3225
Testing l=14 and delta=6: SNR=0dB   NMSE=0.4014
---------------------------------------------------------------------------
Testing l=16 and delta=4 (same as training): SNR=5dB   NMSE=0.1808
Testing l=8 and delta=2: SNR=5dB   NMSE=0.1742
Testing l=14 and delta=6: SNR=5dB   NMSE=0.289
---------------------------------------------------------------------------
Testing l=16 and delta=4 (same as training): SNR=10dB   NMSE=0.1127
Testing l=8 and delta=2: SNR=10dB   NMSE=0.0945
Testing l=14 and delta=6: SNR=10dB   NMSE=0.203
---------------------------------------------------------------------------
Testing l=16 and delta=4 (same as training): SNR=15dB   NMSE=0.0711
Testing l=8 and delta=2: SNR=15dB   NMSE=0.0583
Testing l=14 and delta=6: SNR=15dB   NMSE=0.1373
---------------------------------------------------------------------------
Testing l=16 and delta=4 (same as training): SNR=20dB   NMSE=0.0448
Testing l=8 and delta=2: SNR=20dB   NMSE=0.0388
Testing l=14 and delta=6: SNR=20dB   NMSE=0.1074
---------------------------------------------------------------------------
```
</div>
</details>

And this for the Seq2Seq-attn-R:

<details>
  <summary>Click to expand test results</summary>

<div style="height: 200px; overflow: auto;">

```console
Testing l=16 and delta=4 (same as training): SNR=-5dB   NMSE=0.5075
Testing l=8 and delta=2: SNR=-5dB   NMSE=0.519
Testing l=14 and delta=6: SNR=-5dB   NMSE=0.6161
---------------------------------------------------------------------------
Testing l=16 and delta=4 (same as training): SNR=0dB   NMSE=0.3164
Testing l=8 and delta=2: SNR=0dB   NMSE=0.3041
Testing l=14 and delta=6: SNR=0dB   NMSE=0.4243
---------------------------------------------------------------------------
Testing l=16 and delta=4 (same as training): SNR=5dB   NMSE=0.1993
Testing l=8 and delta=2: SNR=5dB   NMSE=0.1876
Testing l=14 and delta=6: SNR=5dB   NMSE=0.2988
---------------------------------------------------------------------------
Testing l=16 and delta=4 (same as training): SNR=10dB   NMSE=0.1217
Testing l=8 and delta=2: SNR=10dB   NMSE=0.118
Testing l=14 and delta=6: SNR=10dB   NMSE=0.2083
---------------------------------------------------------------------------
Testing l=16 and delta=4 (same as training): SNR=15dB   NMSE=0.0739
Testing l=8 and delta=2: SNR=15dB   NMSE=0.0747
Testing l=14 and delta=6: SNR=15dB   NMSE=0.148
---------------------------------------------------------------------------
Testing l=16 and delta=4 (same as training): SNR=20dB   NMSE=0.0466
Testing l=8 and delta=2: SNR=20dB   NMSE=0.0523
Testing l=14 and delta=6: SNR=20dB   NMSE=0.119
---------------------------------------------------------------------------
```
</div>
</details>

## Citation
📚 If you are using this code and/or the provided dataset for your research, please cite

```bibtex
@article{rizzello2024reverse,
    author={Rizzello, Valentina and B{\"o}ck, Benedikt and Joham, Michael and Utschick, Wolfgang},
    journal={IEEE Open Journal of Signal Processing}, 
    title={{Reverse Ordering Techniques for Attention-Based Channel Prediction}}, 
    year={2024},
    volume={},
    number={},
    pages={1-9},
}
```
