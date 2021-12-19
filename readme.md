# Feature Alignment for Robust Acoustic Scene Classification across Devices

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Results](#results)
- [References](#references)

## Background
We evaluate the proposed method on DCASE 2019 Task1b and DCASE 2020 Task1a. 
Both are used to evaluate ASC algorithms across recording devices. 
DCASE 2020 contains 15480 samples captured by 9 devices: 14400 samples recorded by 3 real devices (A, B, C) and 1080 samples of 6 simulated devices (S1-S6).
Note that the samples of the S4-S6 devices do not appear in the training set. 
DCASE 2019 contains 16560 segments, including 14400/1080/1080 samples from the devices A/B/C, respectively. 

## Install
Download the [data](https://doi.org/10.5281/zenodo.3670185) and Change the path in config.py to your own.
```sh
$ pip install -r requirements.txt
```
```sh
$ python features.sh
```
```sh
$ python train.py

```

## Results

The AdamW optimizer was used for network training, with 200 epochs on an Nvidia RTX 3090 card. The initial learning rate was set to 0.001 and the batch size was set to 64.

<img src="https://github.com/Jingqiao-Zhao/FAASC/blob/main/result.png"/>



## References
D. S. Park, W. Chan, Y. Zhang, C.-C. Chiu, B. Zoph, E. D.Cubuk, and Q. V. Le, “Specaugment: A simple data augmentation method for automatic speech recognition,” arXivpreprint arXiv:1904.08779, 2019.

Y. Tang, Y. Wang, Y. Xu, B. Shi, C. Xu, C. Xu, and C. Xu,“Beyond dropout: Feature map distortion to regularize deep neural networks,” arXiv preprint arXiv:2002.11022, 2020.

A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang,T. Weyand, M. Andreetto, and H. Adam, “Mobilenets: Efficient convolutional neural networks for mobile vision applications,” arXiv preprint arXiv:1704.04861, 2017.




