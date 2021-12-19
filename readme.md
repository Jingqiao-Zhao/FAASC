# Feature Alignment for Robust Acoustic Scene Classification across Devices

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Results](#results)
- [References](#references)

## Background
The [DCASE2020](http://dcase.community/) dataset consists of 10 classes sounds captured in airport, shopping mall, metro station, pedestrian street, public,square, street traffic, tram, bus, metro and park. This challenge provides two datasets, development and evaluation, for algorithm development. The sub-task B of TAU Urban Acoustic Scenes 2020 dataset contains 40 hour audio recordings which are balanced between classes and recorded at 48kHz sampling rate with 24-bit resolution in stereo. Each sound recording was spitted into 10-second audio samples.

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

<img width="400" height="300" src="https://github.com/Jingqiao-Zhao/FAASC/blob/master/result.png"/>



## References
D. S. Park, W. Chan, Y. Zhang, C.-C. Chiu, B. Zoph, E. D.Cubuk, and Q. V. Le, “Specaugment: A simple data augmentation method for automatic speech recognition,” arXivpreprint arXiv:1904.08779, 2019.

Y. Tang, Y. Wang, Y. Xu, B. Shi, C. Xu, C. Xu, and C. Xu,“Beyond dropout: Feature map distortion to regularize deep neural networks,” arXiv preprint arXiv:2002.11022, 2020.

A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang,T. Weyand, M. Andreetto, and H. Adam, “Mobilenets: Efficient convolutional neural networks for mobile vision applications,” arXiv preprint arXiv:1704.04861, 2017.




