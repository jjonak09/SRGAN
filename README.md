# Keras SRGAN
Implementation of SRGAN(Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network) with keras.

Paper: [here](https://arxiv.org/abs/1609.04802)

# Result

![SRGAN](./result.jpg)

left image size: 64x64(I got this image from my implementation of [DRAGAN-keras](https://github.com/jjonak09/DRAGAN-keras))

center image size: 256x256(super resolution)

right image size: 256x256 (Bicubic補間)

Bicubic補間より綺麗に拡大できているが、色がくすんだ感じになっている。次はESRGANに手を出してみる。
## Environment
- OS: Windows 10
- CPU: Intel(R) Core(TM)i7-8700
- GPU: NVIDIA GTX1060 6GB
- RAM: 16GB
