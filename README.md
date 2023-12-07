<h1 align="center">
MetaScript: Few-Shot Handwritten Chinese Content Generation via Style-based Generative Adversarial Networks
</h1>
<p align="center">
    Project of AI3604 Computer Vision, 2023 Fall, SJTU
    <br />
    <a href="https://github.com/Bujiazi"><strong>Jiazi Bu</strong></a>
    &nbsp;
    <a href="https://github.com/IApple233"><strong>Qirui Li</strong></a>
    &nbsp;
    <a href="https://github.com/Loping151"><strong>Kailing Wang</strong></a>
    &nbsp;
    <a href="https://github.com/xxyQwQ"><strong>Xiangyuan Xue</strong></a>
    &nbsp;
    <a href="https://github.com/wdask"><strong>Zhiyuan Zhang</strong></a>
    <br />
</p>

## Requirements

To ensure the code runs correctly, following packages are required:

* `python`
* `hydra`
* `opencv`
* `pytorch`

You can install them following the instructions below.

* Create a new conda environment and activate it:
  
    ```bash
    conda create -n metaswap python=3.10
    conda activate metaswap
    ```

* Install [pytorch](https://pytorch.org/get-started/previous-versions/) with appropriate CUDA version, e.g.
  
    ```bash
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
    ```

* Install `hydra` and `opencv`:
  
    ```bash
    pip install hydra-core
    pip install opencv-python
    ```

Latest version is recommended for all the packages, but make sure that your CUDA version is compatible with your `pytorch`.
