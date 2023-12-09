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

## Training

The dataset used for training is mainly adapted from [CASIA-HWDB-1.1](http://www.nlpr.ia.ac.cn/databases/handwriting/Offline_database.html). We put the characters by the same writer into the same directory. The folder name represents the writer and the file name represents the character. We render the template characters from [Source-Han-Sans](https://github.com/adobe-fonts/source-han-sans). You can download the dataset [here](https://drive.google.com/file/d/1iwa6RfWIPXzb9J4ASp1H4ljYvwy8Va5c/view) and extract it. If you have installed `gdown`, this step can be done by running the following commands:

```bash
gdown 1iwa6RfWIPXzb9J4ASp1H4ljYvwy8Va5c -O dataset.zip
unzip dataset.zip
```

You can also build the dataset by yourself for customization. The directory structure is as follows:

```
dataset
├── script
│   ├── writer_1
│   │   ├── character_1.png
│   │   ├── character_2.png
│   │   └── ...
│   ├── writer_2
│   │   ├── character_1.png
│   │   ├── character_2.png
│   │   └── ...
│   └── ...
└── template
    ├── character_1.png
    ├── character_2.png
    └── ...
```

Then modify the configuration file in the `config` directory, where `dataset_path` must be correctly set as the path to your dataset. You can also modify the hyperparameters or create a new configuration file as you like, but remember to modify the `hydra` arguments in `training.py` accordingly. Here we provide a template configuration file `config/training.yaml`. The batch size is set to 16 by default, which requires at least 6GB GPU memory.

Run the following command to train the model:

```bash
python training.py
```
