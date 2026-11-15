# RelaVPR: Relation-Based Knowledge Distillation for Efficient Visual Place Recognition

Official implementation of RelaVPR: Relation-Based Knowledge Distillation for Efficient Visual Place Recognition.


## 🛠️Setup

This repo follows the framework of [CricaVPR](https://github.com/Lu-Feng/CricaVPR/tree/main). 

We use [GSV-Cities](https://github.com/amaralibey/gsv-cities) for training, and the [Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark) for evaluation. You can download the GSV-Cities datasets [HERE](https://www.kaggle.com/datasets/amaralibey/gsv-cities), and refer to [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader) to prepare test datasets.

The test dataset should be organized in a directory tree as such:
```bash
├── datasets_vg
    └── datasets
        └── pitts30k
            └── images
                ├── train
                │   ├── database
                │   └── queries
                ├── val
                │   ├── database
                │   └── queries
                └── test
                    ├── database
                    └── queries
```

## 🏋️‍♂Training

Before we begin, please download the ImAge weights [LINK](https://huggingface.co/fenglu96/ImAge4VPR/resolve/main/ImAge_Merged.pth) and Dinov2-small weights [LINK](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth).

Please use the following code for distillation :

```bash
python distillation.py
```

## 🧩Evaluation

Pleas download the model (single teacher) weights [LINK]() into the directory, and use the following code for evaluation:
```bash
python eval.py
```

Pleas download the model (two teacher) weights [LINK]() into the directory, and use the following code for evaluation:
```bash
python eval_two_teacher.py
```

## Acknowledgements

We thank the authors of [DINOv2](https://github.com/facebookresearch/dinov2), ImAge, [EDTformer](https://github.com/Tong-Jin01/EDTformer), [CricaVPR](https://github.com/Lu-Feng/CricaVPR), [BoQ](https://github.com/amaralibey/Bag-of-Queries), and [SALAD](https://github.com/serizba/salad) for their contributions. We also thank [GSV-Cities](https://github.com/amaralibey/gsv-cities), the [Visual Geo-localization Benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark), and [VPR-datasets-downloader](https://github.com/gmberton/VPR-datasets-downloader) for their dataset and evaluation resources.



