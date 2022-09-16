## OOD-Skin-Lesion
Out-of-Distribution Detection for skin lesion images

This repository contains the code for the MICCAI 2022 paper "Out-of-Distribution Detection for Long-tailed and Fine-grained Skin Lesion Images" (https://arxiv.org/abs/2206.15186)

Note that the code is only for ISIC2019 dataset as our in-house dataset could not be publicly released.

<img src="https://github.com/DevD1092/ood-skin-lesion/blob/main/figures/Fig2.png" >

<img src="https://github.com/DevD1092/ood-skin-lesion/blob/main/figures/Fig3.png" >

## Train
There are four experiments to be trained as listed below.

- Baseline model with only Cross Entropy Loss without any Mixup strategies
```sh
python train.py --loss Softmax --mixup 0
```

- Model employing mixup strategies
```sh
python train.py --loss Softmax --mixup 1
```

- Model with only Prototype Loss
```sh
python train.py --loss GCPLoss --mixup 0
```

- Model with integration of Mixup strategies with the Prototype Loss
```sh
python train.py --loss GCPLoss --mixup 1
```

As can be noted the arguments of `--loss` and `--mixup` control the different experimental settings for the methods proposed.

## Test & Evaluation
For testing the above trained models, please follow the below commands.
- For the standard Cross Entropy loss trained models with / without Mixup strategies
```sh
python val.py --loss Softmax --checkpoint <checkpoint_path> --output-filename <output_filename.csv>
```

- For the Prototype Loss trained models with / without Mixup strategies
```sh
python val.py --loss GCPLoss --checkpoint <checkpoint_path> --output-filename <output_filename.csv>
```

Here the `--checkpoint` corresponds to the checkpoint path where the trained model checkpoint has been saved and `--output-filename` corresponds to the output filename where the testing result will be stored. The output filename should be given as a .csv file.
