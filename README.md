# SeA
PyTorch Implementation for Our ECCV'24 Paper: "SeA: Semantic Adversarial Augmentation for Last Layer Features from Unsupervised Representation Learning"

## Requirements
* Python 3.9
* PyTorch 1.12

## Usage:
SeA with features extracted from pre-trained models
```
python main.py --train-feat-path /path/to/train/feat --train-label-path /path/to/train/label --test-feat-path /path/to/test/feat --test-label-path /path/to/test/label
```

## Citation
If you use the package in your research, please cite our paper:
```
@inproceedings{qian2024sea,
  author    = {Qi Qian and
               Yuanhong Xu and
               Juhua Hu},
  title     = {SeA: Semantic Adversarial Augmentation for Last Layer Features from Unsupervised Representation Learning},
  booktitle = {The 18th European Conference on Computer Vision, {ECCV} 2024},
  year      = {2024}
}
