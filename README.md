# BNN-NIDS

## Usage
### Train
```shell
python train.py --train-datapath /path/to/data --test-datapath /path/to/data
```
### Inference   
Add '--coarse-norm 1' to enable hardware efficient normalization layer  
Add '--trace 1' to dump in-between tensor values
```shell
python inference.py --model /path/to/model.pth --test-datapath /path/to/data
```

## Reference
Binarization module and general model architecture:
https://github.com/itayhubara/BinaryNet.pytorch


