# STADB_ReID
Official implementation of "STADB: A Self-Thresholding Attention Guided ADB Network for Person Re-identification", Bo Jiang, Sheng Wang, Xiao Wang, Aihua Zheng [[arxiv](https://arxiv.org/abs/2007.03584)] 

## Background and Motivation
Recently, Batch DropBlock network (BDB) has demonstrated its effectiveness on person image representation and re-identification task via feature erasing. However, BDB drops the features randomly which may lead to sub-optimal results. In this paper, we propose a novel Self-Thresholding attention guided Adaptive DropBlock network (STADB) for person re-ID which can adaptively erase the most discriminative regions. Specifically, STADB first obtains an attention map by channel-wise pooling and returns a drop mask by thresholding the attention map. Then, the input features and self-thresholding attention guided drop mask are multiplied to generate the dropped feature maps. In addition, STADB utilizes the spatial and channel attention to learn a better feature map and iteratively trains the feature dropping module for person re-ID. Experiments on several benchmark datasets demonstrate that the proposed STADB outperforms many other related methods for person re-ID. 


## Framework 
![fig-1](https://github.com/wangxiao5791509/STADB_ReID/blob/main/framework.png)


## Feature Visualization && Results 
![fig-2](https://github.com/wangxiao5791509/STADB_ReID/blob/main/results.png)

## Environment  require

pip install python3
pip install cython
pip install torch
pip install torchvision
pip install scikit-learn
pip install tensorboardX
pip install fire


## Training 

### Traning Market1501
```bash
python main_reid.py train --save_dir='./pytorch-ckpt/market-bfe' --max_epoch=400 --eval_step=30 --dataset=market1501 --test_batch=128 --train_batch=128 --optim=adam --adjust_lr
```


## Testing 
### Test Market1501
```bash
python3 main_reid.py train --save_dir='./pytorch-ckpt/market_bfe' --model_name=bfe --train_batch=32 --test_batch=32 --dataset=market1501 --pretrained_model='./pytorch-ckpt/market_bfe/best_model.pth.tar' --evaluate
```

## Citation 
If you find this work useful for your research, please cite this paper: 
```
@article{bojiang2020stadb,
  title={STADB: A Self-Thresholding Attention Guided ADB Network for Person Re-identification},
  author={Bo Jiang, Sheng Wang, Xiao Wang, Aihua Zheng},
  journal={arXiv:2007.03584},
  year={2020}
}
```

If you have any questions about this work, please submit an issue or send emails to the authors. Thanks for your attention!



























