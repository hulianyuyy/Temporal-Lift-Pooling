# Temporal Lift Pooling
This repo holds codes of the paper: Temporal Lift Pooling for Continuous Sign Language Recognition.(ECCV 2022) [[paper]](https://arxiv.org/abs/2207.08734)

This repo is based on [VAC (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Min_Visual_Alignment_Constraint_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html). Many thanks for their great work!

Our training and inference procedure is modified from VAC (ICCV 2021). If you are familiar with VAC, you can play with TLP easily!
## Prerequisites

- This project is implemented in Pytorch (>1.8). Thus please install Pytorch first.

- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode.

- sclite [[kaldi-asr/kaldi]](https://github.com/kaldi-asr/kaldi), install kaldi tool to get sclite for evaluation. After installation, create a soft link toward the sclite:    
  `ln -s PATH_TO_KALDI/tools/sctk-2.4.10/bin/sclite ./software/sclite`

- [SeanNaren/warp-ctc](https://github.com/SeanNaren/warp-ctc) for ctc supervision.

## Implementation
The implementation for TLP is given in [./modules/tconv.py](https://github.com/hulianyuyy/Temporal-Lift-Pooling/blob/main/modules/tconv.py). You can choose to use TLP or max pooling or average pooling in line 77-79. 

We notice that the kernel sizes of predictor and updater in lift pooling are set as 3 practically (which we wrongly report as 5 in the camera-ready version)

You can flexibly use TLP for other temporal tasks, e.g. action recogniton or video captioning.

## Data Preparation
You can choose any one of following datasets to verify the effectiveness of TLP.

### PHOENIX2014 dataset
1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/). Our experiments based on phoenix-2014.v3.tar.gz.

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET/phoenix2014-release ./dataset/phoenix2014`

3. The original image sequence is 210x260, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python data_preprocess.py --process-image --multiprocessing
   ```

### PHOENIX2014-T dataset
1. Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET/PHOENIX-2014-T-release-v3/PHOENIX-2014-T ./dataset/phoenix2014-T`

3. The original image sequence is 210x260, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python data_preprocess-T.py --process-image --multiprocessing
   ```

### CSL dataset
The results of TLP on CSL dataset is placed in the supplementary material.

1. Request the CSL Dataset from this website [[download link]](https://ustc-slr.github.io/openresources/cslr-dataset-2015/index.html)

2. After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.   
   `ln -s PATH_TO_DATASET ./dataset/CSL`

3. The original image sequence is 1280x720, we resize it to 256x256 for augmentation. Run the following command to generate gloss dict and resize image sequence.     

   ```bash
   cd ./preprocess
   python data_preprocess-CSL.py --process-image --multiprocessing
   ``` 
## Inference

### PHOENIX2014 dataset

| Backbone | Dev WER  | Test WER  | Pretrained model                                             |
| -------- | ---------- | ----------- | ------------------------------------------------------------ |
| Baseline | 21.2%      | 22.3%       |  ------------------------------------------------------------ | 
| ResNet18 | 19.7%      | 20.8%       | [[Baidu]](https://pan.baidu.com/s/1QRws8gylNzlpXvU52VCLww) (passwd: tsa2)<br />[[Google Drive]](https://drive.google.com/file/d/1uCIYCz0O7twKG1k_BE9sZ5Q1hga4DXRI/view?usp=sharing) |

### PHOENIX2014-T dataset

| Backbone | Dev WER  | Test WER  | Pretrained model                                             |
| -------- | ---------- | ----------- | ------------------------------------------------------------ |
| Baseline | 21.1%      | 22.8%       |  ------------------------------------------------------------ | 
| ResNet18 | 19.4%      | 21.2%       | [[Baidu]](https://pan.baidu.com/s/1o8IvZhFuTWM9pZI1U8Y2YQ) (passwd: c6cq)<br />[[Google Drive]](https://drive.google.com/file/d/1xFv0ttMQdU6SMvncEnHT0OT6osUCSXVK/view?usp=sharing) |

### CSL dataset

| Backbone |  WER  | Pretrained model                                             |
| -------- | ----------- | ------------------------------------------------------------ |
| Baseline | 7.3%      | ------------------------------------------------------------ | 
| ResNet18 | 1.8%   | [[Baidu]](https://pan.baidu.com/s/1s9pRFSpmr8mrSqXlU9hzqg) (passwd: e1md)<br />[[Google Drive]](https://drive.google.com/file/d/1U0Bnl9E711nfzEZ5FyWFPXF4gv8Bw0Fy/view?usp=sharing) |

​	To evaluate the pretrained model, (if you test on the CSL dataset, you should first change the kernel size of predictor and updater in lift pooling to 5 with padding 2 (Line 12 and Line 19 in [./modules/tconv.py](https://github.com/hulianyuyy/Temporal-Lift-Pooling/blob/main/modules/tconv.py))), run the command below：   
`python main.py --device your_device --load-weights path_to_weight.pt --phase test`

### Training

The priorities of configuration files are: command line > config file > default values of argparse. To train the SLR model on phoenix14, run the command below:

`python main.py --device your_device`

Note that you can choose the target dataset from phoenix2014/phoenix2014-T/CSL in line 3 in ./config/baseline.yaml.

If you want to train your model on the CSL dataset, please remind to (1) change the training epochs to 20, with steps of [10, 15] to decrease the learning rate in configs/ baseline.yaml. (2) change the kernel size of predictor and updater in lift pooling to 5 with padding 2 (Line 12 and Line 19 in [./modules/tconv.py](https://github.com/hulianyuyy/Temporal-Lift-Pooling/blob/main/modules/tconv.py)).
 
### Citation

If you find this repo useful in your research works, please consider citing:

```latex
@inproceedings{hu2022temporal,
  title={Temporal lift pooling for continuous sign language recognition},
  author={Hu, Lianyu and Gao, Liqing and Liu, Zekang and Feng, Wei},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XXXV},
  pages={511--527},
  year={2022},
  organization={Springer}
}
```
