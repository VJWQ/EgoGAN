# EgoGAN: Generative Adversarial Network for Future Hand Segmentation from Egocentric Video (ECCV 2022)
This is the official code release for our ECCV2022 paper on introducing a novel task of predicting a time series of future hand masks from egocentric videos, together with the first deep generative model (EgoGAN) that generate egocentric motion cues for visual anticipations.

**[[Paper](https://arxiv.org/abs/2203.11305)] [[Supplement](https://vjwq.github.io/EgoGAN/assets/EgoGAN-supp.pdf)] [[Project Page](https://vjwq.github.io/EgoGAN/)] [[Poster](https://vjwq.github.io/EgoGAN-page/assets/EgoGAN_poster.pdf)] [[Presentation](https://vjwq.github.io/EgoGAN/assets/EgoGAN_video.mp4)]**

<img src='https://vjwq.github.io/EgoGAN-page/assets/teaser.png'>


## Requirements
Our method requires the same dependencies as SlowFast. We refer to the official implementation fo [SlowFast](https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md) for installation details.
```shell
conda env create -f environment.yml
conda activate egogan
```

## Demo
<img src='https://vjwq.github.io/EgoGAN-page/assets/demo.gif'>


## Data Preparation
### Epic-Kitchen Dataset
### EGTEA Dataset
### Ego4D Dataset


## Training
```shell
python tools/run_net.py --cfg /path/to/Ego4D-Future-Hand-Prediction/configs/Ego4D/I3D_8x8_R50.yaml OUTPUT_DIR /path/to/ego4d-hand_ant/output/
```

## Evaluation
```shell
```

```shell
```

- Evaluation function
```shell

```


## Important directories and explanation
| Directory | Location | Description |
| --------- | -------- | -------- |
| cropped_videos_ant | ./slowfast/datasets/ego4dhand.py | Put your rescaled video clips in this folder |
| PATH_TO_DATA_DIR: ../data-path/ | ./configs/Ego4D/I3D_8x8_R50.yaml | Put your cropped_videos_ant folder and annotation folders under this path |
| OUTPUT_DIR: ../checkpoints/ | ./configs/Ego4D/I3D_8x8_R50.yaml  ./tools/test_net.py | Define store location of checkpoints and output file |
| SAVE_RESULTS_PATH: output.pkl | ./configs/Ego4D/I3D_8x8_R50.yaml  ./tools/test_net.py | Define output file name |

## Citation

If you use this code for your research, please cite our paper:

**Generative Adversarial Network for Future Hand Segmentation from Egocentric Video**.  
[Wenqi Jia](https://vjwq.github.io/),
[Miao Liu](https://aptx4869lm.github.io/),
[James Rehg](https://rehg.org/).  
In ECCV 2022.

Bibtex:
```
@inproceedings{jia2022generative,
  title={Generative Adversarial Network for Future Hand Segmentation from Egocentric Video},
  author={Jia, Wenqi and Liu, Miao and Rehg, James M.},
  booktitle={ECCV},
  year={2022}
}
```

## Ego4D Hand Movement Prediction Challenge
Please refer to the [future hand prediction repo](https://github.com/EGO4D/forecasting/tree/main/Ego4D-Future-Hand-Prediction) for more details! 
Check our leaderboard [here](https://eval.ai/web/challenges/challenge-page/1630/overview). 
