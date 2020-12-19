# UNIXKD

This repo is the implementation of CIFAR100 part in paper [Computation-Efficient Knowledge Distillation via Uncertainty-Aware Mixup](https://arxiv.org/abs/2012.09413).

<img src="https://github.com/xuguodong03/UNIXKD/raw/master/frm.jpg" width="100%" height="100%">

## Prerequisite
This repo is tested with Ubuntu 16.04.5, Python 3.7, PyTorch 1.5.0, CUDA 10.2.
Make sure to install pytorch, torchvision, tensorboardX, numpy before using this repo.

## Running

### Teacher Training
An example of teacher training is:
```
python teacher.py --arch wrn_40_2 --lr 0.05 --gpu-id 0
```
where you can specify the architecture via flag `--arch`

You can also download all the pre-trained teacher models [here](https://drive.google.com/drive/folders/1vJ0VdeFRd9a50ObbBD8SslBtmqmj8p8r?usp=sharing). 
If you want to run `student.py` directly, you have to re-organise the directory. For instance, when you download *vgg13.pth*, you have to make a directory for it, say *teacher_vgg13*, and then make a new directory *ckpt* inside *teacher_vgg13*. Move the *vgg13.pth* into *teacher_vgg13/ckpt* and rename it as *best.pth*. If you want a simpler way to use pre-trained model, you can edit the code in `student_v0.py` (line 96).

### Student Training
An example of student training is:
```
python student_v0.py --teacher-path ./experiments/teacher_wrn_40_2/ --student-arch wrn_16_2 --lr 0.05 --gpu-id 0
```
The meanings of flags are:
> `--teacher-path`: teacher's checkpoint path. Automatically search the checkpoint containing 'best' keyword in its name.

> `--student-arch`: student's architecture.

All the commands can be found in `command/command.sh`

## Results (Top-1 Acc) on CIFAR100

The accuracies are slightly different from those in the paper. Average on 4 runs.

### Cross-Architecture

| Teacher <br> Student | vgg13 <br> MobieleNetV2 | ResNet50 <br> MobileNetV2 | ResNet50 <br> vgg8 | resnet32x4 <br> ShuffleV1 |  resnet32x4 <br> ShuffleV2 | wrn40-2 <br> ShuffleV1|
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:--------------------:|:-----------:|:-------------:|
| Teacher <br> Student |    75.38 <br> 65.79    |    79.10 <br> 65.79    |    79.10 <br> 70.68    |    79.63 <br> 70.77     | 79.63 <br> 73.12 | 76.46 <br> 70.77 |
| Ft/Fs | 38.17 | 174.00 | 13.56 | 27.16 | 23.49 | 8.22 |
| KD <br> Computation | 67.94 <br> 100% | 68.33 <br> 100% | 73.43 <br> 100% | 74.52 <br> 100% | 75.07 <br> 100% | 76.04 <br> 100% |
| UNIXKD <br> Computation | 68.09 <br> 77.49% | 68.76 <br> 75.57% | 74.02 <br> 81.43% | 76.48 <br> 78.43% | 76.86 <br> 78.92% | 77.06 <br> 84.79% |

### Simialr-Architecture

| Teacher <br> Student | wrn40-2 <br> wrn16-2 | wrn40-2 <br> wrn40-1 | resnet56 <br> resnet20 | resnet32x4 <br> resnet8x4 |  vgg13 <br> vgg8 |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:--------------------:|:-----------:|
| Teacher <br> Student |    76.46 <br> 73.64    |    76.46 <br> 72.24    |    73.44 <br> 69.63    |     79.63 <br> 72.51     | 75.38 <br> 70.68 |
| Ft/Fs | 3.25 | 3.93 | 3.06  | 6.12 | 2.97 | 4.14 |
| KD <br> Computation| 75.40 <br> 100% | 73.77 <br> 100% | 70.72<br> 100% | 73.34 <br> 100% | 73.38 <br> 100% | 72.10 <br> 100% |
| UNIXKD <br> Computation | 75.40 <br> 94.06% | 74.38 <br> 91.88% | 70.41 <br> 94.76% | 74.86 <br> 87.32% |73.55 <br> 95.10% | 73.06 <br> 91.29% |


## Citation
If you find this repo useful for your research, please consider citing the paper
```
@misc{xu2020unixkd,
      title={Computation-Efficient Knowledge Distillation via Uncertainty-Aware Mixup}, 
      author={Guodong Xu and Ziwei Liu and Chen Change Loy},
      year={2020},
      eprint={2012.09413},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

