# A PyTorch implementation of MPIIGaze and MPIIFaceGaze

[![MIT License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/hysts/pytorch_mpiigaze.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/hysts/pytorch_mpiigaze)

[Here](https://github.com/hysts/pytorch_mpiigaze_demo) is a demo program.
See also [this repo](https://github.com/hysts/pl_gaze_estimation).

## Requirements

* Linux (Tested on Ubuntu only)
* Python >= 3.7

```bash
pip install -r requirements.txt
```


## Download the dataset and preprocess it

### MPIIGaze

```bash
bash scripts/download_mpiigaze_dataset.sh
python tools/preprocess_mpiigaze.py --dataset datasets/MPIIGaze -o datasets/
```

### MPIIFaceGaze

```bash
bash scripts/download_mpiifacegaze_dataset.sh
python tools/preprocess_mpiifacegaze.py --dataset datasets/MPIIFaceGaze_normalized -o datasets/
```


## Usage

This repository uses [YACS](https://github.com/rbgirshick/yacs) for
configuration management.
Default parameters are specified in
[`gaze_estimation/config/defaults.py`](gaze_estimation/config/defaults.py)
(which is not supposed to be modified directly).
You can overwrite those default parameters using a YAML file like
[`configs/mpiigaze/lenet_train.yaml`](configs/mpiigaze/lenet_train.yaml).


### Training and Evaluation

By running the following code, you can train a model using all the
data except the person with ID 0, and run test on that person.

```bash
python train.py --config configs/mpiigaze/lenet_train.yaml
python evaluate.py --config configs/mpiigaze/lenet_eval.yaml
```

Using [`scripts/run_all_mpiigaze_lenet.sh`](scripts/run_all_mpiigaze_lenet.sh) and
[`scripts/run_all_mpiigaze_resnet_preact.sh`](scripts/run_all_mpiigaze_resnet_preact.sh),
you can run all training and evaluation for LeNet and ResNet-8 with
default parameters.


## Results

### MPIIGaze

| Model           | Mean Test Angle Error [degree] | Training Time |
|:----------------|:------------------------------:|--------------:|
| LeNet           |              6.52              |  3.5 s/epoch  |
| ResNet-preact-8 |              5.73              |   7 s/epoch   |

The training time is the value when using GTX 1080Ti.

![](figures/mpiigaze/lenet.png)

![](figures/mpiigaze/resnet_preact_8.png)

### MPIIFaceGaze

| Model     | Mean Test Angle Error [degree] | Training Time |
|:----------|:------------------------------:|--------------:|
| AlexNet   |              5.06              |  135 s/epoch  |
| ResNet-14 |              4.83              |   62 s/epoch  |

The training time is the value when using GTX 1080Ti.

![](figures/mpiifacegaze/alexnet.png)

![](figures/mpiifacegaze/resnet_simple.png)




### Demo

This demo program runs gaze estimation on the video from a webcam.

1. Download the dlib pretrained model for landmark detection.

    ```bash
    bash scripts/download_dlib_model.sh
    ```

2. Calibrate the camera.

    Save the calibration result in the same format as the sample
    file [`data/calib/sample_params.yaml`](data/calib/sample_params.yaml).

4. Run demo.

    Specify the model path and the path of the camera calibration results
    in the configuration file as in
    [`configs/demo_mpiigaze_resnet.yaml`](configs/demo_mpiigaze_resnet.yaml).

    ```bash
    python demo.py --config configs/demo_mpiigaze_resnet.yaml
    ```

## Related repos

- https://github.com/hysts/pl_gaze_estimation
- https://github.com/hysts/pytorch_mpiigaze_demo

## References

* Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "Appearance-based Gaze Estimation in the Wild." Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. [arXiv:1504.02863](https://arxiv.org/abs/1504.02863), [Project Page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/)
* Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation." Proc. of the IEEE Conference on Computer Vision and Pattern Recognition Workshops(CVPRW), 2017. [arXiv:1611.08860](https://arxiv.org/abs/1611.08860), [Project Page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation/)
* Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation." IEEE transactions on pattern analysis and machine intelligence 41 (2017). [arXiv:1711.09017](https://arxiv.org/abs/1711.09017)
* Zhang, Xucong, Yusuke Sugano, and Andreas Bulling. "Evaluation of Appearance-Based Methods and Implications for Gaze-Based Applications." Proc. ACM SIGCHI Conference on Human Factors in Computing Systems (CHI), 2019. [arXiv](https://arxiv.org/abs/1901.10906), [code](https://git.hcics.simtech.uni-stuttgart.de/public-projects/opengaze)
