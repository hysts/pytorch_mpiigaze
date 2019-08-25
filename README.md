# A PyTorch implementation of MPIIGaze

## Requirements

* Python >= 3.7

```bash
pip install -r requirements.txt
```


## Download the dataset and preprocess it

```bash
bash scripts/download_mpiigaze_dataset.sh
python tools/preprocess_data.py --dataset datasets/MPIIGaze -o datasets/preprocessed
```


## Usage

This repository uses [YACS](https://github.com/rbgirshick/yacs) for
configuration management.
Default parameters are specified in
[`mpiigaze/config/defaults.py`](mpiigaze/config/defaults.py) (which is not
supposed to be modified directly).
You can overwrite those default parameters using a YAML file like
[`configs/lenet_train.yaml`](configs/lenet_train.yaml).


### Training and Evaluation

By running the following code, you can train a model using all the
data except the person with ID 0, and run test on that person.

```bash
python train.py --config configs/lenet_train.yaml
python evaluate.py --config configs/lenet_eval.yaml
```

Using [`scripts/run_all_lenet.sh`](scripts/run_all_lenet.sh) and
[`scripts/run_all_resnet_preact.sh`](scripts/run_all_resnet_preact.sh),
you can run all training and evaluation for LeNet and ResNet-8 with
default parameters.


## Results

| Model           | Mean Test Angle Error [degree] | Training Time |
|:----------------|:------------------------------:|--------------:|
| LeNet           |              6.52              |  3.5 s/epoch  |
| ResNet-preact-8 |              5.73              |   7 s/epoch   |

The training time is the value when using GTX 1080Ti.

![](figures/lenet.png)

![](figures/resnet_preact_8.png)


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
    in the configuration file as in [`configs/demo.yaml`](configs/demo.yaml).

    ```bash
    python demo.py --config configs/demo.yaml
    ```


## References

* Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "Appearance-based Gaze Estimation in the Wild." Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015. [arXiv:1504.02863](https://arxiv.org/abs/1504.02863), [Project Page](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/)
* Zhang, Xucong, Yusuke Sugano, Mario Fritz, and Andreas Bulling. "MPIIGaze: Real-World Dataset and Deep Appearance-Based Gaze Estimation." IEEE transactions on pattern analysis and machine intelligence 41 (2017). [arXiv:1711.09017](https://arxiv.org/abs/1711.09017)



