# deep-learning-study

The purpose of this study project is to develop a program that takes pictures of a car as input and "guesses" the model of the car and how much that model would be worth used.


## Dependencies

- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## Dataset

We use the Cars Dataset, which contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split.

You can get it from [Cars Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html):

Please put both tgz files into the folder in order to run properly. 

## Usage

### Data Pre-processing
Extract 8,144 training images, and split them by 80:20 rule (6,515 for training, 1,629 for validation):
```bash
$ python pre-process.py
```

### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### Analysis
Update "model_weights_path" in "utils.py" with your best model, and use 1,629 validation images for result analysis:
```bash
$ python analyze.py
```

### Test
```bash
$ python test.py
```

### Demo

```bash
$ python custom_demo.py --i [image_path]
```
If no argument, a sample image is used; will return the model, confidence, and median price

