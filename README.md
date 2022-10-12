# Demonstration of building a MISTK model with a YOLOv5

This repository is a basic MISTK model using a YOLOv5 model as an inference only example.

To build this model, clone the repository and pull the YOLOv5 submodule and make the docker image
```
$ git clone https://github.com/shelbydavis/mistk-yolov5.git
$ cd mistk-yolov5
$ git submodule init
$ git submodule update
$ make
```

After the Docker image is built, it will be available locally as ``sml-model/mistk-yolov5`` to use within the ``mistk_test_harness``

```
$ python -m venv mistk
$ source mistk/bin/activate
(mistk) $ python -m pip install --upgrade pip wheel
(mistk) $ python -m pip install https://github.com/mistkml/mistk/releases/download/1.0.0/mistk-1.0.0-py3-none-any.whl
(mistk) $ python -m pip install https://github.com/mistkml/mistk/releases/download/1.0.0/mistk_test_harness-1.0.0-py3-none-any.whl
```