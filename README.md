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

To run the model in ``mistk_test_harness``, create a directory ``images`` for the images to predict on, a directory ``prediction`` to save the resulst in, and ``models`` to load the pytorch model from (included in the repository as ``yolov5n.pt``)
```
(mistk) $ mkdir images
(mistk) $ mv *.jpg images
(mistk) $ mkdir predictions
(mistk) $ mkdir models
(mistk) $ cp mistk-yolov5/yolov5n.pt models
(mistk) $ python3 -m mistk_test_harness \
    --predict images \
    --predictions-path predictions \
    --model-path models \
    --model sml-models/yolov5-model 
```
