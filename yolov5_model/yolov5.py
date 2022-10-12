# -*- coding: utf-8 -*-

import os
import json, pandas, logging

from mistk.abstract_model import AbstractModel
import mistk.log
logger = mistk.log.get_logger()

import torch
import sys
from pathlib import Path
sys.path.append("/usr/src/models/yolov5-model/yolov5")
ROOT = Path("/usr/src/models/yolov5-model/yolov5")

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

# YOLOv5Model class implements the AbstractModel class methods. 
class YOLOv5Model (AbstractModel):
    def __init__(self):
        AbstractModel.__init__(self)
        # reference to actual model 
        self._model = None
        # model properties
        self._props = None
        # model hyper parameters
        self._hparams = None
        self._objectives = None
        # model file to save trained model to
        self._yolov5_model = 'yolov5n'
        # data
        self._training_data = None
        self._testing_data = None
        # predictions
        self._predictions = []

  
    def do_initialize(self, objectives: list, props : dict, hparams : dict):
        # Initialize the model. This includes any model properties or
        # hyper parameters that the model requires.
        self._props = props or {}
        self._hparams = hparams or {}
        self._objectives = objectives
        logger.setLevel(logging.DEBUG)
        logger.info("do_initialize called")
        logger.info("Initializing the Test model with: " + str(self._props))
        
        # set model name to something other than the default
        if 'yolov5_model' in self._props:
            self._yolov5_model = self._props['yolov5_model']
        
    def do_load_data(self, dataset_map: dict): 
        logger.info("do_load_data called")
        logger.info("Loading data based on: " + str(dataset_map))
        # check for and load training data and/or test data
        if 'train' not in dataset_map and 'test' not in dataset_map:
            raise RuntimeError('No datasets provided')
        if 'train' in dataset_map:
            dataset = dataset_map['train']
            # read training dataset here and set
            self._training_data = []
            logger.info("Loaded training data: \n" + str(self._training_data))
        if 'test' in dataset_map:
            dataset = dataset_map['test']
            directory = dataset.data_path
            # read test dataset here and set
            self._testing_data= [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".jpg")]
            logger.info("Loaded test data: " + str(self._testing_data))
 

    def do_build_model(self, path=None):
        logger.info("do_build_model called")
        # Load model
        device = select_device("cpu")
        imgsz=(640, 640)
        self._model = DetectMultiBackend(os.path.join(path, self._yolov5_model + ".pt"), 
                    device=device, dnn=False, data=ROOT / 'data/coco128.yaml', fp16=False)
        stride, names, pt = self._model.stride, self._model.names, self._model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        self._model.warmup(imgsz=(1, 3, *imgsz))  # warmup
        

    def do_train(self):
        logger.info("do_train called")
        logger.info("Training model with  \n" + str(self._training_data))
        # train model code here
        
        # status update to client
        self.update_status({"trained": self._training_data})
    
    def do_save_model(self, path):
        logger.info("do_save_model called")
        path = os.path.join(path, self._model_file_name)
        logger.info("Saving model to " + path)
        # save model here
            
    def do_pause(self):
        logger.info("do_pause called")
        
        # pause model code

    def do_resume_training(self):
        logger.info("do_resume_training called")
        
        # resume training code

    def do_resume_predict(self):
        logger.info("do_resume_predict called")
        
        # resume predict code
    
    def do_predict(self):
        logger.info("do_predict called")
        self._predictions = []
        stride, names, pt = self._model.stride, self._model.names, self._model.pt
        dataset = LoadImages(self._testing_data, img_size=(640,640), stride=stride, auto=pt, vid_stride=1)
        dt = (Profile(), Profile(), Profile())
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(self._model.device)
                im = im.float()
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
            # Inference
            with dt[1]:
                pred = self._model(im, augment=False, visualize=False)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
            
            for det in pred :
                if len(det) > 0:
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        self._predictions.append(
                            {"image" : os.path.split(path)[1],
                                "label" : self._model.names[int(cls)],
                                "confidence" : float(conf),
                                "x0"    : int(xyxy[0]),
                                "y0"    : int(xyxy[1]),
                                "x1"    : int(xyxy[2]),
                                "y1"    : int(xyxy[3])})
            
        # status update to client for size of predictions
        self.update_status({"samples_predicted": len(self._predictions)})
    
    def do_save_predictions(self, dataPath):
        logger.info("do_save_predictions called")
        # predictions should be saved to predictions.csv file
        dataPath = os.path.join(dataPath, "predictions.csv")
        logger.info("Saving predictions to " + dataPath)
        import csv
        
        with open(dataPath, "w") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=["image", "label", "confidence", "x0", "y0", "x1", "y1"])
            writer.writeheader()
            for result in self._predictions:
                writer.writerow(result)
                
    def do_stream_predict(self, data_map: dict):
        logger.info("do_stream_predict called")
        logger.info("Stream prediction based on: " + json.dumps(data_map))
        predictions = {}
        
        # code to do streaming predictions here based on data_map
        
        return predictions

    def do_terminate(self):
        # terminate the model
        
        pass

    def do_reset(self):
        # result the model
        
        pass
    
    def do_generate(self):
        msg = "this model doesn't support 'generate'"
        raise NotImplementedError(msg)    
                    
    def do_save_generations(self, dataPath):
        msg = "this model doesn't support 'save_generations'"
        raise NotImplementedError(msg)
    
    # helper class to read csv data using pandas
    def read_dataset(self, data_path):
        logger.info("Loading dataset from %s", data_path)
        with open(data_path) as reader:
            dataframe = pandas.read_csv(reader, header=None)
        return dataframe.values
