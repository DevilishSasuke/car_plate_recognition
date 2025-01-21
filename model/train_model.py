from ultralytics import YOLO
from os import getcwd as osgetcwd
from os.path import join as osjoin


model = YOLO("yolov8n.yaml")

config_path = osjoin(osgetcwd(), 'config.yaml')
resutls = model.train(data=config_path, 
                      epochs=100,
                      device=0,
                      patience=10,
                      )