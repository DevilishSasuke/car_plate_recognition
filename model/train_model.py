def main():
  from ultralytics import YOLO
  from os import getcwd as osgetcwd
  from os.path import join as osjoin

  model = YOLO("yolov8s.yaml")

  config_path = osjoin(osgetcwd(), 'config.yaml')
  results = model.train(data=config_path, 
                        epochs=100,
                        lr0=0.005,
                        lrf=0.01,
                        warmup_bias_lr=0.05,
                        optimizer='AdamW',
                        )

if __name__ == "__main__":
  main()