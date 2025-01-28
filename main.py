import os
import cv2
from ultralytics import YOLO
from utils import *

correct_plates = set()
wrong_plates = set()

def main():
  model = YOLO("model/plates.pt")

  test_dir = os.path.join(os.getcwd(), 'test')
  for filename in os.listdir(test_dir):
    img_path = os.path.join(test_dir, filename)
    img = cv2.imread(img_path)
    results = model.predict(source=img, save=False, conf=0.75)
    for result in results:
      if result.boxes.shape[0] == 0:
        continue
      x1, y1, x2, y2 = map(int, result.boxes.xyxy[0].cpu().numpy())
      plate_img = img[y1:y2, x1:x2]

      plate_img = enlarge_image(plate_img)
      plate_img = get_threshold_plate(plate_img)

      plate_text = read_plate_text(plate_img)
      plate_text = crop_plate_text(plate_text)

      cv2.imshow(" ", plate_img)
      cv2.waitKey(0)

      if plate_text is not None and result != '':
        if is_valid_plate_format(plate_text):
          correct_plates.add(plate_text)
        else:
          wrong_plates.add(plate_text)
      
  return


if __name__ == "__main__":
  main()
  