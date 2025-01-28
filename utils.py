import cv2
import numpy as np
from easyocr import Reader

min_h = 200
min_w = 600
def enlarge_image(img: np.ndarray) -> np.ndarray:
  h, w, _ = img.shape
  scale = max(min_w / w, min_h / h)

  if scale < 1:
    return img
  
  img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

  return img

def get_threshold_plate(image: np.ndarray) -> np.ndarray:
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, blockSize=129, C=1)
  
  return image

reader = Reader(['en'])
def read_plate_text(plate_img: np.ndarray) -> str | None:
  texts = reader.readtext(plate_img)
  result = ''

  for text in texts:
    _, t, _ = text
    t = t.upper().replace(" ", "")
    result += t

  print(f"Found text on plate: {t}")
  return result

def is_valid_plate_format(plate_text: str) -> bool:
  return False

def crop_plate_text(plate_text: str) -> str | None:
  pass