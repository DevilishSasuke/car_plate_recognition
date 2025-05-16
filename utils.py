import cv2
import numpy as np
from typing import Optional, Generator, Tuple
from ultralytics.engine.results import Boxes, Results
from paddleocr import PaddleOCR
import re

min_ph = 40 # минимальная высота номера(для увеличения)
min_pw = 120 # минимальная ширина номера
plate_alphabet = "ABEKMHOPCTYX0123456789" # алфавит доступных номерных символов

def get_frames(cap: cv2.VideoCapture) -> Generator[np.ndarray, None, None]:
  while cap.isOpened():
    ret, frame = cap.read()
    if ret:
      yield frame
    else:
      break
  cap.release()
  return

def get_boxes(results: Results) -> Generator[Boxes, None, None]:
  for result in results:
            for box in result.boxes:
               yield box

  return

def enlarge_image(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max(min_pw / w, min_ph / h)

    if scale <= 1:
        return img

    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def remove_incorrect_chars(plate_text: str) -> str:
  return re.sub(fr"[^{plate_alphabet}]", "", plate_text).strip()

replacements = { "0": "O", "8": "B", "2": "Z", }
def make_replacements(text: str) -> str:
  letters = text[0] + text[4:6]
  numbers = text[1:4]
  region = text[6:] if len(text) > 6 else ""

  for k, v in replacements.items():
    letters = letters.replace(k, v)
    numbers = numbers.replace(v, k)
    region = region.replace(v, k)

  return f"{letters[0]}{numbers}{letters[1:]} {region}"

plate_letters = "ABEKMHOPCTYX"
plate_region = r" \d{2,3}$"
plate_regexes = [
    re.compile(rf"^[{plate_letters}]\d{{3}}[{plate_letters}]{{2}}{plate_region}"),  # A000AA 000
    re.compile(rf"^[{plate_letters}]{{2}}\d{{3}}{plate_region}"),                   # AA000 000
    re.compile(rf"^[{plate_letters}]{{2}}\d{{4}}{plate_region}"),                   # AA0000 000
    re.compile(rf"^\d{{4}}[{plate_letters}]{{2}}{plate_region}"),                   # 0000AA 000
]

def is_valid_plate(text: str) -> bool:
    return any(regex.match(text) for regex in plate_regexes)
