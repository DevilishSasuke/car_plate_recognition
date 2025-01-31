import cv2
import numpy as np
from typing import Optional, Generator, Tuple
from ultralytics.engine.results import Boxes, Results
from easyocr import Reader
from re import sub as re_sub

min_ph = 200 # минимальная высота номера(для увеличения)
min_pw = 600 # минимальная ширина номера
plate_alphabet = "ABEKMHOPCTYX0123456789" # алфавит доступных номерных символов
reader = Reader(["en"]) # объект для чтения текста с изображения

def create_video_writer(cap: cv2.VideoCapture, output_path: str) -> cv2.VideoWriter:
  # Параметры видео
  frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = cap.get(cv2.CAP_PROP_FPS)

  # Создание объекта для записи видео
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

def get_frames(cap: cv2.VideoCapture) -> Generator[np.ndarray, None, None]:
  while cap.isOpened():
    ret, frame = cap.read()
    if ret:
      yield frame
    else:
      break
  cap.release()
  return

def draw_rectangle(frame: np.ndarray,
                    box: Boxes, 
                    label: Optional[str] = "none",
                    color: Optional[Tuple[int, int, int]] = (0, 255, 0)):
  
  x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
  confidence = box.conf[0].item()
  label = f"{label}: {confidence:.2f}"

  cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
  cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def get_boxes(results: Results) -> Generator[Boxes, None, None]:
  for result in results:
            for box in result.boxes:
               yield box

  return

def enlarge_image(img: np.ndarray) -> np.ndarray:
  h, w, _ = img.shape
  scale = max(min_pw / w, min_ph / h)

  if scale < 1:
    return img
  
  return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def read_plate_text(plate_img: np.ndarray) -> str:
  texts = reader.readtext(plate_img)
  result = ""

  for text in texts:
    _, t, _ = text
    t = t.upper().replace(" ", "")
    result += t

  if not result or len(result) == 0:
    return ""
  return result

def remove_incorrect_chars(plate_text: str) -> str:
   return re_sub(fr"[^{plate_alphabet}]", "", plate_text).strip()