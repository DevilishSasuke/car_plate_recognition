import cv2
import asyncio
import time
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from utils import *
from db import PlateDatabase


start_time = time.time()

acceptable_classes = [2, 6]  # машины и автобусы
executor = ThreadPoolExecutor(max_workers=2)
db = PlateDatabase()
recent_plates = defaultdict(lambda: datetime.min)
# Время в секундах, когда повторяющийся знак не будет вновь записан
plate_cooldown = 20 
reader = PaddleOCR(use_angle_cls=False, 
                   lang='en', 
                   show_log=False) # объект для чтения текста с изображения

async def process_video(video_path: str, 
                        car_model_path: Optional[str] = "yolo11m.pt",
                        plate_model_path: Optional[str] = "model/plates.pt",
                        car_model: Optional[YOLO] = None,
                        plate_model: Optional[YOLO] = None):
  """
  Обнаруживает автомобили и автобусы на видео с помощью YOLO, выделяя их синим цветом.
  Обнаруживает номера автомобилей и автобусов на видео с помощью YOLO, выделяя их красным цветом.

  Args:
    video_path (str): Путь к входному видео.
    car_model_path (str, optional): Путь к файлу модели YOLO. По умолчанию "yolo11m.pt".
    plate_model_path (str, optional) = Путь к файлу модели детекции номеров. По умолчанию: "model/plates.pt",
    car_model (YOLO, optional): Готовая модель для определения автомобилей на видео,
    plate_model (YOLO, optional): = Готовая модель для определения номерных знаков на видео):
  """
  await db.init()
  await db.clear_table()

  # Загрузка предварительно обученной модели YOLO
  if car_model is None:
    car_model = YOLO(car_model_path)
  # Загрузка собственной модели для детекции номерных знаков
  if plate_model is None:
    plate_model = YOLO(plate_model_path)

  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
      raise ValueError(f"Невозможно открыть видео: {video_path}")

  frame_count = 0
  for frame in get_frames(cap):
    frame_count += 1
    if frame_count % 3 != 0:
      continue

    # Обнаружение объектов с помощью YOLO
    car_results = car_model.predict(source=frame, save=False, verbose=False)

    # Отображение обнаруженных автомобилей и автобусов на кадре
    for cbox in get_boxes(car_results):
      if cbox.cls.item() in acceptable_classes:
        
        x1, y1, x2, y2 = map(int, cbox.xyxy[0].tolist())
        vehicle_img = frame[y1:y2, x1:x2]

        # Обнаружение номерных знаков на обнаруженных машинах
        plate_img = detect_plates_in_vehicle(vehicle_img, plate_model)
        if plate_img is None:
          continue
        plate_text = read_plate_text(plate_img)
        timestamp = datetime.now()
        if plate_text:
          await handle_plate_text(plate_text, timestamp)

  await db.close()
  end_time = time.time()
  elapsed_time = end_time - start_time
  print(f"The task took {elapsed_time:.2f} seconds to complete.")

def detect_plates_in_vehicle(vehicle_img: np.ndarray, 
                             plate_model: YOLO):
  plate_results = plate_model.predict(source=vehicle_img, save=False, conf=0.7, verbose=False)
  for pbox in get_boxes(plate_results):
    x1_, y1_, x2_, y2_ = map(int, pbox.xyxy[0].tolist())
    plate_img = vehicle_img[y1_:y2_, x1_:x2_]
    return plate_img
  return None

def read_plate_text(plate_img: np.ndarray) -> str:
  texts = reader.ocr(plate_img, cls=False)
  result  = ""
  if not texts or not texts[0]:
    return ""
  for text in texts[0]:
    t = text[1][0].upper().replace(" ", "")
    result += t
  
  return result

async def handle_plate_text(text: str, timestamp: datetime = datetime.now()):
  plate_text = remove_incorrect_chars(text)
  plate_text = make_replacements(plate_text) \
    if len(plate_text) >= 6 else plate_text
  if not is_valid_plate(plate_text):
    return None
  
  recent_time = recent_plates[plate_text]
  if (timestamp - recent_time).total_seconds() < plate_cooldown:
    return plate_text

  await db.insert_plate(plate_text, timestamp)
  recent_plates[plate_text] = timestamp
  return plate_text


if __name__ == "__main__":
  asyncio.run(process_video(
    video_path="test/test2.mp4",
    car_model_path="model/yolo11m.pt",
    plate_model_path="model/plates.pt"
  ))