import cv2
import asyncio
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO
from utils import *
import time

start_time = time.time()

acceptable_classes = [2, 6]  # машины и автобусы
executor = ThreadPoolExecutor(max_workers=2)
plates_texts = []

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

  # Загрузка предварительно обученной модели YOLO
  if car_model is None:
    car_model = YOLO(car_model_path)
  # Загрузка собственной модели для детекции номерных знаков
  if plate_model is None:
    plate_model = YOLO(plate_model_path)

  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
      raise ValueError(f"Невозможно открыть видео: {video_path}")

  tasks = []
  frame_count = 0

  for frame in get_frames(cap):
    frame_count += 1

    if frame_count % 3 != 0:
      continue

    # Обнаружение объектов с помощью YOLO
    car_results = car_model.predict(source=frame, save=False)

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
        if plate_text:
          tasks.append(handle_plate_text(plate_text))

    if len(tasks) >= 10:  # пакетная обработка
      await asyncio.gather(*tasks)
      tasks.clear()

  # Обработка оставшихся задач
  if tasks:
    await asyncio.gather(*tasks)

  with open("recognized_plates.txt", "w", encoding="utf-8") as f:
    for plate in plates_texts:
      if plate.strip():  # фильтрация пустых строк
        f.write(plate + "\n")

  end_time = time.time()
  elapsed_time = end_time - start_time
  print(f"The task took {elapsed_time:.2f} seconds to complete.")

def detect_plates_in_vehicle(vehicle_img: np.ndarray, 
                             plate_model: YOLO):
  plate_results = plate_model.predict(source=vehicle_img, save=False, conf=0.7)
  for pbox in get_boxes(plate_results):
    x1_, y1_, x2_, y2_ = map(int, pbox.xyxy[0].tolist())
    plate_img = vehicle_img[y1_:y2_, x1_:x2_]
    return plate_img
  return None

async def handle_plate_text(text: str):
  corrected_text = remove_incorrect_chars(text)
  corrected_text = make_replacements(corrected_text) \
    if len(corrected_text) >= 6 else corrected_text
  plates_texts.append(corrected_text)
  return corrected_text

async def async_recognize_plate(plate_img):
  result = await asyncio.get_event_loop().run_in_executor(executor, sync_recognize_plate, plate_img)
  plates_texts.append(result)
  return

def sync_recognize_plate(plate_img):
  plate_img = enlarge_image(plate_img)
  text = read_plate_text(plate_img)
  
  if text:
    corrected_text = remove_incorrect_chars(text)
    return make_replacements(corrected_text) \
      if len(corrected_text) >= 6 else corrected_text
  return ""


if __name__ == "__main__":
  asyncio.run(process_video(
    video_path="test/test2.mp4",
    car_model_path="model/yolo11m.pt",
    plate_model_path="model/plates.pt"
  ))