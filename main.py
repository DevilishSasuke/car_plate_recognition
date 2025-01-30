import cv2
from ultralytics import YOLO
from typing import Optional, Dict
from utils import *

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
acceptable_classes = [2, 6] # Допустимые классы, которые модель будет распознавать
def detect_cars_and_buses_on_video(video_path: str, 
                                   output_path: Optional[str] = "output.mp4", 
                                   car_model_path: Optional[str] = "yolo11m.pt",
                                   plate_model_path: Optional[str] = "model/plates.pt",
                                   car_model: Optional[YOLO] = None,
                                   plate_model: Optional[YOLO] = None):
    """
    Обнаруживает автомобили и автобусы на видео с помощью YOLO, выделяя их синим цветом.
    Обнаруживает номера автомобилей и автобусов на видео с помощью YOLO, выделяя их красным цветом.

    Args:
        video_path (str): Путь к входному видео.
        output_path (str, optional): Путь для сохранения выходного видео. По умолчанию "output.mp4".
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
      raise ValueError(f"Неправильный путь к видео: {video_path}")
    
    out = create_video_writer(cap, output_path)

    car_names = car_model.names
    # Чтение кадров с видео
    for frame in get_frames(cap):
        processed_frame = process_frame(frame, car_model, plate_model, car_names)

        # Запись обработанного кадра в выходное видео
        out.write(processed_frame)

    # Закрытие объектов
    out.release()
    cv2.destroyAllWindows()
    print(f"Готово. Видео сохранено в {output_path}")

def process_frame(frame: np.ndarray, car_model: YOLO, plate_model: YOLO, 
                  car_names:  Dict[int, str]) -> np.ndarray:
  # Обнаружение объектов с помощью YOLO
  car_results = car_model.predict(source=frame, save=False)

  # Отображение обнаруженных автомобилей и автобусов на кадре
  for cbox in get_boxes(car_results):
    if cbox.cls.item() in acceptable_classes:
      draw_rectangle(frame, cbox, title=car_names[int(cbox.cls.item())], color=blue)
      
      x1, y1, x2, y2 = map(int, cbox.xyxy[0].tolist())
      vehicle_img = frame[y1:y2, x1:x2]

      # Обнаружение номерных знаков на обнаруженных машинах
      plate_results = plate_model.predict(source=vehicle_img, save=False, conf=0.5)
      for pbox in get_boxes(plate_results):
            draw_rectangle(vehicle_img, pbox, title="plate", color=red)

  return frame


if __name__ == "__main__":
    video_path = "test2.mp4"  # Замените на путь к вашему видео
    detect_cars_and_buses_on_video(video_path)