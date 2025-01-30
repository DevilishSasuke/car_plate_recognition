import cv2
import numpy as np
from typing import Optional, Generator, Tuple
from ultralytics.engine.results import Boxes, Results

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
                    title: Optional[str] = "none",
                    color: Optional[Tuple[int, int, int]] = (0, 255, 0)):
  
  x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
  confidence = box.conf[0].item()
  label = f"{title}: {confidence:.2f}"

  cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
  cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def get_boxes(results: Results) -> Generator[Boxes, None, None]:
  for result in results:
            for box in result.boxes:
               yield box

  return

def create_video_writer(cap: cv2.VideoCapture, output_path: str) -> cv2.VideoWriter:
   # Параметры видео
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Создание объекта для записи видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

  