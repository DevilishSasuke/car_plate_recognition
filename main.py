import os
import cv2

img_dir = os.path.join(os.getcwd(), 'img')
target_size = (1080, 1920) # 1080p image size

def handle_image(img_name):
  img_path = os.path.join(img_dir, img_name)
  image = cv2.imread(img_path)
  if image is None:
    return None

  return resize_with_padding(image)

def resize_with_padding(image):
  height, width = image.shape[:2] # размер изображения
  if (height, width) == target_size:
    return image # если размер не требует изменений
  
  # значение для изменения размера
  scale = min(target_size[0] / height, target_size[1] / width)
  new_height, new_width = int(height * scale), int(width * scale)
 
  # измененное изображение
  resized_img = cv2.resize(image, (new_width, new_height))

  # значения границ
  top = (target_size[0] - new_height) // 2
  bottom = target_size[0] - new_height - top
  left = (target_size[1] - new_width) // 2
  right = target_size[1] - new_width - left 
  # добавляем рамки для изображения
  return cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

def main():
  extensions = ('.png', '.jpg', '.jpeg', '.bmp') # расширения изображений
  resized_dir = os.path.join(os.getcwd(), 'handled')

  for filename in os.listdir('img'):
    if filename.lower().endswith(extensions):
      cv2.imwrite(os.path.join(resized_dir, filename), handle_image(filename))


if __name__ == "__main__":
  main()
  