import sys
import traceback
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO

model_path = r"c:\Users\hendr\OneDrive\Desktop\Number Plate\weights\numberplate.pt"

print("Python:", sys.version)
print("cv2 version:", cv2.__version__)
print("pytesseract cmd:", pytesseract.pytesseract.tesseract_cmd)

try:
    model = YOLO(model_path)
    print("Model loaded successfully.")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    results = model(img)
    print("Inference completed. Results type:", type(results))
    print(results)
except Exception as e:
    print("Exception during model load/infer:")
    traceback.print_exc()
