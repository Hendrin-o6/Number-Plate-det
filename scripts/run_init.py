import sys
import traceback
import os

print('Python:', sys.version)

# 1) imports
try:
    import cv2
    import pytesseract
    from ultralytics import YOLO
    from openpyxl import Workbook, load_workbook
    print('Imports OK: cv2, pytesseract, ultralytics, openpyxl')
except Exception as e:
    print('Import error:', e)
    traceback.print_exc()
    raise SystemExit(1)

# 2) Tesseract path
try:
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    print('Tesseract cmd set to:', pytesseract.pytesseract.tesseract_cmd)
    # check version
    import subprocess
    try:
        out = subprocess.run([pytesseract.pytesseract.tesseract_cmd, '--version'], capture_output=True, text=True)
        print('Tesseract version output (first line):', out.stdout.splitlines()[0] if out.stdout else 'no output')
    except Exception as e:
        print('Could not run tesseract executable:', e)
except Exception as e:
    print('Tesseract setting failed:', e)
    traceback.print_exc()

# 3) Load YOLO model
model_path = r"c:\Users\hendr\OneDrive\Desktop\Number Plate\weights\numberplate.pt"
try:
    print('Attempting to load model from', model_path)
    model = YOLO(model_path)
    print('Model loaded OK')
except Exception as e:
    print('Model load failed:', e)
    traceback.print_exc()

# 4) Open webcam
try:
    cap = cv2.VideoCapture(0)
    print('VideoCapture created, isOpened() =', cap.isOpened())
    if cap.isOpened():
        ret, frame = cap.read()
        print('Read frame ret =', ret)
        if ret:
            print('Frame shape:', frame.shape)
        cap.release()
    else:
        print('Cap not opened; camera inaccessible or permission issue')
except Exception as e:
    print('Camera open/read failed:', e)
    traceback.print_exc()

print('Init test complete')
