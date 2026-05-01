"""
Standalone test: load a sample plate image from log/plates and run the OCR pipeline.
This isolates the crash to OCR or preprocessing without needing a webcam/model.
"""
import os
import cv2
import pytesseract
import re
import traceback
from glob import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLATES_DIR = os.path.join(BASE_DIR, 'log', 'plates')

# Set tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Find first plate image
images = sorted(glob(os.path.join(PLATES_DIR, '*.jpg')))
if not images:
    print(f'No images found in {PLATES_DIR}')
    raise SystemExit(1)

test_img_path = images[0]
print(f'Testing OCR on: {test_img_path}')

# Load image
plate_img = cv2.imread(test_img_path)
if plate_img is None:
    print(f'Failed to read {test_img_path}')
    raise SystemExit(1)

print(f'Image shape: {plate_img.shape}')

# Replicate the perform_ocr function from main.py
def perform_ocr(img):
    """Try several preprocessing steps and tesseract configs, return best text or empty string."""
    candidates = []
    try:
        # common tesseract configs for plates: whitelist alphanum, OEM 3
        configs = [
            '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ]

        # prepare scaled images to help OCR on small crops
        scales = [1.0, 1.5, 2.0]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f'  Gray conversion OK, shape: {gray.shape}')
        
        # basic denoise
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        print(f'  Bilateral filter OK, shape: {blur.shape}')

        attempt = 0
        for s in scales:
            if s != 1.0:
                h, w = blur.shape[:2]
                new = cv2.resize(blur, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
            else:
                new = blur

            # try simple threshold and adaptive
            th1 = cv2.threshold(new, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            th2 = cv2.adaptiveThreshold(new, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

            for prep_name, prep in (('gray', new), ('otsu', th1), ('adapt', th2)):
                for cfg_idx, cfg in enumerate(configs):
                    attempt += 1
                    print(f'    Attempt {attempt}: scale={s}, prep={prep_name}, cfg_idx={cfg_idx}')
                    try:
                        data = pytesseract.image_to_data(prep, output_type=pytesseract.Output.DICT, config=cfg)
                        print(f'      image_to_data OK')
                    except pytesseract.TesseractNotFoundError as e:
                        print(f'      Tesseract not found: {e}')
                        return "OCR_FAILED"
                    except Exception as e:
                        print(f'      image_to_data failed: {e}')
                        traceback.print_exc()
                        raise

                    # join text pieces and compute mean confidence (ignore -1)
                    texts = [t.strip() for t in data.get('text', []) if t.strip()]
                    print(f'      texts: {texts}')
                    
                    confs_raw = data.get('conf', [])
                    print(f'      conf values: {confs_raw[:10]}')  # first 10
                    
                    confs = [int(c) for c in confs_raw if c.strip().isdigit() and int(c) >= 0]
                    text_join = " ".join(texts)
                    mean_conf = (sum(confs) / len(confs)) if confs else 0
                    print(f'      result: "{text_join}", mean_conf={mean_conf}')
                    candidates.append((text_join, mean_conf))

        print(f'Total candidates: {len(candidates)}')
        
        # pick highest mean confidence with non-empty text
        candidates = [c for c in candidates if c[0]]
        print(f'Candidates with non-empty text: {len(candidates)}')
        
        if not candidates:
            return ""
        
        best = max(candidates, key=lambda x: x[1])
        print(f'Best candidate: "{best[0]}", conf={best[1]}')
        
        # cleanup: keep alphanum and common separators
        cleaned = re.sub(r'[^A-Z0-9\- ]', '', best[0].upper())
        result = cleaned.strip()
        print(f'Cleaned result: "{result}"')
        return result
    except pytesseract.TesseractNotFoundError:
        return "OCR_FAILED"

print('\nRunning perform_ocr...')
try:
    text = perform_ocr(plate_img)
    print(f'\nFinal OCR result: "{text}"')
except Exception as e:
    print(f'\nOCR function crashed: {e}')
    traceback.print_exc()
    raise SystemExit(1)
