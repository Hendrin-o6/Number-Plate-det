import os
import csv
import cv2
import pytesseract
import numpy as np
import re
from glob import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLATES_DIR = os.path.join(BASE_DIR, 'log', 'plates')
OUT_CSV = os.path.join(BASE_DIR, 'log', 'ocr_diagnostics.csv')

if not os.path.exists(PLATES_DIR):
    print('No plates directory found at', PLATES_DIR)
    raise SystemExit(1)

configs = [
    '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    '--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    '--oem 3 --psm 6'
]

scales = [1.0, 1.5, 2.0]

def ocr_candidates(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    candidates = []
    for s in scales:
        if s != 1.0:
            h, w = blur.shape[:2]
            scaled = cv2.resize(blur, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
        else:
            scaled = blur

        th_otsu = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        th_adapt = cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)

        for prep_name, prep in (('gray', scaled), ('otsu', th_otsu), ('adapt', th_adapt)):
            for cfg in configs:
                try:
                    data = pytesseract.image_to_data(prep, output_type=pytesseract.Output.DICT, config=cfg)
                except pytesseract.TesseractNotFoundError:
                    return [('OCR_FAILED', 0, cfg, prep_name, s)]

                texts = [t.strip() for t in data.get('text', []) if t.strip()]
                confs = [int(c) for c in data.get('conf', []) if c.strip().lstrip('-').isdigit()]
                mean_conf = (sum(confs) / len(confs)) if confs else 0
                text_join = ' '.join(texts)
                cleaned = re.sub(r'[^A-Z0-9\- ]', '', text_join.upper())
                candidates.append((cleaned.strip(), mean_conf, cfg, prep_name, s))

    return candidates

def diagnose_all():
    images = sorted(glob(os.path.join(PLATES_DIR, '*.jpg')))
    if not images:
        print('No images found in', PLATES_DIR)
        return

    rows = []
    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print('Failed to read', img_path)
            continue

        cands = ocr_candidates(img)
        if not cands:
            best = ('', 0, '', '', 1.0)
        else:
            # choose by highest mean_conf, then length
            best = max(cands, key=lambda x: (x[1], len(x[0])))

        text, conf, cfg, prep, scale = best
        print(f'{os.path.basename(img_path)} -> "{text}" (conf={conf:.1f}) cfg="{cfg}" prep={prep} scale={scale}')
        rows.append([os.path.basename(img_path), text, f'{conf:.1f}', cfg, prep, scale, img_path])

    # save CSV
    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image', 'best_text', 'mean_conf', 'config', 'prep', 'scale', 'path'])
        w.writerows(rows)

    print('\nWrote diagnostics to', OUT_CSV)

if __name__ == '__main__':
    diagnose_all()
