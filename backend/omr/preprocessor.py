import cv2
import numpy as np
import os
from pdf2image import convert_from_path

def preprocess_image(input_path: str, output_path: str):
    ext = os.path.splitext(input_path)[-1].lower()

    if ext == ".pdf":
        pages = convert_from_path(input_path, dpi=300)
        if not pages:
            raise ValueError(f"No pages found in PDF: {input_path}")
        img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Could not load image from: {input_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # can experiment with different preprocessing methods for optimal results
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(output_path, thresh)