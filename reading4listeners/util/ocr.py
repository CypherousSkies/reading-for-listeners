import math
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import random

import cv2
import numpy as np
from pdf2image import convert_from_path
from tqdm import tqdm


# https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr/
# TODO: test sample_borders & integrate it into TrOCR pipeline
def sample_borders(pdfpath):
    print("converting to imgs")
    imgs = convert_from_path(pdfpath)
    print("sampling pages")
    imgs = random.sample(imgs, math.ceil(len(imgs) / 10))  # sample a 10% of pages to see what the shape looks like
    imgs = [np.array(img)[:, :, ::-1].copy() for img in imgs]  # convert to opencv
    borders = [border(img) for img in imgs]
    print(borders)
    p1 = (min(p1[0] for p1, _ in borders), min(p1[1] for p1, _ in borders))
    p2 = (max(p2[0] for _, p2 in borders), max(p2[1] for _, p2 in borders))
    print(p1, p2)
    return p1, p2


def border(img):
    # Preprocessing the image starts

    # Convert the image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # Specify structure shape and kernel size.
    # Kernel size increases or decreases the area
    # of the rectangle to be detected.
    # A smaller value like (10, 10) will detect
    # each word instead of a sentence.
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)

    boxes = [cv2.boundingRect(cnt) for cnt in contours]
    print(boxes)
    mean_width = sum(w for _, _, w, _ in boxes) / len(boxes)
    tolerance = 0.1
    boxes = [(x, y, w, h) for x, y, w, h in boxes if
             mean_width * (1 - tolerance) <= w and w <= mean_width * (1 + tolerance)]
    p1 = (min(x for x, _, _, _ in boxes), min(y for _, y, _, _ in boxes))
    p2 = (max(w for _, _, w, _ in boxes) + p1[0], max(h for _, _, _, h in boxes) + p1[1])
    return p1, p2


class TrOCR:
    def __init__(self, model_name="microsoft/trocr-base-printed"):
        try:  # Use cached version if possible (making offline-mode default)
            self.processor = TrOCRProcessor.from_pretrained(model_name, local_files_only=True)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name, local_files_only=True)
        except:  # Models not cached
            print("> Downloading TrOCR models")
            self.processor = TrOCRProcessor.from_pretrained(model_name, force_download=True)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name, force_download=True)

    def _get_pages(self, fpath):
        imgs = convert_from_path(fpath)
        # TODO: Crop imgs to just main body of text
        return imgs

    def _trocr(self, image):
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

    def extract_text(self, fpath):
        if fpath[-3:] != 'pdf':
            raise Exception("TrOCR is only defined on PDFs")
        pages = self._get_pages(fpath)
        text = ""
        for page in tqdm(pages):
            gen_text = self._trocr(page)
            text += gen_text
        return text
