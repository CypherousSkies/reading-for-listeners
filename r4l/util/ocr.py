import cv2
import numpy as np
from pdf2image import convert_from_path
from transformers import AutoTokenizer, AutoModel
import random
import math


def sample_borders(pdfpath):
    print("converting to imgs")
    imgs = convert_from_path(pdfpath)
    print("sampling pages")
    imgs = random.sample(imgs, math.ceil(len(imgs) / 10))  # sample a 10% of pages to see what the shape looks like
    imgs = [np.array(img)[:, :, ::-1].copy() for img in imgs]  # convert to opencv
    borders = [get_border(img) for img in imgs]
    print(borders)
    p1 = (np.min(p1[0] for p1, _ in borders), np.min(p1[1] for p1, _ in borders))
    p2 = (np.max(p2[0] for _, p2 in borders), np.max(p2[1] for _, p2 in borders))
    print(p1,p2)
    return p1, p2


def get_border(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    mask = np.ones(img.shape, np.uint8) * 255
    for cnt in contours:
        size = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if 10000 > size > 500 and w * 2.5 > h:
            cv2.drawContours(mask, [cnt], -1, (0, 0, 0), -1)
    kernel = np.ones((50, 50), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    gray_op = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
    _, threshold_op = cv2.threshold(gray_op, 150, 255, cv2.THRESH_BINARY_INV)
    contours_op, hierarchy_op = cv2.findContours(threshold_op, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours_op, key=cv2.contourArea)
    _, _, angle = rect = cv2.minAreaRect(cnt)
    (h, w) = img.shape[:2]
    (center) = (w // 2, h // 2)
    box = cv2.boxPoints(rect)
    a, b, c, d = box = np.int0(box)
    bound = [a, b, c, d]
    bound = np.array(bound)
    p1 = (bound[:, 0].min(), bound[:, 1].min())
    p2 = (bound[:, 0].max(), bound[:, 1].max())
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    cv2.imshow('edited', img)
    #print(p1, p2)
    return (p1, p2)


class TrOCR:
    def __init__(self, model_name="nielsr/trocr-base-printed"):
        self.tokenizer = AutoTokenizer(model_name)
        self.model = AutoModel(model_name)
