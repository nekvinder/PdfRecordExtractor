from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image, ImageChops
import pytesseract
import numpy as np
import argparse
import cv2
import os
multiplier = 2


class PdfRecord:
    def __init__(self, name, image):
        self.keys = [name]
        self.image = image

    def addKey(self, key):
        self.keys.append(key)

    def hasKey(self, key) -> bool:
        return (key in self.keys)

    def getImage(self):
        return self.image

    def __str__(self):
        return "-".join(self.keys)


def getImageText(img) -> str:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = str(pytesseract.image_to_string(gray))
    return(text.lower())


def getRecordImages(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 500, 150, apertureSize=5)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 1000)
    boxes, images = [], []
    for line in lines:
        for r, theta in line:
            boxes.append(int((np.sin(theta)*r) + 1000*(np.cos(theta))))
    boxes = sorted(boxes)
    boxes = [boxes[x] for x in range(0, len(boxes), 2)]
    for i in range(1, len(boxes)-1):
        images.append(img[boxes[i+1]-140:boxes[i+1], ])
    return images


def getNameBlock(img):
    return img[0:60, 100:500]


def getRecordsDict(pdfPath):
    rcd = []
    images = convert_from_path(pdfPath, dpi=200)
    for x in range(len(images)):
        images[x].save("tmpimg/out{}.jpg".format(x), 'JPEG')
        img = cv2.imread("tmpimg/out{}.jpg".format(x))
        records = getRecordImages(img)
        for s in records:
            names = getImageText(getNameBlock(s)).splitlines()
            tmpRecord = PdfRecord(names[0], s)
            if len(names) > 1:
                [tmpRecord.addKey(names[x]) for x in range(1, len(names))]
            rcd.append(tmpRecord)
    return rcd


def findRecord(records: [], name):
    for rcd in records:
        if rcd.hasKey(name):
            return rcd
    print("Record Not found")


dcts = getRecordsDict('salary.pdf')
print([dct.keys for dct in dcts])
# print(findRecord(dcts, 'BHOLA MAHTO').getImage())
# cv2.imshow("s", s)
# cv2.waitKey(1000)
# break

# cv2.destroyAllWindows()
