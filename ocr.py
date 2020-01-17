from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image, ImageChops
import pytesseract
import numpy as np
import argparse
import cv2
import collections
import os
import json
multiplier = 2


class Record:
    def __init__(self, name, image, date):
        self.keys = [name]
        self.image = image
        self.date = date.split("to")
        cv2.rectangle(self.image, (0, 0),
                      (120, self.image.shape[0]), (255, 255, 255), -1)
        self.image = cv2.putText(self.image, str(self.date[0]).upper().strip(), (10, 20),
                                 cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 2)
        self.image = cv2.putText(self.image, "TO", (50, 50),
                                 cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 2)
        self.image = cv2.putText(self.image, str(self.date[1]).upper().strip(), (10, 80),
                                 cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 2)

        # cv2.imshow("test", self.image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit(0)

    def addKey(self, key):
        self.keys.append(key)

    def hasKey(self, key) -> bool:
        return (key in self.keys)

    def getImage(self):
        return self.image

    def __str__(self):
        return "-".join(self.keys)


class RecordsHolder:
    def __init__(self):
        pass


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


def getDateBlock(img):
    return img[0:45, :]


def getRecordsDict(pdfPath):
    rcd = []
    images = convert_from_path(pdfPath, dpi=200)
    for x in range(len(images)):
        images[x].save("tmpimg/out{}.jpg".format(x), 'JPEG')
        img = cv2.imread("tmpimg/out{}.jpg".format(x))
        date = getImageText(getDateBlock(img)).replace(
            "salary sheet", "").replace("()", "").strip().replace(", ", " 20")
        records = getRecordImages(img)
        for s in records:
            names = getImageText(getNameBlock(s)).splitlines()
            tmpRecord = Record(names[0], s, date)
            if len(names) > 1:
                [tmpRecord.addKey(names[x]) for x in range(1, len(names))]
            rcd.append(tmpRecord)
    return rcd


def findRecord(records: [], name):
    for rcd in records:
        if rcd.hasKey(name):
            return rcd
    print("Record Not found")


files = os.listdir("./pdfs/")
dcts = []

print("processing")
for file in files:
    print(file)
    dcts.append(getRecordsDict('pdfs/{}'.format(file)))

namesList = collections.defaultdict(list)
print("processed")

for dctx in dcts:
    for dct in dctx:
        for name in dct.keys:
            namesList[name].append(dct.getImage())

for k in namesList.keys():
    cv2.imwrite("records/{}.jpg".format(k), cv2.vconcat(namesList[k]))


# cv2.destroyAllWindows()
