from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image, ImageChops
import pytesseract
import numpy as np
import argparse
import shutil
import cv2
import collections
import os
import json
multiplier = 2


class Record:
    def __init__(self, name, image, date):
        self.keys = [name]
        self.images = [image]
        self.date = date.split("to")
        self.prepareImage()

    def prepareImage(self):
        for image in self.images:
            # cv2.imshow('a', image)
            cv2.rectangle(image, (0, 0),
                          (120, image.shape[0]), (255, 255, 255), -1)
            image = cv2.putText(image, str(self.date[0]).upper().strip(), (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 2)
            image = cv2.putText(image, "TO", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 2)
            image = cv2.putText(image, str(self.date[1]).upper().strip(), (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 2)

    def addKey(self, key):
        self.keys.append(key)

    def hasKey(self, key) -> bool:
        return (key in self.keys)

    def getImage(self):
        self.prepareImage()
        return self.images[0]

    def __str__(self):
        return "-".join(self.keys)


class RecordCollection:
    def __init__(self):
        self.records = list()

    def addRecord(self, rcd: Record):
        # if self.findRecordByName(rcd.name)
        # for name in rcd.name:
        # if not self.findRecordByName(rcd.keys):
        self.records.append(rcd)
        pass

    def findRecordByName(self, names: []) -> Record:
        for rcd in self.records:
            for name in names:
                if rcd.hasKey(name):
                    return rcd
        print("Record Not found")


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
    # cv2.imshow('s', img[0:60, 100:500])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img[0:60, 100:500]


def getDateBlock(img):
    return img[0:45, :]


def getRecordsCollection(pdfPath, rcd) -> RecordCollection:
    pages = convert_from_path(pdfPath, dpi=200)
    for x in range(len(pages)):
        pages[x].save("tmpimg/out{}.jpg".format(x), 'JPEG')
        page = cv2.imread("tmpimg/out{}.jpg".format(x))
        date = getImageText(getDateBlock(page)).replace(
            "salary sheet", "").replace("()", "").strip().replace(", ", " 20")
        records = getRecordImages(page)
        for s in records:
            names = getImageText(getNameBlock(s)).splitlines()
            tmpRecord = Record(" ".join(names), s, date)
            if len(names) > 1:
                [tmpRecord.addKey(names[x]) for x in range(1, len(names))]
            rcd.addRecord(tmpRecord)
    return rcd


def createDirs():
    names = ['tmpimg', 'records', 'tmp']
    for name in names:
        shutil.rmtree('./{}/'.format(name))
        os.makedirs("./{}/".format(name))


pdfs = os.listdir("./pdfs/")
createDirs()

dcts = RecordCollection()

print("processing")

for pdf in pdfs:
    print(pdf)
    getRecordsCollection('pdfs/{}'.format(pdf), dcts)

namesList = collections.defaultdict(list)

for record in dcts.records:
    namesList[record.keys[0]].append(record.getImage())

for k in namesList.keys():
    cv2.imwrite("records/{}.jpg".format(str(k).replace("/", "-")),
                cv2.vconcat(namesList[k]))

print("processed")
# cv2.destroyAllWindows()
