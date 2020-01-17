import cv2
import os
files = os.listdir("./records/")
record = []
for file in files:
    record.append(cv2.imread("records/{}".format(file)))

cv2.imwrite("final.jpg", cv2.vconcat(record))
cv2.waitKey(0)
cv2.destroyAllWindows()
