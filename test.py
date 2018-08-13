import cv2
from prepare_data import *

im = cv2.imread("./2016/BaieDeception16011408300000.jpg")
im = resizeAndPad(img=im, size=(48, 48))
cv2.imshow('image', im)
cv2.waitKey(0)
cv2.destroyAllWindows()
