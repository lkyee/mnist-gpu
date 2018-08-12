import cv2 as cv


img_org=cv.imread("2.png")
img_conv=255-img_org
cv.imwrite("2_1.png",img_conv)
cv.imshow("img_org",img_conv)
cv.waitKey(0)
cv.destroyAllWindows()
