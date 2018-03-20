import numpy as np
import sys
import cv2

from train import *

image = cv2.imread('test.png')
cv2.imshow('originial', image)
cv2.waitKey()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gary image', gray)
cv2.waitKey()

blurred = cv2.GaussianBlur(gray, (5,5), 0)
cv2.imshow('blurred img', blurred)
cv2.waitKey()

edged = cv2.Canny(blurred, 30,150)
cv2.imshow("edged image", edged)
cv2.waitKey()
#find contours
_ ,contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#print(contours)
#sort out the controues frm left to righ
contours = sorted(contours, key = x_cord_contour, reverse = False)
full_number = []

#loopong over the contours
for c in contours:
    (x,y,w,h) = cv2.boundingRect(c)
    cv2.drawContours(image, contours,-1, (0,255,0), 3)
    cv2.imshow("Countors_img", image)


    if w >=5  and h>=25:
        roi = blurred[y:y+h, x:x+w]
        knn = cv2.ml.KNearest_create()
        
        ret, roi=cv2.threshold(roi,127,255,cv2.THRESH_BINARY_INV)
        #roi=int(roi)
        print(type(roi))
        squared = makeSquare(roi)
        final = resize_to_pixel(20,squared)
        cv2.imshow("final_img", final)
        final_array = final.reshape((1,400))
        final_array = final_array.astype(np.float32)
        ret, result, neighbours, dist = knn.findNearest(final_array,k=1)
        number = str(int(float(result[0])))
        full_number.append(number)
        #draw the rectanngle arround the image we classified
        cv2.rectangle(image, (x,y), (x+w, y+h), (0,0,255), 2)
        cv2.putText(image, number, (x, y+170),cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0),2)
        cv2.imshow("and texted image", image)
        cv2.waitKey()
         
    
cv2.destroyAllWindows()
print("the number is:" + ''.join(full_number))

