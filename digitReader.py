import cv2
import numpy as np

#
image = cv2.imread('nodigits.png')
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small = cv2.pyrDown(image)
cv2.imshow('Digitimg', small)
cv2.waitKey(0)
cv2.destroyAllWindows()

#split the data into 50X100X20X20
cells = [np.hsplit(row, 100) for row in np.vsplit(gray_img, 50)]

#convert it into numpy array
X = np.array(cells)
print("The shape of our cells array:" + str(X.shape))

#split the full datssr into two segment
train = X[:, :70].reshape(-1, 400).astype(np.float32)
teat = X[:, 70:100].reshape(-1,400).astype(np.float32)

#create lable for train and test data
k = [0,1,2,3,4,5,6,7,8,9]
train_labels = np.repeat(k,350)[:,np.newaxis]
test_labels = np.repeat(k,150)[:, np.newaxis]

#initieate our knn algo
knn = cv2.ml.KNearest_create()
knn.train(train,cv2.ml.ROW_SAMPLE, train_labels)
res , result, neighbour, distance = knn.findNearest(test, k=3)
#test the accuracu of model
maches = result == test_labels
correct = np.count_nonzero(matches)
accuracy = correct * (100/ result.size)
print("accuracy is: = %.2f" % accuracy + "%")



