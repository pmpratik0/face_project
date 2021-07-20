import cv2;
import matplotlib.pyplot as plt
import numpy as np

'''
img  = cv2.imread('news.JPG');
print(img.shape)
print(img)


while (True):
    cv2.imshow('window',img)
    # 27 is ASCII of esc
    if cv2.waitKey(2)==27:
        break
cv2.destroyAllWindows()


haar_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#haar_data.detectMultiScale(img)

# to draw rectangle on faces
#cv2.rectangle(img,(x,y),(w,h),(b,g,r),border_thickness)

while (True):
    faces = haar_data.detectMultiScale(img)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
    cv2.imshow('window',img)
    # 27 is ASCII of esc
    if cv2.waitKey(2)==27:
        break
cv2.destroyAllWindows()



haar_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# TO start Camera
# to collect data withoutmsk
capture = cv2.VideoCapture(0)
data=[]
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img);
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face,(50,50))
            print(len(data))
            if len(data) < 400:
                data.append(face)

        cv2.imshow('window',img)
        if(cv2.waitKey(2) == 27):
            break
capture.release()
cv2.destroyAllWindows()

np.save('with_mask.npy',data)

'''
#plt.imshow(data[0])



with_mask = np.load('with_mask.npy')
without_mask = np.load('without_mask.npy')

print(with_mask.shape)
print(without_mask.shape)

with_mask = with_mask.reshape(400,50*50*3)
without_mask = without_mask.reshape(400,50*50*3)

print(with_mask.shape)
print(without_mask.shape)


X = np.r_[with_mask, without_mask]
print(X.shape)

labels = np.zeros(X.shape[0])
labels[400:] = 1.0

names = {0 : 'Mask', 1: 'No Mask'}

#svm - support vector machine
#SVC - support vector classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.20)

print(x_train.shape)

from sklearn.decomposition import PCA

pca = PCA(n_components = 3)
x_train = pca.fit_transform(x_train)

print(x_train[0])
print(x_train.shape)


svm = SVC()
svm.fit(x_train, y_train)

x_test = pca.fit_transform(x_test)
y_pred = svm.predict(x_test)

print(accuracy_score(y_test, y_pred))


# TO start Camera
# to collect data withoutmsk
haar_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)
data=[]
font = cv2.FONT_HERSHEY_DUPLEX
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img);
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face,(50,50))
            face = face.reshape(1,-1)
            face = pca.transform(face)
            pred = svm.predict(face)[0]
            n = names[int(pred)]
            cv2.putText(img, n, (x,y), font, 1,(244,250,250), 2)
            print(n)
            #svm.predict(face)
        cv2.imshow('window',img)
        if(cv2.waitKey(2) == 27):
            break
capture.release()
cv2.destroyAllWindows()

