import cv2

faceCascsade = cv2.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')

imgOrg = cv2.imread("Resources/trio.jpg")
img = cv2.resize(imgOrg, (700,460))
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascsade.detectMultiScale(imgGray,1.1,4)

for (x,y,w,h) in faces :
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

cv2.imshow("Shirley", img)
cv2.waitKey(0)