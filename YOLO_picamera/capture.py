import cv2
import time

cap = cv2.VideoCapture(-1)
cap.set(3, 800)
cap.set(4, 600)
cnt = 0
while cap.isOpened():
    cnt += 1
    _, img = cap.read()
    img = img[100:500, 130:650]
    name = 'img' + str(cnt) + '.jpg'
    cv2.imwrite(name, img)
    cv2.imshow("img", img)
    time.sleep(1)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
