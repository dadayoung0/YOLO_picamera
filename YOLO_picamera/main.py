from yolo import Yolo
import cv2


if __name__ == "__main__":
    # YOLO 객체 생성
    y = Yolo()

    # 카메라가 사용되는 동안 반복
    while y.cap.isOpened():
        # 사진 찍기
        img = y.capture()

        # 사물 인식
        img = y.detect(img, img.shape[0], img.shape[1])

        # fps 표시
        cv2.putText(img, str(y.fps()), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

        # 사진 보여주기
        cv2.imshow("Result", img)

        # q 누를 시 break
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    # 종료
    y.close()
