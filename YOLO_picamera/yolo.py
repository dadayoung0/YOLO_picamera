import numpy as np
import time
import cv2


# 동영상 해상도
FRAME_W = 800
FRAME_H = 600

# 잘라낼 수치
CUT_X1 = 130
CUT_X2 = 650
CUT_Y1 = 100
CUT_Y2 = 500

# YOLO 파일 위치
LABELS_FILE = './mydata2/obj.names'
CONFIG_FILE = './mydata2/yolo-obj2.cfg'
WEIGHTS_FILE = './weight2/yolo-obj2_4000.weights'

# 정확도 최소값
CONFIDENCE_THRESHOLD = 0.3


class Yolo:
    # 초기화
    def __init__(self):
        # 카메라
        self.cap = cv2.VideoCapture(-1)
        self.cap.set(3, FRAME_W)
        self.cap.set(4, FRAME_H)
        self.prevTime = 0
        self.curTime = 0

        # yolo-tiny 읽어오기
        self.net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

        # label 이름, 색상 배열 생성
        self.labels = open(LABELS_FILE).read().strip().split("\n")
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")

        # layer name 읽어오기
        ln = self.net.getLayerNames()
        self.ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    # fps 계산
    def fps(self):
        # 현재 시간 저장
        self.curTime = time.time()

        # fps 계산
        sec = self.curTime - self.prevTime
        fps = 1 / sec

        # 현재 시간을 이전 시간으로 저장
        self.prevTime = self.curTime

        # 반올림 후 반환
        return round(fps, 1)

    # 사진 찍기
    def capture(self):
        # 카메라에서 프레임 받아오기
        _, img = self.cap.read()

        # 사진 크기 조정 후 반환
        return img[CUT_Y1:CUT_Y2, CUT_X1:CUT_X2]

    # 객체 탐지
    def detect(self, img, h, w):
        # blob 객체 생성
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (320, 320), swapRB=True, crop=False)
        self.net.setInput(blob)

        # 객체 탐지 결과
        start = time.time()
        layer_outputs = self.net.forward(self.ln)
        end = time.time()

        print("[INFO] YOLO took {:.2f} seconds".format(end - start))

        # 객체 정보를 저장할 배열 생성
        boxes = []          # 객체 테두리 좌표
        confidences = []    # 객체 정확도
        class_ids = []      # 객체 class id 번호

        # 결과 출력하기
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # 정확도에 따라 필터링
                if confidence > CONFIDENCE_THRESHOLD:
                    # 테두리 좌표 정보 저장
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    boxes.append([int(centerX - (width / 2)), int(centerY - (height / 2)), int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # 같은 물체에 대한 테두리 제거
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, CONFIDENCE_THRESHOLD)

        if len(indexes) != 0:
            # 객체 정보 표시
            for i in indexes.flatten():
                (x, y, w, h) = boxes[i][0:4]

                color = [int(c) for c in self.colors[class_ids[i]]]

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.labels[class_ids[i]], confidences[i])
                cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return img

    # 프로그램 종료
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
