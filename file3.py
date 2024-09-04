import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('parking1.mp4')

with open("coco.txt", "r") as my_file:
    data = my_file.read()
class_list = data.split("\n")

areas = [
    [(52,364),(30,417),(73,412),(88,369)],
    [(105,353),(86,428),(137,427),(146,358)],
    [(159,354),(150,427),(204,425),(203,353)],
    [(217,352),(219,422),(273,418),(261,347)],
    [(274,345),(286,417),(338,415),(321,345)],
    [(336,343),(357,410),(409,408),(382,340)],
    [(396,338),(426,404),(479,399),(439,334)],
    [(458,333),(494,397),(543,390),(495,330)],
    [(511,327),(557,388),(603,383),(549,324)],
    [(564,323),(615,381),(654,372),(596,315)],
    [(616,316),(666,369),(703,363),(642,312)],
    [(674,311),(730,360),(764,355),(707,308)]
]

while True:    
    ret, frame = cap.read()
    if not ret:
        break
    time.sleep(1)
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    
    counts = [0] * 12

    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(int, row)
        c = class_list[d]
        if 'car' in c:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            for i, area in enumerate(areas):
                if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    counts[i] += 1
                    cv2.putText(frame, c, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    for i, count in enumerate(counts):
        color = (0, 0, 255) if count == 1 else (0, 255, 0)
        cv2.polylines(frame, [np.array(areas[i], np.int32)], True, color, 2)
        cv2.putText(frame, str(i + 1), (50 + 75 * i, 441), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

    total_cars = sum(counts)
    space = 12 - total_cars
    cv2.putText(frame, str(space), (23, 30), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
