import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8s.pt')

def handle_mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_position = [x, y]
        print(mouse_position)

cv2.namedWindow('Video')
cv2.setMouseCallback('Video', handle_mouse_event)

video_capture = cv2.VideoCapture('parking1.mp4')

with open("coco.txt", "r") as file:
    class_names = file.read().splitlines()

polygon_points = [(511, 327), (557, 388), (603, 383), (549, 324)]

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    frame_resized = cv2.resize(frame, (1020, 500))

    predictions = model.predict(frame_resized)
    
    boxes = predictions[0].boxes.data
    box_df = pd.DataFrame(boxes, columns=['x1', 'y1', 'x2', 'y2', 'conf', 'class']).astype("float")
    
    detected_cars = []
    
    for _, box in box_df.iterrows():
        x1, y1, x2, y2, class_id = map(int, [box['x1'], box['y1'], box['x2'], box['y2'], box['class']])
        class_name = class_names[class_id]
        
        if 'car' in class_name:
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            distance_to_polygon = cv2.pointPolygonTest(np.array(polygon_points, np.int32), (center_x, center_y), False)
            if distance_to_polygon >= 0:
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame_resized, (center_x, center_y), 3, (0, 0, 255), -1)
                detected_cars.append(class_name)
    
    if len(detected_cars) == 1:
        cv2.polylines(frame_resized, [np.array(polygon_points, np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame_resized, '9', (591, 398), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    else:
        cv2.polylines(frame_resized, [np.array(polygon_points, np.int32)], True, (0, 255, 0), 2)
        cv2.putText(frame_resized, '9', (591, 398), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Video", frame_resized)

    if cv2.waitKey(1) & 0xFF == 27:
        break

video_capture.release()
cv2.destroyAllWindows()
