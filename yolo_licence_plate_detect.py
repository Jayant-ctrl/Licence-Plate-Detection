from ultralytics import YOLO
import cv2
import math
import easyocr
# Load a YOLOv8n PyTorch model
model = YOLO("licence.pt")

# Export the model to NCNN format
# model.export(format="ncnn")  # creates 'yolov8n_ncnn_model'
frame = cv2.imread('PIC1.jpeg')
# Load the exported NCNN model
# ncnn_model = YOLO("licence_ncnn_model")

# Run inference
results = model("PIC1.jpeg")

for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1,y1,x2,y2)
            # cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255), 3)
            w, h = x2 -x1, y2 - y1
            
            imgCropped = frame[y1:y1+h, x1:x1+w]
            gray = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2GRAY)

            bfilter = cv2.bilateralFilter(gray, 11, 17, 17)   # Noise Reduction
            edged = cv2.Canny(bfilter, 30, 200) #Edge Detection
            
            reader = easyocr.Reader(['en'])
            result = reader.readtext(imgCropped)
            # print(result[0][1])

            cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (255,0,255), 2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            cv2.rectangle(frame, (x1, y1), (x1+w, y1+h-70), (255,0,255), -1)
            cv2.putText(frame, f'{result[0][1]}', (x1,y1), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255,255,255), thickness=2)
cv2.imshow('frame', frame)
cv2.waitKey(0)
