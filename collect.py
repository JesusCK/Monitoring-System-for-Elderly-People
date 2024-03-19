import cv2
import os

DATASET_PATH =  'DATASET'
CLASES =  ['Sit down', 'Stand up']
DATAPOINT = 'DATAIMG2'

for clase in CLASES:
    videoPath = os.path.join(DATASET_PATH,clase)
    for videoFile in os.listdir(videoPath):
        print(videoFile)
        n_frame = 0
        cap =cv2.VideoCapture(videoFile)
        totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(totalFrames)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow('Frame', frame)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                   break
            elif key == ord('s'):
                clase_path = os.path.join(DATAPOINT, clase)
                frame_path = os.path.join(DATAPOINT, clase, f"frame_{n_frame}.jpg" )
                if not os.path.exists(clase_path):
                     os.makedirs(clase_path)
                cv2.imwrite(frame_path,frame)
                print(f"Frame guardado: {frame_path}")
                n_frame += 1
