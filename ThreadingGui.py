#from multiprocessing import Process ,Queue, Lock
import mediapipe as mp
import pose_media as pm
import numpy as np
import cv2
import tensorflow as tf
import time
from threading import Thread
from queue import Queue

def Main(queue,p):
    cTime = 0
    pTime = 0
    count = 0
    control = 0
    cap =  cv2.VideoCapture("http://169.254.65.103/cgi-bin/stream")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if control % 3 == 0:
            queue.put(frame)
            print(f'main {count}')
            count += 1
        cTime =  time.time()
        fps = 1/(cTime-pTime)
        pTime =  cTime

        control += 1
        while not p.empty():

            sentence = p.get()
        
            cv2.putText(frame, str(sentence),(50,40),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2, cv2.LINE_AA)
            cv2.imshow('',frame)
            cv2.waitKey(1)
    cap.release()


def ExtractionKeypoints(queue, send):
    pose = pm.mediapipe_pose()
    model = pose.mp_holistic.Holistic()
    count = 0
    while True:
        while not queue.empty():
            frame = queue.get()

            frame, results = pose.mediapipe_detection(frame, model)
            keypoints = pose.extract_keypoints(results)
            send.put(keypoints)
            print(f'Processes {count}')


            count += 1


def Predict(queue,p):
    new_model =  tf.keras.models.load_model('TrainedModel\ModeloBacano6.h5') 
    threshold = 0.5
    actions = np.array(['Falling down','Headache','Nausea','Sit down','Stand up','still sitting','Walking'])
    sentence, sequence = [], []
    count = 0
    while True:
        keypoints = queue.get()
        sequence.append(keypoints)
        sequence =  sequence[-30:]
        if len(sequence) == 30:
            res =  new_model.predict(np.expand_dims(sequence, axis = 0))[0]
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])
            if len(sentence) > 1:
                sentence = sentence[-1:]
        print(sentence)
        print(f'Predict {count}')
        count += 1

        p.put(sentence)
        
        
        



if __name__ == "__main__":
    

    q1 = Queue()
    q2 = Queue()
    q3 = Queue()
    

    p1 = Thread(target=Main, args= (q1,q3))
    p2 = Thread(target=ExtractionKeypoints, args= (q1, q2))
    p3 = Thread(target=Predict, args= (q2,q3))

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()


