import mediapipe as mp
import pose_media as pm
import numpy as np
import cv2
import tensorflow as tf
import time
import customtkinter
from PIL import Image, ImageTk

from threading import Thread
from queue import Queue

class AppCamera:
    cTime = 0
    pTime = 0
    threshold = 0.5
    actions = np.array(['Falling down',
                    'Headache',
                    'Nausea',
                    'Sit down',
                    'Stand up',
                    'still sitting',
                    'Walking'])
    def __init__(self, window, window_title):
        self.window =  window
        self.window.title(window_title)
        

        self.q1 = Queue()
        self.q2 = Queue()
        self.q3 =  Queue()
        self.q4 = Queue()

        self.pose = pm.mediapipe_pose()
        self.pt = self.pose.mp_holistic.Holistic()
        self.new_model = tf.keras.models.load_model('TrainedModel/ModeloBacano6.h5')

        self.cap = cv2.VideoCapture(0) 

        self.canvas = customtkinter.CTkCanvas(window, width=800, height=600)
        self.canvas.pack() 

        self.label = customtkinter.CTkLabel(window, fg_color="transparent")
        self.label.pack()

        self.button =  customtkinter.CTkButton(window, text="Start", command=self.camera)
        self.button.pack()
        




        self.sequence = []
        self.sentence = []

        self.window.mainloop()



    def camera(self):
        def processing():
            green = (0, 255, 0)
            cap = cv2.VideoCapture("http://169.254.184.230/cgi-bin/stream")
            while cap.isOpened():
                ret,frame = cap.read()
                frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
                if not ret:
                    break
                try:
                    frame,results = self.pose.mediapipe_detection(frame,self.pt)
                except:
                    pass
                self.pose.draw_styled_landmarks(frame,results)
                keypoints = self.pose.extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-30:]
                if len(self.sequence) == 30:
                    res = self.new_model.predict(np.expand_dims(self.sequence, axis=0))[0]
                    if res[np.argmax(res)] > self.threshold:
                        if len(self.sentence) > 0:
                            if self.actions[np.argmax(res)] != self.sentence[-1]:
                                self.sentence.append(self.actions[np.argmax(res)])
                        else:
                            self.sentence.append(self.actions[np.argmax(res)])
                    if len(self.sentence) > 1: 
                        self.sentence = self.sentence[-1:]
                text = ' '.join(self.sentence)
                self.label.configure(text=text)

                
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=self.photo, anchor=customtkinter.NW)
                self.cTime = time.time()
                
        Thread(target=processing, daemon=True).start()
            
 

                
                    






                

                
                
            
if __name__ == "__main__":
    customtkinter.set_appearance_mode("dark")  # Modes: system (default), light, dark
    customtkinter.set_default_color_theme("blue")
    root = customtkinter.CTk()
    root.geometry("960x720")
    app = AppCamera(root, "Camera App")


