import mediapipe as mp
import pose_media as pm
import numpy as np
import cv2
import tensorflow as tf
import customtkinter
from PIL import Image, ImageTk
from threading import Thread
from queue import Queue
import time
import imageio

class AppCamera:
    threshold = 0.5
    actions = np.array(['Alerta de Caida',
                        'Normal',
                        'Normal',
                        'Sentandose',
                        'Levantandose',
                        'Sentado',
                        'Caminando'])

    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.q1 = Queue()
        self.q2 = Queue()
        self.q3 = Queue()
        self.q4 = Queue()


        self.pose = pm.mediapipe_pose()
        self.pt = self.pose.mp_holistic.Holistic()
        self.new_model = tf.keras.models.load_model('TrainedModel/ModeloBacano6.h5')


        self.cap = cv2.VideoCapture(0)

        self.canvas = customtkinter.CTkCanvas(window, width=800, height=600)
        self.canvas.pack()

        self.label = customtkinter.CTkLabel(window, fg_color="transparent")
        self.label.pack()

        self.button = customtkinter.CTkButton(window, text="Start", command=self.camera)
        self.button.pack()

        self.button_draw = customtkinter.CTkButton(window, text="Drawing on", command=self.drawing)
        self.button_draw.pack(pady=10)

        self.sequence = []
        self.sentence = []

        self.GIF = []
        self.draw = True
        self.window.mainloop()

    def drawing(self):
        self.draw = not(self.draw)
        if self.button_draw.cget("text") == "Drawing on":
            self.button_draw.configure(text="Drawing off")
        else:
            self.button_draw.configure(text="Drawing on")

    def camera(self):
        def processing():
            cap = cv2.VideoCapture(0)
            while cap.isOpened():
                ret, frameVisual = cap.read()
                frame = cv2.resize(frameVisual, (int(frameVisual.shape[1] / 4), int(frameVisual.shape[0] / 4)))
                if not ret:
                    break
                try:
                    frame, results = self.pose.mediapipe_detection(frame, self.pt)
                except:
                    pass
                if self.draw:
                    self.pose.draw_styled_landmarks(frameVisual, results)
                keypoints = self.pose.extract_keypoints(results)
                self.sequence.append(keypoints)
                if len(self.sequence) > 30:
                    self.sequence = self.sequence[-30:]
                if len(self.sequence) == 30:
                    res = self.new_model.predict(np.expand_dims(self.sequence, axis=0))[0]
                    if res[np.argmax(res)] > self.threshold:
                        action = self.actions[np.argmax(res)]
                        #print(action)
                        
                        if action == 'Alerta de Caida':
                            frameV = cv2.cvtColor(frameVisual, cv2.COLOR_BGR2RGB) 
                            self.GIF.append(frameV)
                        else:
                            if len(self.GIF) > 0:
                                imageio.mimwrite('Caida2.gif', self.GIF, 'GIF')
                                print('GIF guardado')
                                self.GIF = []
                                
                            else:
                                pass
                        if len(self.sentence) == 0 or action != self.sentence[-1]:
                            self.sentence.append(action)
                    if len(self.sentence) > 1:
                        self.sentence = self.sentence[-1:]
                text = ' '.join(self.sentence)
                self.label.configure(text=text)
                frame = cv2.putText(frameVisual, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frameVisual, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=photo, anchor=customtkinter.NW)


        
                    
                    
                

        def sendAlert():
            pass
        

                
        Thread(target=processing, daemon=True).start()
        #Thread(target=GenerateGIF,daemon=True).start()

if __name__ == "__main__":
    customtkinter.set_appearance_mode("dark")  # Modes: system (default), light, dark
    customtkinter.set_default_color_theme("blue")
    root = customtkinter.CTk()
    root.geometry("960x720")
    app = AppCamera(root, "Camera App")
