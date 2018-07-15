import numpy as np
import cv2
import Tkinter as tk
from PIL import Image, ImageTk
import trainer
import json

rs = json.load(open("results.xml"))
trn = trainer.Trainer()
trn.results = rs

window = tk.Tk()
window.wm_title("Smile Recognition")
window.config(background="#FFFFFF")

imageFrame = tk.Frame(window, width=600, height=500)
imageFrame.grid(row=0, column=0, padx=10, pady=2)
text = tk.Label(window, text="isSmiling Text")

text.grid(row=1, column=0)

lmain = tk.Label(imageFrame)
lmain.grid(row=0, column=0)
cap = cv2.VideoCapture(0)


def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    filename = "/home/brkkrgl/PycharmProjects/Smile-Recognition/lbpcascade_frontalface.xml"
    classifier = cv2.CascadeClassifier(filename)
    faces = classifier.detectMultiScale(cv2image)
    for x, y, w, h in faces:
        cv2.rectangle(cv2image, (x, y), (x+w, y+h), (255,0,0), 2)

    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


sliderFrame = tk.Frame(window, width=600, height=100)
sliderFrame.grid(row=600, column=0, padx=10, pady=2)


show_frame()
window.mainloop()
