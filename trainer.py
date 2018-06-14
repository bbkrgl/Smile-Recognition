from sklearn import datasets
import Tkinter as tk
from PIL import Image, ImageTk
from sklearn.svm import SVC
import json

face_dataset = datasets.fetch_olivetti_faces()
svc = SVC(kernel="linear")


class Trainer:
    def __init__(self):
        self.results = {}
        self.imgs = face_dataset.images

        self.index = 0

    def increment_face(self):
        if self.index + 1 >= len(self.imgs):
            return self.index
        else:
            while str(self.index) in self.results:
                print self.index
                self.index += 1
            return self.index

    def record_result(self, smile=True):
        self.results[str(self.index)] = smile


trainer = Trainer()

window = tk.Tk()

window.wm_title("Smile Recognition")
window.config(background="#FFFFFF")


def display_face(face):
    img = Image.fromarray(face)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain = tk.Label(window)
    lmain.configure(image=imgtk)
    lmain.image = imgtk
    lmain.grid(row=0, column=0)


def update_smiling(b):
    trainer.record_result(smile=b)
    trainer.increment_face()

smiling_button = tk.Button(window, text="Smiling", command=lambda x=True: update_smiling(x))
smiling_button.grid(row=1, column=0)
nonsmiling_button = tk.Button(window, text="Not Smiling", command=lambda x=False: update_smiling(x))
nonsmiling_button.grid(row=1, column=1)

display_face(trainer.imgs[trainer.index]*256)

window.mainloop()

""""
with open("results.xml", "w") as f:
    json.dump(trainer.results, f)
"""
