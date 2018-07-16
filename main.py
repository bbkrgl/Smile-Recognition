import cv2
import Tkinter as tk
from PIL import Image, ImageTk
import classifier as svm
from scipy.ndimage import zoom

window = tk.Tk()
window.wm_title("Smile Recognition")
window.config(background="#FFFFFF")

classifier = svm

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
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    filename = "lbpcascade_frontalface.xml"
    clf = cv2.CascadeClassifier(filename)
    faces = clf.detectMultiScale(cv2image)

    for face in faces:
        (x, y, w, h) = face
        if w > 100:
            cv2.rectangle(cv2image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            horizontal_offset = 0.15 * w
            vertical_offset = 0.2 * h
            extracted_face = cv2image[int(y + vertical_offset):int(y + h),
                             int(x + horizontal_offset):int(x - horizontal_offset + w)]
            new_extracted_face = zoom(extracted_face, (64. / extracted_face.shape[0],
                                                       64. / extracted_face.shape[1]))

            res = svm.predict([new_extracted_face.ravel()])

            if res == 1:
                text.config(text="True")
            else:
                text.config(text="False")

    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


sliderFrame = tk.Frame(window, width=600, height=100)
sliderFrame.grid(row=600, column=0, padx=10, pady=2)


show_frame()
window.mainloop()
