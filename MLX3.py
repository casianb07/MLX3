import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
# ---------------------------
import time
import datetime
# ---------------------------
import tensorflow.keras
# ---------------------------
import tkinter as tk
from PIL import ImageTk, Image
from notifypy import Notify


def threat3():
    notification = Notify()
    notification.title = "WARNING"
    notification.message = "Unauthorised person detected!"
    notification.icon = './alert.ico'
    notification.application_name = "Security"
    notification.audio = "Alarm.wav"
    notification.send()


def threat4():
    notification = Notify()
    notification.title = "WARNING"
    notification.message = "Unknown person detected!"
    notification.icon = './alert.ico'
    notification.application_name = "Security"
    notification.audio = "Alarm.wav"
    notification.send()


def clicked1():
    path = 'ImagesDeposit'
    images = []
    classNames = []
    myList = os.listdir(path)
    print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    print(classNames)

    def findEncodings(images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    encodeListKnown = findEncodings(images)
    print('Encoding complete')

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            else:
                threat4()

            if classNames[matchIndex] == 'Cristi Borcea':
                name = classNames[matchIndex].upper()

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                threat3()

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break


# ----------------------------------------------------------------------------------------------------------------------


def clicked2():
    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    body_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_fullbody.xml")

    detection = False
    detection_stopped_time = None
    timer_started = False
    SECONDS_TO_RECORD_AFTER_DETECTION = 5

    frame_size = (int(cap.get(3)), int(cap.get(4)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    while True:
        _, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        bodies = body_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) + len(bodies) > 0:
            if detection:
                timer_started = False
            else:
                detection = True
                current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
                out = cv2.VideoWriter(
                    f"{current_time}.mp4", fourcc, 20, frame_size)
                print("Started Recording!")
        elif detection:
            if timer_started:
                if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                    detection = False
                    timer_started = False
                    out.release()
                    print('Stop Recording!')
            else:
                timer_started = True
                detection_stopped_time = time.time()

        if detection:
            out.write(frame)

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break


# ----------------------------------------------------------------------------------------------------------------------


def clicked3():
    model = tensorflow.keras.models.load_model('classes.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    cam = cv2.VideoCapture(0)
    text = ""

    while True:

        _, img = cam.read()
        img = cv2.resize(img, (224, 224))

        image_array = np.asarray(img)

        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        data[0] = normalized_image_array

        prediction = model.predict(data)
        for i in prediction:
            if i[0] > 0.7:
                text = "cat"
            if i[1] > 0.7:
                text = "mouse"
            if i[2] > 0.7:
                text = "scissors"
            if i[3] > 0.7:
                text = "bottle"
            if i[4] > 0.7:
                text = "tv remote"
            if i[5] > 0.7:
                text = "shoe"
            if i[6] > 0.7:
                text = "car keys"
            if i[7] > 0.7:
                text = "cup"
            if i[8] > 0.7:
                text = "apple"
            if i[9] > 0.7:
                text = "pen"
            if i[10] > 0.7:
                text = "person"
            if i[11] > 0.7:
                text = "phone"
            if i[12] > 0.7:
                text = "banana"
            if i[13] > 0.7:
                text = "marker"
            if i[14] > 0.7:
                text = "knife"
                threat1()
            if i[15] > 0.7:
                text = "firearm"
                threat2()
            img = cv2.resize(img, (500, 500))
            cv2.putText(img, text, (160, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
        cv2.imshow('img', img)


def threat1():
    notification = Notify()
    notification.title = "WARNING"
    notification.message = "harmful object detected!"
    notification.icon = './alert.ico'
    notification.application_name = "Object detector"
    notification.audio = "Alarm.wav"
    notification.send()


def threat2():
    notification = Notify()
    notification.title = "WARNING"
    notification.message = "firearm detected!"
    notification.icon = './alert.ico'
    notification.application_name = "Object detector"
    notification.audio = "Alarm.wav"
    notification.send()


# ----------------------------------------------------------------------------------------------------------------------


root = tk.Tk()
root.geometry('600x400')
root.title('MLX3')
root.iconbitmap(r'icon.ico')

tk.Button(root, text='Face Recognition', font="HERSHEY_SIMPLEX 16", width=16, command=clicked1).pack(side=tk.LEFT)

tk.Button(root, text='Security Camera', font="HERSHEY_SIMPLEX 16", width=16, command=clicked2).pack(side=tk.RIGHT)

tk.Button(root, text='Image Detection', font="HERSHEY_SIMPLEX 16", width=16, command=clicked3).pack(side=tk.RIGHT)

my_img1 = Image.open("face.png")
resized1 = my_img1.resize((177, 177), Image.ANTIALIAS)
new_img1 = ImageTk.PhotoImage(resized1)
my_label1 = tk.Label(image=new_img1)
my_label1.place(x=5, y=10)

my_img2 = Image.open("detect.png")
resized2 = my_img2.resize((165, 165), Image.ANTIALIAS)
new_img2 = ImageTk.PhotoImage(resized2)
my_label2 = tk.Label(image=new_img2)
my_label2.place(x=210, y=10)

my_img3 = Image.open("cam.png")
resized3 = my_img3.resize((177, 177), Image.ANTIALIAS)
new_img3 = ImageTk.PhotoImage(resized3)
my_label3 = tk.Label(image=new_img3)
my_label3.place(x=415, y=10)

my_img4 = Image.open("geometry.png")
resized4 = my_img4.resize((600, 200), Image.ANTIALIAS)
new_img4 = ImageTk.PhotoImage(resized4)
my_label4 = tk.Label(image=new_img4)
my_label4.place(x=1, y=220)

root.mainloop()

# ----------------------------------------------------------------------------------------------------------------------
