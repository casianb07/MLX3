import tensorflow.keras
import numpy as np
import cv2

# incarcam modelul extras
model = tensorflow.keras.models.load_model('classes.h5')

# creez matricea de dimensiuni potrivite pentru a alimenta modelul keras
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# calea catre camera mea web
cam = cv2.VideoCapture(0)

text = ""

while True:

    _, img = cam.read()
    img = cv2.resize(img, (224, 224))

    # convertim imaginea intr-un numpy array prin np.asarray
    image_array = np.asarray(img)

    # 'normalizam' imaginea = procesarea imaginii pentru a-i schimba nivelul intensivatii al pixelilor
    # scopul este de a a aduce un contrast mai bun acolo unde avem stralucire proasta(glare)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # incarcam imaginea in array
    data[0] = normalized_image_array

    # rulam deducerea
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
            text = "painting"
        if i[5] > 0.7:
            text = "laptop"
        if i[6] > 0.7:
            text = "tv remote"
        if i[7] > 0.7:
            text = "shoe"
        if i[8] > 0.7:
            text = "car keys"
        if i[9] > 0.7:
            text = "cup"
        if i[10] > 0.7:
            text = "water tap"
        if i[11] > 0.7:
            text = "plug"
        if i[12] > 0.7:
            text = "apple"
        if i[13] > 0.7:
            text = "pen"
        if i[14] > 0.7:
            text = "person"
        if i[15] > 0.7:
            text = "phone"
        img = cv2.resize(img, (500, 500)) #redimensionam imaginea la 500 X 500
        cv2.putText(img, text, (160, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2) #afisarea prezicerii

    if cv2.waitKey(1) == ord('q'): #apasare tasta 'q' pentru a iesi
        break
    cv2.imshow('img', img)

#cata layere ascunse am
#daca am folosit o retea neuronala convolitionala CNN
#ce ecuatii de activare folosesc pentru modele
#cate date de antrenare am, si cate de test
#numirea proprietati ale unei retele neuronale
#0,7 acuratete sau precizie?
#masuratori de performanta ale retelei, SK learn biblioteca F1 score, recall, precision

#studiu, procentul de recunoastere, cat la suta pentru doua obiecte in cadru
#senzor proximity