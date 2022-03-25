# Output Keluaran LED
from gpiozero import LED
red = LED(14)
green = LED(15)

# impor paket atau library yang diperlukan
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from utils import CFEVideoConf, image_resize
from imutils.video import VideoStream
import numpy as np
import pygame
import imutils
import time
import cv2
import os

# WhatsApp
from twilio.rest import Client

# Akun SID dan Token dari twilio.com/console 
#account_sid = 'ACd51bf8d72a9d0ab79a41b4d23a46f65b'
  
#auth_token = '35862c0e4839d1e6be685d4673cad3ec'

#File Untuk Logo
img_path = '/home/pi/STTP.png'
logo = cv2.imread(img_path, -1)
watermark = image_resize(logo, height=300)
watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2BGRA)

# Program inti yaitu untuk mendeteksi masker        
def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:

        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

# memuat file "face detector" dari tempat folder /home/pi/....
prototxtPath = r"face_detector/deploy.prototxt"
weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# memuat file "mask_detector.model" dari tempat folder /home/pi/....
maskNet = load_model("mask_detector.model")

# untuk memulai video streaming
print("MULAI MEREKAM VIDEO...")
vs = VideoStream(src=0).start()

# pengulangan untuk video streaming agar tidak berhenti "while"
while True:
    # mengatur ukuran video di layar
    frame = vs.read()
    frame = imutils.resize(frame, width=1024)
    
    # Pengatur Logo
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    frame_h, frame_w, frame_c = frame.shape

    # Pengatur Ukuran Logo dan Penempatan Logo
    overlay = np.zeros((frame_h, frame_w, 4), dtype='uint8')
    watermark_h, watermark_w, watermark_c = watermark.shape
    for i in range(0, watermark_h):
        for j in range(0, watermark_w):
            if watermark[i,j][3] != 0:
                offset = 10
                overlay[ i, j] = watermark[i,j]

    cv2.addWeighted(overlay, 0.75, frame, 1.0, 0, frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    # mendeteksi wajah dalam bingkai dan menentukan apakah mereka mengenakan a
    # masker wajah atau tidak
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # pengulangan lokasi wajah yang terdeteksi dan yang sesuai lokasi
    for (box, pred) in zip(locs, preds):
        # pengaturak label persegi pada wajah dan prediksi
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # tentukan nama label persegi dan warna yang akan kita gunakan
        # menampilkan tulisan "masker" dan "tidak pakai masker" di wajah
        label = "Masker" if mask > withoutMask else "Tidak Pakai Masker"
        color = (0, 255, 0) if label == "Masker" else (0, 0, 255)
        
        #File Untuk Pemutar Audio
        if label == "Masker":
            #client = Client(account_sid, auth_token)
  
            #message = client.messages.create( body='Terima kasih telah memakai masker',from_='whatsapp:+14155238886', to='whatsapp:+6282338729964')
  
            #print(message.sid)
            red.off()
            green.on()
            pygame.mixer.init()
            pygame.mixer.music.load("YA.wav")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() == True:
                    continue
            green.off()
            red.off()
        else:
            #client = Client(account_sid, auth_token)
  
            #message = client.messages.create( body='Ada yang terdeteksi tidak pakai masker',from_='whatsapp:+14155238886', to='whatsapp:+6282338729964')
  
            #print(message.sid)
            green.off()
            red.on()
            pygame.mixer.init()
            pygame.mixer.music.load("TIDAK.wav")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() == True:
                    continue
            green.off()
            red.off()
        # sertakan probabilitas ukuran di label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        
        # pengaturan ukuran label persegi di wajah
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    
    # Menampilkan output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(20) & 0xFF

    # jika tombol `q` ditekan, maka video streaming akan berhenti
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
