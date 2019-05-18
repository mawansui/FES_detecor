# -*- coding: utf-8 -*-

import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

def detector(gray, frame):
    """
    Принимает grayscale картинку и цветную (frame)
    Возвращает цветную картинку с нанесёнными на неё рамками вокруг лица, глаз,
    улыбки.
    """
    
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Определили лицо, теперь надо вокруг него квадрат нарисовать
    
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_frame = frame[y:y+h, x:x+w]
        
        eyes = eyes_cascade.detectMultiScale(roi_gray, 1.1, 3)
        
        for (e_x, e_y, e_w, e_h) in eyes:
            cv2.rectangle(roi_frame, (e_x, e_y), (e_x + e_w, e_y + e_h), 
                          (0, 255, 0), 2)
        
        smile = smile_cascade.detectMultiScale(roi_gray, 5, 20)
        
        for (s_x, s_y, s_w, s_h) in smile:
            cv2.rectangle(roi_frame, (s_x, s_y), (s_x + s_w, s_y + s_h),
                          (0, 0, 255), 2)
    
    return frame

video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detector(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()