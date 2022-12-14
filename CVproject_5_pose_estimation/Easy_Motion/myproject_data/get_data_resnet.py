import os
import cv2
import math

# Загрузка данных и кадрирование
def get_frames(video_file, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    count = 0
    cap = cv2.VideoCapture(video_file)  # загрузка видео
    frameRate = cap.get(5)  # частота кадров
    while cap.isOpened():
        frameId = cap.get(1)  # номер текущего кадра
        ret, frame = cap.read()
        if not ret:
            break
        elif frameId % math.floor(frameRate) == 0:
            filename = "frame%d.jpg" % count
            count += 1
            directory = os.path.join(folder_name, filename)
            cv2.imwrite(directory, frame)

    cap.release()


