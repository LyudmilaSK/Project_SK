import os
from os.path import exists, join, basename, splitext
from pathlib import Path
import cv2
import math 
import numpy as np
from scipy import stats
import sys

sys.path.append('/content/drive/MyDrive/SF/Project_5/Easy_Motion/')
sys.path.append('/content/drive/MyDrive/SF/Project_5/Easy_Motion/My_project')
sys.path.append('/content/drive/MyDrive/SF/Project_5/Easy_Motion/myproject_data')
sys.path.append('/content/drive/MyDrive/SF/Project_5/Easy_Motion/myproject_models/scripts')

from My_project.show import show_local_mp4_video
from transform_openpose_predict import select_kpoints_prob
from My_project.evaluation import get_cos_similarity, get_weight_distance

path_project = '/content/drive/MyDrive/SF/Project_5/Easy_Motion/'

# путь к входным файлам .mp4
video_model_path = os.path.join(path_project,'myproject_data/video_coach/coach.mp4')
video_person_path = os.path.join(path_project,'myproject_data/video_student/student.mp4')

#загрузка модели
git_repo_url = 'https://github.com/CMU-Perceptual-Computing-Lab/openpose.git'
project_name = splitext(basename(git_repo_url))[0]
if not exists(project_name):
  # install new CMake becaue of CUDA10
  os.system('wget -q https://cmake.org/files/v3.13/cmake-3.13.0-Linux-x86_64.tar.gz')
  os.system('tar xfz cmake-3.13.0-Linux-x86_64.tar.gz --strip-components=1 -C /usr/local')
  # clone openpose
  os.system("git clone -q --depth 1 %s" %git_repo_url)
  os.system("sed -i 's/execute_process(COMMAND git checkout master WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}\/3rdparty\/caffe)/execute_process(COMMAND git checkout f019d0dfe86f49d1140961f8c7dec22130c83154 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}\/3rdparty\/caffe)/g' openpose/CMakeLists.txt")
  # install system dependencies
  os.system('apt-get -qq install -y libatlas-base-dev libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libgflags-dev libgoogle-glog-dev liblmdb-dev opencl-headers ocl-icd-opencl-dev libviennacl-dev')
  # install python dependencies
  os.system('pip install -q youtube-dl')
  # build openpose
  os.system('cd openpose && rm -rf build || true && mkdir build && cd build && cmake .. && make -j`nproc`')
  
# путь к выходным файлам .mp4
path_openpose_student = os.path.join(path_project,'myproject_models/videos/openpose_student.mp4')
path_openpose_coach = os.path.join(path_project,'myproject_models/videos/openpose_coach.mp4')

# запуск модели на пользовательских данных
os.system(f"cd openpose && ./build/examples/openpose/openpose.bin \
--video {video_person_path} \
--keypoint_scale 1 \
--net_resolution '656x368' \
--write_json ./output/ \
--display 0 \
--write_video ../openpose.avi")
# convert the result into MP4
os.system(f'ffmpeg -y -loglevel info -i openpose.avi {path_openpose_student}')
  

os.system(f"cd openpose && ./build/examples/openpose/openpose.bin \
--video {video_model_path} \
--keypoint_scale 1 \
--net_resolution '656x368' \
--display 0  \
--write_video ../openpose.avi \
--write_json ./output/ ")
# convert the result into MP4
os.system(f'ffmpeg -y -loglevel info -i openpose.avi {path_openpose_coach}')

# воспроизведение коуча и студента со скелетом
show_local_mp4_video(path_openpose_student)
show_local_mp4_video(path_openpose_coach)

# получение отобранных опорных точек
student_key_points, coach_key_points, student_prob, _ = select_kpoints_prob(path_openpose_student, path_openpose_coach)

num_frame = len(student_key_points)

# оценка соответствия поз коуча и студента
cos_sim, cos_sim_print = get_cos_similarity(num_frame,
                                            student_key_points,
                                            coach_key_points)
distance, distance_print = get_weight_distance(num_frame,
                                               student_key_points,
                                               coach_key_points,
                                               student_prob)

# Нанесение метрик на видео
step_save = 6
folder_person = os.path.join(path_project,'myproject_models/images_openpose/person_skeleton')
if not os.path.exists(folder_person):
  os.makedirs(folder_person)

cap = cv2.VideoCapture(path_openpose_student)
frameRate = cap.get(5)  # частота кадров

# Определение кодека и создание видеозаписи
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)

fourcc = cv2.VideoWriter_fourcc(*'MPEG')
folder_video = os.path.join(path_project,'myproject_models/videos/output_opеnpose.mp4')
out = cv2.VideoWriter(folder_video,fourcc, 2.0, size)

id_metric = 0
while (cap.isOpened()):
    frameId = cap.get(1) # номер текущего кадра
    ret, frame = cap.read() 
    if (ret != True):
      break
    elif (frameId % math.floor(frameRate) == 0):
      font = cv2.FONT_HERSHEY_SIMPLEX

      cv2.putText(frame,
                  ''.join(['Limb direction ', cos_sim_print[id_metric]]),
                  (50, 50), 
                  font, 0.5, 
                  (255, 0, 0), 
                  1, 
                  cv2.LINE_4
                  )
      
      cv2.putText(frame,
                  ''.join([' Pose distance ', distance_print[id_metric]]),
                  (50, 75), 
                  font, 0.5, 
                  (255, 0, 0), 
                  1, 
                  cv2.LINE_4
                  )
      # сохранение одного фрейма из секундного интервала в формате изображения
      if (frameId % frameRate) % step_save == 0:
        filename ="frame%d.jpg" % (frameId/frameRate)
        directory = os.path.join(folder_person, filename)
        cv2.imwrite(directory, frame)
      # выход из фрейма
      out.write(frame)
      id_metric+=1
        
      # ключ для выхода из записи
      if cv2.waitKey(1) & 0xFF == ord('a'):
          break
  
cap.release()
out.release()
cv2.destroyAllWindows()

# Сохранение паралельных кадров
cap = cv2.VideoCapture(path_openpose_coach)
frameRate = cap.get(5)  # частота кадров

# создание папки для вывода 
folder_tandem = os.path.join(path_project,'myproject_models/images_openpose/tandem')
if not os.path.exists(folder_tandem):
  os.makedirs(folder_tandem)

while (cap.isOpened()):
    frameId = cap.get(1) # номер текущего кадра
    ret, frame = cap.read() 
    if (ret != True):
      break
      
    # обработка фрейма из каждого шестого секундного интервала
    elif (frameId % (frameRate*step_save)) == 0:

      # изменение размера изображения коуча
      resized_coach = cv2.resize(frame, size,
                                 interpolation = cv2.INTER_AREA)
      
      # открытие изображения со студентом на той же секунде
      num_second = int(frameId/frameRate)
      path = os.path.join(folder_person,'frame'+str(num_second)+'.jpg')
      img_person = cv2.imread(path) 
      
      # сохранение файла c коучем и студентом
      filename ="coach_person_%d.jpg" % num_second
      directory = os.path.join(folder_tandem, filename)
      tandem_img = np.concatenate((resized_coach, img_person),
                                  axis=1)
      cv2.imwrite(directory, tandem_img)
      
      # выход из фрейма
      out.write(frame)

      # ключ для выхода из записи
      if cv2.waitKey(1) & 0xFF == ord('a'):
          break
  
cap.release()
out.release()
cv2.destroyAllWindows()

# количество ключевых точек на всех фотографиях
num_kpoints_person = 25*num_frame
# количество учтённых ключевых точек
num_kpoints = 0
for frame in range(num_frame):
  num_kpoints += student_key_points[frame].shape[0]
print(f'Average limb direction = {(stats.mode(cos_sim)[0].item()):.2%}')
print(f"Average pose distance = {'%d'%(stats.mode(distance)[0].item())}")
print('0% images were unvalued')
print(f'{(1- num_kpoints/num_kpoints_person):.0%} key points were deleted')
