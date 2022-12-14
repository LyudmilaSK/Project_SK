import numpy as np
import cv2
import json  

def get_kpoints(file):
  k_points_count = []
  
  # проход по всем найденным людям
  for person in file['people']:
    # подсчет обнаруженных опорных точек у каждой персоны
    # деление на 3 - по каждой точке данны 2 кординаты и 1 вероятность
    non_zero = len([item for item in person['pose_keypoints_2d'] if item != 0])/3
    k_points_count.append(non_zero)
  
  # номер человека с максимальным количеством детектированных точек
  id_person = k_points_count.index(max(k_points_count))
  # ключевые точки выбранной персоны
  k_points =file['people'][id_person]['pose_keypoints_2d']
  # перевод данных в numpy размером 25*3
  # 25-количество точек. идентифицируемых моделью
  k_points_person = np.array(k_points).reshape((25, 3))
  
  return k_points_person

# чтение полученных файлов и сохранение всех опорных точек  
def reformat_kpoints(path_st, path_ch):
  # путь с хранением файлов .json
  path_folder = '/content/openpose/output/'
  
  cap_st = cv2.VideoCapture(path_st)
  frameRate_st = cap_st.get(5)  # частота кадров
  prop_fps_st = cap_st.get(7) # количество фреймов
  
  cap_ch = cv2.VideoCapture(path_ch)
  frameRate_ch = cap_ch.get(5)  # частота кадров
  prop_fps_ch = cap_ch.get(7) # количество фреймов
  
  dict_video = {'id': [0,1],
              'role': ['student', 'coach'],
              'frame_frequency': [int(frameRate_st),int(frameRate_ch)],
              'num_frame': [int(prop_fps_st),int(prop_fps_ch)]}
  
  kpoints_st = []
  kpoints_ch = []
  prob_st = []
  prob_ch = []
  for index_role in dict_video.get('id'):
    # проход по всем файлам с соответствующей частотой
      for index_file in range (0,
                               dict_video.get('num_frame')[index_role],
                               dict_video.get('frame_frequency')[index_role]):
        path_file = path_folder + dict_video.get('role')[index_role] + '_' +  ('%012d'% (index_file)) + '_keypoints.json'
        
        # запись всех опорных точек в соответствующие переменные
        with open(path_file) as f:
          file = json.load(f)
        if index_role == 0:
          kpoints_st.append(get_kpoints(file)[:,:2])
          prob_st.append(get_kpoints(file)[:,:1])
        else:
          kpoints_ch.append(get_kpoints(file)[:,:2])
          prob_ch.append(get_kpoints(file)[:,:-1])
  
  return kpoints_st, kpoints_ch, prob_st, prob_ch
  
# Редактирование данных для расчета метрик
def select_kpoints_prob(path_st, path_ch):
  
  kpoints_st, kpoints_ch, prob_st, prob_ch = reformat_kpoints(path_st, path_ch)
  num_frame = len(kpoints_st)
  
  coach_key_points = []
  coach_prob = []
  student_key_points = []
  student_prob = []
  for frame in range(num_frame):
    # поиск нулевых векторов на фрейме студента
    id_zero_st = np.where(kpoints_st[frame].sum(axis=1) == 0)
    # поиск нулевых векторов на фрейме коуча
    id_zero_ch = np.where(kpoints_ch[frame].sum(axis=1) == 0)
    # объединение индексов нулевых векторов
    ids_pass = np.concatenate((id_zero_st[0],id_zero_ch[0])).astype(int)
    id_pass = np.unique(ids_pass)
    # удаление опорных точек, нулевых либо у тренера, либо у студента
    student_key_points.append(np.delete(kpoints_st[frame], id_pass, axis = 0))
    student_prob.append(np.delete(prob_st[frame], id_pass, axis = 0))
    coach_key_points.append(np.delete(kpoints_ch[frame], id_pass, axis = 0))
    coach_prob.append(np.delete(prob_ch[frame], id_pass, axis = 0))
  
  return student_key_points, coach_key_points, student_prob, coach_prob