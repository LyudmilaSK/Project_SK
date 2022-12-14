import os
import cv2
import numpy as np
from scipy import stats
import sys

sys.path.append('/content/drive/MyDrive/SF/Project_5/Easy_Motion/')
sys.path.append('/content/drive/MyDrive/SF/Project_5/Easy_Motion/My_project')
sys.path.append('/content/drive/MyDrive/SF/Project_5/Easy_Motion/myproject_data')
sys.path.append('/content/drive/MyDrive/SF/Project_5/Easy_Motion/myproject_models/scripts')

from My_project.affine import affine_transformation
from My_project.draw import draw_keypoints_per_person, draw_skeleton_per_person
from My_project.evaluation import get_cos_similarity, get_oks
from myproject_data.get_data_resnet import get_frames
from resnet_predict import get_predict
from transform_resnet_predict import select_kpoints_bbox

sigmas = np.array([
    .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87,
    .87, .89, .89]) / 10.0

# частота видео для визуализации тандема в секундах
step_imshow = 1
path_project = '/content/drive/MyDrive/SF/Project_5/Easy_Motion/'

# путь к файлам .mp4
video_model_path = os.path.join(path_project,'myproject_data/video_coach/coach.mp4')
video_person_path = os.path.join(path_project,'myproject_data/video_student/student.mp4')

# путь к папкам с результатом кадрироавния видео
frames_model_path = os.path.join(path_project,'myproject_models/images_resnet/model/')
frames_person_path =os.path.join(path_project,'myproject_models/images_resnet/person/')

get_frames(video_model_path, frames_model_path)
get_frames(video_person_path, frames_person_path)

num_foto = sum(os.path.isfile(os.path.join(frames_person_path, f))
               for f in os.listdir(frames_person_path))

# predict resnet
prediction_model, prediction_person = get_predict(num_foto)

# выбор значимых предсказаний
model_keypoints, person_keypoints, b_box, sigmas_cut, id_pose, id_cut = select_kpoints_bbox(num_foto, prediction_model, prediction_person, sigmas)

# аффинное преобразование набора опорных точек студента
person_keypoint_aff = []
for foto in range(len(id_pose)):
  keypoint_aff = affine_transformation(person_keypoints[foto],
                                       model_keypoints[foto])
  person_keypoint_aff.append(keypoint_aff)

# оценка соответствия поз коуча и студента
cos_sim, cos_sim_print = get_cos_similarity(len(id_pose),
                                            person_keypoint_aff,
                                            model_keypoints)
oks, oks_print = get_oks(len(id_pose), person_keypoint_aff, model_keypoints,
                         sigmas_cut, b_box)

# сообщения в фотографии, где детектировано более одного человека
message = "error"
for id in id_cut:
  cos_sim_print.insert(id, message)
  oks_print.insert(id, message)

# Визуализация keypoint и результатов оценки

for foto in range(0, num_foto, step_imshow):
    path_model = os.path.join(frames_model_path, 'frame'+str(foto)+'.jpg')
    path_person = os.path.join(frames_person_path, 'frame'+str(foto)+'.jpg')

    img_arr_model = cv2.imread(path_model)
    img_arr_model = cv2.cvtColor(img_arr_model, cv2.COLOR_BGR2RGB)

    img_arr_person = cv2.imread(path_person)
    img_arr_person = cv2.cvtColor(img_arr_person, cv2.COLOR_BGR2RGB)

    # коуч с опорными точками
    model_with_point = draw_keypoints_per_person(
        img_arr_model,
        all_keypoints=prediction_model[foto]['keypoints'],
        all_scores=prediction_model[foto]['keypoints_scores'],
        confs=prediction_model[foto]['scores'],
        thickness=5
    )

    # коуч с опорными точками и каркасом
    model_with_skeleton = draw_skeleton_per_person(
        model_with_point,
        all_keypoints=prediction_model[foto]['keypoints'],
        all_scores=prediction_model[foto]['keypoints_scores'],
        confs=prediction_model[foto]['scores'],
        thickness=3
    )

    # студент с опорными точками
    person_with_point = draw_keypoints_per_person(
        img_arr_person,
        all_keypoints=prediction_person[foto]['keypoints'],
        all_scores=prediction_person[foto]['keypoints_scores'],
        confs=prediction_person[foto]['scores'],
        thickness=4
    )

    # студент с опорными точками и каркасом
    person_with_skeleton = draw_skeleton_per_person(
        person_with_point,
        all_keypoints=prediction_person[foto]['keypoints'],
        all_scores=prediction_person[foto]['keypoints_scores'],
        confs=prediction_person[foto]['scores'],
        thickness=2
    )

    coach_bgr = cv2.cvtColor(model_with_skeleton, cv2.COLOR_RGB2BGR)
    person_bgr = cv2.cvtColor(person_with_skeleton, cv2.COLOR_RGB2BGR)

    # нанесение метрик на фото студента
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(person_bgr,
                ''.join(['Limb direction ', cos_sim_print[foto]]),
                (50, 50),
                font, 0.5,
                (255, 0, 0),
                1,
                cv2.LINE_4
                )

    cv2.putText(person_bgr,
                ''.join([' Joint position ', oks_print[foto]]),
                (50, 75),
                font, 0.5,
                (255, 0, 0),
                1,
                cv2.LINE_4
                )
    # изменение размера изображения коуча
    new_size = person_bgr.shape[:-1][::-1]
    resized_coach = cv2.resize(coach_bgr, new_size,
                               interpolation=cv2.INTER_AREA)
    # сохранение студента и коуча в одном файле
    folder_name = os.path.join(path_project,'myproject_models/images_resnet/tandem/')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    filename = "coach_person_%d.jpg" % foto
    directory = os.path.join(folder_name, filename)
    tandem_img = np.concatenate((resized_coach, person_bgr),
                                axis=1)
    cv2.imwrite(directory, tandem_img)



# Нанесение метрик на видео
# создание видео из всех изображений студента
cap = cv2.VideoCapture(video_person_path)

# Определение кодека и создание видеозаписи

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter(os.path.join(path_project,'myproject_models/videos/output_resnet.mp4'), fourcc, 2.0, size)
id_foto = 0

while id_foto != num_foto:
    ret, frame = cap.read()
    if ret:
        path = os.path.join(frames_person_path,'frame'+str(id_foto)+'.jpg')
        img_arr = cv2.imread(path)
        prediction = prediction_person[id_foto]

        # студент с опорными точками
        person_with_point = draw_keypoints_per_person(
            img_arr,
            all_keypoints=prediction['keypoints'],
            all_scores=prediction['keypoints_scores'],
            confs=prediction['scores'],
            thickness=4
        )

        # студент с опорными точками и каркасом
        person_with_skeleton = draw_skeleton_per_person(
            person_with_point,
            all_keypoints=prediction['keypoints'],
            all_scores=prediction['keypoints_scores'],
            confs=prediction['scores'],
            thickness=2
        )

        # нанесение метрик на фото
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(person_with_skeleton,
                    ''.join(['Limb direction ', cos_sim_print[id_foto]]),
                    (50, 50),
                    font, 0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_4
                    )

        cv2.putText(person_with_skeleton,
                    ''.join([' Joint position ', oks_print[id_foto]]),
                    (50, 75),
                    font, 0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_4
                    )
        # запись фрема
        out.write(person_with_skeleton)
        id_foto += 1

        # ключ для выхода из записи
        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

cap.release()
out.release()

# количество ключевых точек на всех фотографиях
num_kpoints_person = 17*len(id_pose)
# количество значимых ключевых точек
num_cut_kpoints = 0
for foto in range(len(id_pose)):
  num_cut_kpoints += person_keypoints[foto].shape[0]
# количество незначимых ключевых точек
num_del_kpoints = 1 - num_cut_kpoints/num_kpoints_person

print(f'Average limb direction = {(stats.mode(cos_sim)[0].item()):.2%}')
print(f'Average joint position = {(stats.mode(oks)[0].item()):.2%}')
print(f'{(len(id_cut)/num_foto):.0%} images were unvalued')
print(f'{num_del_kpoints:.0%} key points on valued images were deleted')



