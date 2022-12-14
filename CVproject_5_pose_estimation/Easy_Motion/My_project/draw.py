import cv2
import itertools
import numpy as np

kpoint_threshold = 2
conf_threshold = 0.9

def draw_keypoints_per_person(img, all_keypoints, all_scores, confs, thickness,
                              keypoint_threshold=kpoint_threshold,
                              conf_threshold=conf_threshold):
    # создаём спектр цветов
    cmap = list(itertools.permutations((200, 100, 0), 3))
    # создаём копию изображений
    img_copy = img.copy()
    # для каждого задетектированного человека
    for person_id in range(len(all_keypoints)):
        # проверяем степень уверенности детектора
        if confs[person_id] > conf_threshold:
            # собираем ключевые точки конкретного человека
            keypoints = all_keypoints[person_id, ...]
            # собираем скоры для ключевых точек
            scores = all_scores[person_id, ...]
            # итерируем по каждому скору
            for kp in range(len(scores)):
                # проверяем степень уверенности детектора опорной точки
                if scores[kp] > keypoint_threshold:
                    # конвертируем массив ключевых точек в список целых чисел
                    keypoint = tuple(
                        map(int, keypoints[kp, :2].detach().numpy().tolist())
                    )
                    # рисуем круг радиуса thickness вокруг точки
                    cv2.circle(img_copy, keypoint, thickness, cmap[person_id], -1)

    return img_copy

def get_limbs_from_keypoints(keypoints):
    limbs = [
        [keypoints.index("right_eye"), keypoints.index("nose")],
        [keypoints.index("right_eye"), keypoints.index("right_ear")],
        [keypoints.index("left_eye"), keypoints.index("nose")],
        [keypoints.index("left_eye"), keypoints.index("left_ear")],
        [keypoints.index("right_shoulder"), keypoints.index("right_elbow")],
        [keypoints.index("right_elbow"), keypoints.index("right_wrist")],
        [keypoints.index("left_shoulder"), keypoints.index("left_elbow")],
        [keypoints.index("left_elbow"), keypoints.index("left_wrist")],
        [keypoints.index("right_hip"), keypoints.index("right_knee")],
        [keypoints.index("right_knee"), keypoints.index("right_ankle")],
        [keypoints.index("left_hip"), keypoints.index("left_knee")],
        [keypoints.index("left_knee"), keypoints.index("left_ankle")],
        [keypoints.index("right_shoulder"), keypoints.index("left_shoulder")],
        [keypoints.index("right_hip"), keypoints.index("left_hip")],
        [keypoints.index("right_shoulder"), keypoints.index("right_hip")],
        [keypoints.index("left_shoulder"), keypoints.index("left_hip")],
    ]
    return limbs


def draw_skeleton_per_person(img, all_keypoints, all_scores, confs, thickness,
                             keypoint_threshold=kpoint_threshold,
                             conf_threshold=conf_threshold):

    keypoints = ['nose', 'left_eye', 'right_eye',
                 'left_ear', 'right_ear', 'left_shoulder',
                 'right_shoulder', 'left_elbow', 'right_elbow',
                 'left_wrist', 'right_wrist', 'left_hip',
                 'right_hip', 'left_knee', 'right_knee',
                 'left_ankle', 'right_ankle']

    limbs = get_limbs_from_keypoints(keypoints)

    # создаём спектр цветов
    cmap = list(itertools.permutations((200, 100, 0), 3))
    # создаём копию изображений
    img_copy = img.copy()
    # если keypoints детектированы
    if len(all_keypoints)>0:
        # для каждого задетектированного человека
        for person_id in range(len(all_keypoints)):
            # проверяем степень уверенности детектора
            if confs[person_id]>conf_threshold:
            # собираем ключевые точки конкретного человека
                keypoints = all_keypoints[person_id, ...]

                # для каждой конечности
                for limb_id in range(len(limbs)):
                    # отмечаем начало конечности
                    limb_loc1 = keypoints[limbs[limb_id][0], :2].detach().numpy().astype(np.int32)
                    # отмечаем окончание конечности
                    limb_loc2 = keypoints[limbs[limb_id][1], :2].detach().numpy().astype(np.int32)
                    # определяем скор по конечности как минимальный скор среди ключевых точек конечности
                    limb_score = min(all_scores[person_id, limbs[limb_id][0]], all_scores[person_id, limbs[limb_id][1]])
                    # проверяем степень уверенности детектора опорной точки
                    if limb_score> keypoint_threshold:
                        # рисуем линии вдоль конечности
                        cv2.line(img_copy, tuple(limb_loc1), tuple(limb_loc2), cmap[person_id], thickness)

    return img_copy

