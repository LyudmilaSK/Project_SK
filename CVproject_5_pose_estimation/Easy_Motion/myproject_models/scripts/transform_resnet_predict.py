import numpy as np
import torch

kpoint_threshold = 2
conf_threshold = 0.9

# получение опорных точек с классификацией по степени уверенности детекции
def get_keypoints(output_model, keypoint_threshold=kpoint_threshold, conf_threshold=conf_threshold):
    all_keypoints = output_model['keypoints']
    all_scores = output_model['keypoints_scores']
    confs = output_model['scores']
    best_keypoints_img = []
    bed_keypoints_img = []
    # для каждого задетектированного человека
    for person_id in range(len(all_keypoints)):
        # проверяем степень уверенности детектора
        if (confs[person_id] > conf_threshold):
            # собираем ключевые точки конкретного человека
            keypoints = all_keypoints[person_id, ...]
            # собираем скоры для ключевых точек
            scores = all_scores[person_id, ...]
            # итерируем по каждому скору
            keypoints_best_pose = []
            bed_keypoints_pose = []
            for kp in range(len(scores)):
                # запоминаем индексы опорных точек с низкой степенью уверенности
                if scores[kp] <= keypoint_threshold:
                    bed_keypoints_pose.append(kp)
                # конвертируем массив ключевых точек в список целых чисел
                keypoint = list(
                    map(int, keypoints[kp, :2].detach().numpy())
                )
                keypoints_best_pose.append(keypoint)
            # список ключевых точек с высокой/низкой уверенностью детекции
            best_keypoints_img.append(np.array(keypoints_best_pose))
            bed_keypoints_img.append(np.array(bed_keypoints_pose))

    # форматирование списков для работы
    bed_keypoints_img = np.array(bed_keypoints_img)
    if bed_keypoints_img.shape[0] == 1:
        bed_keypoints_img = bed_keypoints_img.squeeze(0)

    best_keypoints_img = np.array(best_keypoints_img)
    if best_keypoints_img.shape[0] == 1:
        best_keypoints_img = best_keypoints_img.squeeze(0)

    return best_keypoints_img, bed_keypoints_img

# отбор ключевых точек для сопоставления коуча со студентом
def select_kpoints_bbox(num_foto, prediction_model, prediction_person, sigmas,
                   conf_threshold = conf_threshold,):
    # Соберём два набора ключевых точек
    model_keypoints = []
    person_keypoints = []
    # Индексы сохраненных и неучтённых фото
    id_pose = []
    id_cut = []
    # список отредактированных сигма для каждой фотографии с учётом непринятия
    # в учёт незначимых опорных точек
    sigmas_cut = []
    # bounding box у значимых поз
    bounding_box = []
    for foto in range(num_foto):
        model_poses, id_pass_model = get_keypoints(prediction_model[foto])
        person_poses, id_pass_person = get_keypoints(prediction_person[foto])

        # отбор фото, где количество достоверно опознанных тренеров равно
        # количеству достоверно опознанных учеников
        try:
            # индексы опорных точек, детерминируемых с низким score у обоих поз
            ids_pass = np.concatenate(
                (id_pass_model, id_pass_person)).astype(int)
            id_pass = np.unique(ids_pass)

            # удаление в обоих наборах опорных точек,
            # отсутвующих либо у тренера, либо у ученика
            model_poses_cut = np.delete(model_poses, id_pass, axis=0)
            person_poses_cut = np.delete(person_poses, id_pass, axis=0)

            # удаление констант, связанных с удаляемыми опорными точками
            sigmas_cut.append(np.delete(sigmas, id_pass, axis=0))

            # номера фото, участвующих в оценке
            id_pose.append(foto)

            # два набора ключевых точек для оценки
            model_keypoints.append(model_poses_cut)
            person_keypoints.append(person_poses_cut)
        except:
            # номера фото, не участвующих в оценке
            # количество фото, где найдено больше одной позы или не найдено ни одной позы
            id_cut.append(foto)

    # проход по выбранным изображениям
    for foto in id_pose:
        # выбор bounding box у значимой позы
        idx_bool = prediction_model[foto]['scores'] > conf_threshold
        idx_int = idx_bool.nonzero()
        idx = int(idx_int)
        bb = torch.Tensor.numpy(prediction_model[foto]['boxes'][idx])
        bounding_box.append(bb)

    return model_keypoints, person_keypoints, bounding_box, sigmas_cut, id_pose, id_cut