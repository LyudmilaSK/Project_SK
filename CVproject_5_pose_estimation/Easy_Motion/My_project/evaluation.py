import numpy as np

def cos_similarity(pose1, pose2):
    # косинусное сходство между всеми ключевыми точками размерностью = n*n
    cossim = pose1.dot(np.transpose(pose2)) / (
        np.linalg.norm(pose1, axis=1) * np.linalg.norm(pose2, axis=1)
    )
    # косинусное сходство между соответствующими ключевыми точками
    cossim_pair = np.diagonal(cossim)

    # усредненное косинусное сходство по фигуре
    cossim_person = cossim_pair.mean()

    return cossim_person

def get_cos_similarity(enumerator, pose1, pose2):
    cos_sim = []
    cos_sim_print = []

    for foto in range(enumerator):
        cos_sim_pose = cos_similarity(pose1[foto],pose2[foto])
        cos_sim_pose_print = str(f"{cos_sim_pose:.0%}")

        cos_sim.append(cos_sim_pose)
        cos_sim_print.append(cos_sim_pose_print)

    return cos_sim, cos_sim_print


def compute_oks(sigmas, model_bb, input_keypoints, model_keypoints):
    # OKS = exp(-d^2/(2*s^2*k^2))/number_keypoints

    k = 2 * sigmas

    # площадь bounding box
    s = abs(model_bb[2] - model_bb[0]) * abs(model_bb[3] - model_bb[1])

    # расстояние по каждой координате (катеты)
    distance = np.subtract(model_keypoints, input_keypoints)

    # евклидово расстояние в квадрате
    d_square = np.sum(distance ** 2, axis=1)

    degree = d_square / (2 * s ** 2 * k ** 2)

    e = np.exp(-degree)

    return np.mean(e)


def get_oks(enumerator, pose1, pose2, sigmas, bbox):
    oks = []
    oks_print = []
    for foto in range(enumerator):
      oks_pose = compute_oks(sigmas[foto], bbox[foto],
                             pose1[foto], pose2[foto])
      oks_pose_print = str(f"{oks_pose:.0%}")

      oks.append(oks_pose)
      oks_print.append(oks_pose_print)

    return oks, oks_print

def weight_distance(pose1, pose2, conf1):
    # D(U,V) = (1 / sum(conf1)) * sum(conf1 * ||pose1 - pose2||) = sum1 * sum2

    sum1 = 1 / np.sum(conf1)
    sum2 = 0

    for i in range(len(pose1)):
        # каждый индекс i имеет x и y, у которых одинаковая оценка достоверности
        sum2 += np.sum(conf1[i] * abs(pose1[i] - pose2[i]))

    weighted_dist = sum1 * sum2

    return weighted_dist

def get_weight_distance(enumerator, pose1, pose2, conf1):
    distance = []
    distance_print = []

    for foto in range(enumerator):
      distance_pose = weight_distance(pose1[foto],
                                      pose2[foto],
                                      conf1[foto])
      distance_pose_print = '%d'% (distance_pose)
      
      distance.append(distance_pose)
      distance_print.append(distance_pose_print)

    return distance, distance_print