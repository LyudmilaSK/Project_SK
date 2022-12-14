import numpy as np

def affine_transformation(person_pose, model_pose):
    # С помощью расширенной матрицы можно осуществить умножение вектора x
    # на матрицу A и добавление вектора b за счёт единственного матричного умножения.
    # Расширенная матрица создаётся путём дополнения векторов "1" в конце.
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:, :-1]

    # Расширим наборы ключевых точек до [[ x y 1] , [x y 1]]
    Y = pad(model_pose)
    X = pad(person_pose)

    # Решим задачу наименьших квадратов X * A = Y
    # и найдём матрицу аффинного преобразования A.
    A, res, rank, s = np.linalg.lstsq(X, Y)
    A[np.abs(A) < 1e-10] = 0  # превратим в "0" слишком маленькие значения

    # Преобразование входного набора ключевых точек с помощью матрицы А
    A_transform = lambda x: unpad(np.dot(pad(x), A))
    input_transform = A_transform(person_pose)
    return input_transform