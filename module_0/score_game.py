import numpy as np


def game_core(number):
    """На каждой попытке угадать искомое число функция считает среднюю величину
       между существующими границами.
       На первой итерации границами являются 1 и 100.
       Каждый раз границы меняются в зависимости от того,
       больше или меньше искомое число, и зависят от предыдущего предсказания.
       Функция возвращает число попыток.
    """
    count = 1
    right_border = 101  # 101 даёт возможность угадать число 100
    left_border = 1
    predict = (right_border+left_border) // 2
    while number != predict:
        count += 1
        if number > predict:
            left_border = predict
        else:
            right_border = predict
        predict = (right_border+left_border) // 2
    print(f"Случайное число {number} угадано с {count} попытки.")
    return(count)


def score_game(game_core):
    """Функция запускает игру 1000 раз и считает среднюю величину попыток,
       которые были закончены определением искомого числа.
    """
    count_ls = []
    np.random.seed(1)  # фиксируем RANDOM SEED, чтобы эксперимент был воспроизводим
    random_array = np.random.randint(1,101, size=(1000))
    for number in random_array:
        count_ls.append(game_core(number))
    score = int(np.mean(count_ls))
    print(f"Алгоритм угадывает число в среднем за {score} попыток.")
    return(score)


score_game(game_core)