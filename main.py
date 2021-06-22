Table= ['-', 0, 1, 2,
        0, '-', '-', '-',
        1, '-', '-', '-',
        2, '-', '-', '-', ]

def square(Table):
    for i in range (0,16,4):
        Table_str = ''.join(map(str, Table))
        print(Table_str[i:i + 4])

def check(player):
    answer = input(f"Игрок {player}, введите координаты (строка, столбец), например 02. ")

    if (not answer[:].isdigit()) or (len(answer) != 2):
        print("Ввод координат некорректный")
        return False
    else:
        coordinare_srting = int(answer[0])
        coordinare_column = int(answer[1])
        if coordinare_srting not in range(3) or coordinare_column not in range(3):
            print("Ввод координат некорректный")
            return False
        else:
            index_string = Table.index(int(answer[0]), 4, 13)
            index_column = Table.index(int(answer[1]), 0, 4)

            if Table[index_string + index_column] != '-':
                print("Эта клетка уже занята!")
                return False
    Table[Table.index(int(answer[0]), 4, 13) + Table.index(int(answer[1]), 0, 4)] = player
    return True

def win(Table,sign):
   win_schem = ((5,6,7), (9,10,11), (13,14,15), (5,9,13), (6,10,14), (7,11,15), (5,10,15), (7,10,13))
   for each in win_schem:
       if Table[each[0]] == Table[each[1]] == Table[each[2]] == sign:
          return sign
   return False

counter = 0
last_gamer = '0' # Ход начинает "Х"
winner = False
while not winner:
        square(Table)
        if last_gamer == '0':
           while not check('X'):
               pass
           last_gamer = 'X'
        else:
           while not check('O'):
               pass
           last_gamer = '0'
        counter += 1
        if counter > 4:
           result = win(Table,last_gamer)
           if result:
               square(Table)
               print(f"Конец игры. Выиграл {last_gamer} .")
               winner = True
               break
        if Table.count('-') == 1:
            square(Table)
            print("Конец игры. Ничья.")
            break