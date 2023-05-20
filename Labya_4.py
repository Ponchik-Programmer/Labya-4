# Формируется матрица F следующим образом: скопировать в нее А и если в В количество чисел, меньших К в нечетных
# столбцах больше, чем сумма чисел в четных строках, то поменять местами С и Е симметрично, иначе В и Е поменять
# местами несимметрично. При этом матрица А не меняется. После чего если определитель матрицы А больше суммы
# диагональных элементов матрицы F, то вычисляется выражение: A-1*AT – K * F, иначе вычисляется выражение (A-1 +G-FТ)*K,
# где G-нижняя треугольная матрица, полученная из А. Выводятся по мере формирования А, F и все матричные
# операции последовательно.

import numpy as np
import matplotlib.pyplot as plt

K = int(input('Введите K: '))
N = int(input('Введите N: '))

if N < 2:
    print('Длина сторон матрицы А (N,N) должна быть больше 2!')
    exit()

A = np.random.randint(low=-10, high=11, size=(N, N))

n = N // 2  # размерность матриц B, C, D, E (n x n)

w = N // 2
if N % 2 == 0:
    E = A[0:w, 0:w]
    B = A[0:w, w:]
    C = A[w:, w:]
    D = A[w:, 0:w]
else:
    E = A[0:w, 0:w]
    B = A[0:w, w + 1:]
    C = A[w + 1:, w + 1:]
    D = A[w + 1:, 0:w]

print('Матрица A:')
print(A)

print('Матрица E:')
print(E)

print('Матрица B:')
print(B)

print('Матрица C:')
print(C)

print('Матрица D:')
print(D)

a = 0  # количество чисел, меньших К в нечетных столбцах
b = 0  # сумма чисел в четных строках

for i in range(1, n, 2):  # четные строки
    for j in range(n):
        if B[i][j] < K :
            b += B[i][j]

for j in range(0, n, 2):  # нечетные столбцы
    for i in range(n):
        if B[i][j] < K:
            a += 1

F = A.copy()
if a > b:
    print('')
    print('Меняем местами C и E симметрично')
    if N % 2 == 0:
        F[w:, w:] = np.flipud(E)    # flipud - отражение по вертикали,
        F[0:w, 0:w] = np.flipud(C)  # fliplr - по горизонтали, flip - относительно вертикали и горизонтали
    else:
        F[w + 1:, w + 1:] = np.flipud(E)
        F[0:w, 0:w] = np.flipud(C)
else:
    print('')
    print('Меняем местами B и E несимметрично')
    if N % 2 == 0:
        F[0:w, w:] = E
        F[0:w, 0:w] = B
    else:
        F[0:w, w + 1:] = E
        F[0:w, 0:w] = B
print('')
print('Матрица F')
print(F)

det_A = np.linalg.det(A)  # определитель матрицы A
sum_diag = np.trace(F)  # сумма диагональных элементов матрицы F

if det_A > sum_diag:  # определитель матрицы A больше суммы диагональных элементов матрицы F
    print('определитель матрицы A больше суммы диагональных элементов матрицы F')
    result = ((np.linalg.inv(A)).dot(A.T) - K * F)
else:
    G = np.tril(A, -1)  # нижняя трегугольная матрица из матрицы A
    result = (np.linalg.inv(A) + G - F.T) * K

np.set_printoptions(precision=1, suppress=True)  # выводим с точностью до одного знака после запятой
print('Результат:')
print(result)

# работа с графиками
plt.figure(figsize=(16, 9))

# вывод тепловой карты матрицы F
plt.subplot(2, 2, 1)
plt.xticks(ticks=np.arange(F.shape[1]))
plt.yticks(ticks=np.arange(F.shape[1]))
plt.xlabel('Номер столбца')
plt.ylabel('Номер строки')
hm = plt.imshow(F, cmap='Oranges', interpolation="nearest")
plt.colorbar(hm)
plt.title('Тепловая карта элементов')

# вывод диаграммы распределения сумм элементов по строкам в матрице F
sum_by_rows = np.sum(F, axis=1)  # axis = 1 - сумма по строкам
x = np.arange(F.shape[1])
plt.subplot(2, 2, 2)
plt.plot(x, sum_by_rows, label='Сумма элементов по строкам')
plt.xlabel('Номер строки')
plt.ylabel('Сумма элементов')
plt.title('График суммы элементов по строкам')
plt.legend()

# вывод диаграммы распределения количества положительных элементов в столбцах матрицы F
res = []
for col in F.T:
    count = 0
    for el in col:
        if el > 0:
            count += 1
    res.append(count)

x = np.arange(F.shape[1])
plt.subplot(2, 2, 3)
plt.bar(x, res, label='Количество положительных элементов в столбцах')
plt.xlabel('Номер столбца')
plt.ylabel('Количество положительных элементов')
plt.title('График количества положительных элементов в столбцах')
plt.legend()

# вывод круговой диаграммы
x = np.arange(F.shape[1])
plt.subplot(2, 2, 4)
P = []
for i in range(N):
    P.append(abs(F[0][i]))
plt.pie(P, labels=x, autopct='%1.2f%%')
plt.title("График с использованием функции pie")

plt.tight_layout(pad=3.5, w_pad=3, h_pad=4) # расстояние от границ и между областями
plt.suptitle("Использование библиотеки Matplotlib", y=1)
plt.show()