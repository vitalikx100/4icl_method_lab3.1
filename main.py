import math
import numpy as np

def task1():
    A = np.triu(np.random.randint(1, 101, (4,4)))
    print("Матрица A:\n", A)

    B = np.random.randint(1, 101, 4)
    print("Вектор B:\n", B)

    X = np.zeros(4)
    for i in range(3, -1, -1):
        X[i] = (B[i] - np.dot(A[i, i :], X[i :])) / A[i, i]
    print("X:\n", X)

    X_solve = np.linalg.solve(A,B)
    print("Проверка библиотечным способом, X:\n", X_solve)


def task2():
    A = np.array([[15.7, 6.6, -5.7, 11.5],
                  [8.8, -6.7, 5.5, -4.5],
                  [6.3, -5.7, -23.4, 6.6],
                  [14.3, 8.7, -15.7, -5.8]])
    print("Матрица A:\n", A)
    B = np.array([-2.4, 5.6, 7.7, 23.4])
    print("Вектор B:\n", B)

    L = np.identity(4, float)
    U = np.zeros((4, 4), float)
    for i in range(4):
        for j in range(4):
            if i <= j:
                U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])
            if i > j:
                L[i, j] = (A[i, j] - np.dot(L[i, :j], U[:j, j])) / U[j, j]

    print("L:\n", L)
    print("U:\n", U)
    print("LU:\n", np.dot(L,U))

    X = np.zeros(4)
    Y = np.zeros(4)

    for i in range(4):
        Y[i] = (B[i] - np.dot(L[i, :i], Y[ :i])) / L[i, i]

    print("Y:\n", Y)

    for i in range(3,-1,-1):
        X[i] = (Y[i] - np.dot(U[i, i :], X[i:])) / U[i, i]

    print("X:\n", X)

    solve = np.linalg.solve(A, B)
    print("Библиотечное решение A и B\n", solve)

def task3():
    A = np.array([[15.7, 6.6, -5.7, 11.5],
                  [8.8, -6.7, 5.5, -4.5],
                  [6.3, -5.7, -23.4, 6.6],
                  [14.3, 8.7, -15.7, -5.8]])
    print("Матрица A:\n", A)
    B = np.array([-2.4, 5.6, 7.7, 23.4])
    print("Вектор B:\n", B)

    Q = np.zeros_like(A)
    cnt = 0
    for a in A.T:
        u = np.copy(a)
        for i in range(0, cnt):
            u -= np.dot(np.dot(Q[:, i].T, a), Q[:, i])
        e = u / np.linalg.norm(u)
        Q[:, cnt] = e
        cnt += 1

    R = np.triu(np.dot(Q.T, A))
    print("Q:\n", Q)
    print("R:\n", R)

    print("QR:\n", np.dot(Q, R))

    X = np.zeros(4)
    Y = np.zeros(4)

    Y = np.dot(Q.T, B)

    for i in range(3, -1, -1):
        X[i] = (Y[i] - np.dot(R[i, i:], X[i:])) / R[i, i]

    print("Ответ:\n", X)
    print("Библиотечное решение:\n", np.linalg.solve(A, B))

def task4():
    a = np.array([[3.15, -1.72, -1.23], [0.72, 0.67, 1.18], [2.57, -1.34, -0.68]], 'float')
    b = np.array([2.15, 1.43, 1.03], 'float')
    print("A=\n", a, "\nB=\n", b)

    n = a.shape[0]  # сохряняем размерность матрицы А
    # создаем матрицу с и вектор f (x=cx+f), заполненные 0
    c = np.zeros((n, n))
    f = np.zeros((n, 1))

    ## Выполняем преобразования системы, чтобы она удовлетворяла усл для метода зейделя
    # меняем местами 2 и 3ю строки
    a0 = np.array([[3.15, -1.72, -1.23], [0.72, 0.67, 1.18], [2.57, -1.34, -0.68]], 'float')
    a[1,], a[2,] = a[2,], a0[1,]
    b[1], b[2] = b[2], b[1]
    # к 3ей прибавляем вторую
    a[2, :] += a[1, :]
    b[2] += b[1]
    # Я не придумала, какие преобразования еще надо сделать, чтобы условию удовлетворяла и 3-я строка
    # так что я просто в явном виде изменяю в ней значение 3-его элемента
    a[2, 2] += 3

    print("После преобразований:\nA=\n", a, "\nB=\n", b)
    # Заполняем матрицу C и вектор F
    for i in range(n):
        for j in range(n):
            c[i, j] = -a[i, j] / a[i, i]
        f[i] = b[i] / a[i, i]
        c[i, i] = 0
    print('c=\n', c, '\nf=\n', f)

    # Проверка на удовлетворение условия
    for i in range(n):
        for j in range(n):
            if (abs(c[i, j]) >= 1):
                print("!!!\nУсловие не выполнено!\nМетод применять нельзя!!!")
                break

    E = np.identity(3)
    X = np.zeros((3, 1), dtype=float)


    def nextX(x):
        x1 = (1 / a[0, 0]) * (-a[0, 1] * x[1] - a[0, 2] * x[2] + b[0])
        x2 = (1 / a[1, 1]) * (-a[1, 0] * x1 - a[1, 2] * x[2] + b[1])
        x3 = (1 / a[2, 2]) * (-a[2, 0] * x1 - a[2, 1] * x2 + b[2])
        return np.array((x1, x2, x3), dtype=float)

    while ((nextX(X) - X).all() > 0.001):
        X = nextX(X)
    print("Вектор X, посчитанный методом Зейделя: \n", X)
    print("Проверка: \n", np.reshape(np.linalg.solve(a, b), (3, 1)))


def task5():
    A = np.array([[15.7, 6.6, -5.7],
                  [8.8, -6.7, 5.5],
                  [6.3, -5.7, -23.4],
                  [14.3, 8.7, -15.7]])
    print("Матрица A:\n", A)
    B = np.array([-2.4, 5.6, 7.7, 23.4])
    print("Вектор B:\n", B)
    A_T = np.transpose(A)
    A_inv = np.dot(A_T, A)
    A_inv = np.linalg.inv(A_inv)
    X = np.dot(A_inv, A_T)
    X = np.dot(X, B)
    print("Псевдорешение системы:\n", X)

if __name__ == '__main__':
    task4()