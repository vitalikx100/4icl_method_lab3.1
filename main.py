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

    print("Y:\n",Y)

    for i in range(3,-1,-1):
        X[i] = (Y[i] - np.dot(U[i, i :], X[i:])) / U[i, i]

    print("X:\n", X)

    Y_solve = np.linalg.solve(L, B)
    print("LY=B\n", Y_solve)

    X_solve = np.linalg.solve(U, Y_solve)
    print("UX = Y:\n", X_solve)

    solve = np.linalg.solve(A, B)
    print("Библиотечное перемножение A и B\n", solve)

if __name__ == '__main__':
    task2()