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


if __name__ == '__main__':
    task2()