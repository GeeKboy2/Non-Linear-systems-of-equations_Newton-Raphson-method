import matplotlib.pyplot as plt
import numpy as np
from part1 import *


def centriguge_force(k, x, y, x_0, y_0):
    return np.array([k * (x - x_0), k * (y - y_0)])


def gravitational_force(k, x, y, x_0, y_0):
    denominator = ((x - x_0) ** 2 + (y - y_0) ** 2) ** (3/2)
    return -k * np.array([(x - x_0) / denominator, (y - y_0) / denominator])


def centrifuge_jacobian(k):
    return k * np.eye(2)


def gravitational_jacobian(k, x, y, x_0, y_0):
    denominator = ((x - x_0) ** 2 + (y - y_0) ** 2) ** (5/2)
    return k * np.array([[2 * (x - x_0) ** 2 - (y - y_0) ** 2, 3 * (x - x_0) * (y - y_0)], [3 * (x - x_0) * (y - y_0), 2 * (y - y_0) ** 2 - (x - x_0) ** 2]] / denominator)


if __name__ == '__main__':

    # creating the specified functions

    start_vector_1 = np.array([0., 0.])

    start_vector_2 = np.array([1., 0.])

    start_vector_3 = np.array([0.01 / 1.01, 0.])

    def F_1(U):
        return gravitational_force(1., U[0], U[1], start_vector_1[0], start_vector_1[1])

    def J_1(U):
        return gravitational_jacobian(1.,  U[0], U[1], start_vector_1[0], start_vector_1[1])

    def F_2(U):
        return gravitational_force(0.01,  U[0], U[1], start_vector_2[0], start_vector_2[1])

    def J_2(U):
        return gravitational_jacobian(0.01, U[0], U[1], start_vector_2[0], start_vector_2[1])

    def F_3(U):
        return centriguge_force(1.,  U[0], U[1], start_vector_3[0], start_vector_3[1])

    def J_3(U):
        return centrifuge_jacobian(1.)

    f = lambda x:F_1(x)+ F_2(x)

    start_vector = np.array([0.9, 0.])
    equilibrium_2_forces = newton_raphson_backtracking(f, lambda x:J_1(x)+ J_2(x), start_vector)

    print("Pour 2 forces :")
    print("U =",equilibrium_2_forces)
    print("f(U) =",f(equilibrium_2_forces))

    print()
    start_vector = np.array([0., 0.])
    equilibrium_centrifuge = newton_raphson_backtracking(F_3, J_3, start_vector)
    print("Pour une force centrifuge :")
    print("U =",equilibrium_centrifuge) # on cherche 0,0
    print("f(U) =",f(equilibrium_centrifuge))



