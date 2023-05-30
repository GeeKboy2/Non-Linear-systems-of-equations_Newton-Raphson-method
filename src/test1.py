
from part1 import *


def test_Newton_Raphson(N=100,epsilon=1e-7):
    print("Test non-linar solution from different start points...", end=" ")
    # Test case 1
    U0 = np.array([1, 1])
    N = 100
    epsilon = 1e-6
    expected_output = np.array([0, 0])
    actual_output = newton_raphson(f, J_f, U0, N, epsilon)
    assert np.allclose(actual_output, expected_output, rtol=1e-2, atol=1e-2)

    # Test case 2
    U0 = np.array([2, -1])
    N = 100
    epsilon = 1e-6
    expected_output = np.array([0, 0])
    actual_output = newton_raphson(f, J_f, U0, N, epsilon)
    assert np.allclose(actual_output, expected_output, rtol=1e-2, atol=1e-2)

    # Test case 3
    U0 = np.array([0.5, 0.5])
    N = 100
    epsilon = 1e-6
    expected_output = np.array([0, 0])
    actual_output = newton_raphson(f, J_f, U0, N, epsilon)
    assert np.allclose(actual_output, expected_output, rtol=1e-2, atol=1e-2)

    print("\tOK")


def test_Newton_Raphson_BT(N=100,epsilon=1e-7):
    print("Test non-linar solution from different start points with backtracking...", end=" ")
    # Test case 1
    U0 = np.array([1, 1])
    N = 100
    epsilon = 1e-6
    expected_output = np.array([0, 0])
    actual_output = newton_raphson_backtracking(f, J_f, U0, N, epsilon)
    assert np.allclose(actual_output, expected_output, rtol=1e-2, atol=1e-2)

    # Test case 2
    U0 = np.array([2, -1])
    N = 100
    epsilon = 1e-6
    expected_output = np.array([0, 0])
    actual_output = newton_raphson_backtracking(f, J_f, U0, N, epsilon)
    assert np.allclose(actual_output, expected_output, rtol=1e-2, atol=1e-2)

    # Test case 3
    U0 = np.array([0.5, 0.5])
    N = 100
    epsilon = 1e-6
    expected_output = np.array([0, 0])
    actual_output = newton_raphson_backtracking(f, J_f, U0, N, epsilon)
    assert np.allclose(actual_output, expected_output, rtol=1e-2, atol=1e-2)

    print("\tOK")

if __name__ == '__main__':
    test_Newton_Raphson()
    test_Newton_Raphson_BT()


