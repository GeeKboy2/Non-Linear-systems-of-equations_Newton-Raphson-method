from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np


def newton_raphson(f, J, U0, N=100, epsilon=1e-7):
    """
    Solves a system of nonlinear equations using the Newton-Raphson method.
    """
    U = U0
    for _ in range(N):
        U_old = U
        if np.linalg.norm(f(U)) < epsilon:
            break
        U = U - np.linalg.lstsq(J(U), f(U), rcond=None)[0]

    if (np.linalg.norm(U-U_old) > epsilon):
        print("convergence failed")
        return U
    return U


def newton_raphson_backtracking(f, J, U0, N=100, epsilon=1e-7):
    """
    Solves a system of nonlinear equations using the Newton-Raphson method alongside back tracking.
    """
    alpha = 1

    U = U0
    h = np.linalg.lstsq(J(U), f(U), rcond=None)[0]

    for _ in range(N):
        U_old = U
        U = U - alpha * h
        h = np.linalg.lstsq(J(U), f(U), rcond=None)[0]
        alpha = alpha / 1.1
        if alpha < epsilon or np.linalg.norm(f(U - alpha * h)) > np.linalg.norm(f(U)):
            break

    if (np.linalg.norm(U-U_old) > epsilon):
        print("convergence failed")
        return U
    return U


def plot_convergence(f, J, U0, N=1000, epsilon=1e-7):
    """
    Plots the graph of the bowl function and the two types of convergence
    red : without back tracking
    green : with back tracking
    """
    alpha = 1

    U = U0
    x_bt, y_bt, z_bt = [U[0]], [U[1]], [f(U)[0]]
    h = np.linalg.lstsq(J(U), f(U), rcond=None)[0]

    for _ in range(N):
        U = U - alpha * h
        x_bt.append(U[0])
        y_bt.append(U[1])
        z_bt.append(f(U)[0])
        h = np.linalg.lstsq(J(U), f(U), rcond=None)[0]
        alpha = alpha / 1.1
        if alpha < epsilon or np.linalg.norm(f(U - alpha * h)) > np.linalg.norm(f(U)):
            break

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(x_bt, y_bt, z_bt, marker='o', color='green',label="avec backtracking")


    U = U0
    U[0]=-U[0]
    x, y, z = [U[0]], [U[1]], [f(U)[0]]
    for _ in range(N):
        if np.linalg.norm(f(U)) < epsilon:
            break
        U = U - np.linalg.lstsq(J(U), f(U), rcond=None)[0]
        x.append(U[0])
        y.append(U[1])
        z.append(f(U)[0])

        
    ax.plot(x, y, z, marker='o', color='red',label="sans backtracking")

    x_vals = np.linspace(-0.00309185,0.00309185, 100)
    y_vals = x_vals
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f([X, Y])[0]

    ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                    alpha=0.3)

    ax.legend()

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.tight_layout()
    plt.show()


def g(x):
    return np.array([x[0] ** 3 + 3*x[0] - 2])


def J_g(x):
    return np.array([[3 * x[0]**2 + 3]])


def f(x):
    return np.array([x[0]**2+x[1]**2, x[0]*x[1]])


def J_f(x):
    return np.array([[2*x[0], 2*x[1]], [x[1], x[0]]])


if __name__ == "__main__":
    print(g(newton_raphson(g, J_g, np.array([-1, -1]), 100, 1e-7)))
    print(g(newton_raphson_backtracking(g, J_g, np.array([1, 1]), 100, 1e-7)))

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d

    U0_g = np.array([-1])
    U0_f = np.array([0.00309185, 0.00309185]) # use this to show the fast convergence of the methode with Back tracking
    #U0_f = np.array([0.5,0.5])


    print("solution without back_tracking:")
    print("Solution to g(x) = 0:", newton_raphson(g, J_g, U0_g),"expecting [0.596]")
    print("Solution to f(x) = 0:", newton_raphson(
        f, J_f, U0_f), "expecting [0,0]")
    print("solution with back_tracking:")
    print("Solution to g(x) = 0:", newton_raphson_backtracking(g, J_g, U0_g),"expecting [0.596]")
    print("Solution to f(x) = 0:", newton_raphson_backtracking(
        f, J_f, U0_f), "expecting [0,0]")

    plot_convergence(f, J_f, U0_f)
