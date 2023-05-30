from part1 import *
import math


def E(X):
    """
        Compute the total electrostatic energy of a system X
    """
    N=len(X)
    sum1 = 0
    for i in range(N):
        second_sum = 0
        for j in range(N):
            if i != j:
                second_sum += math.log(abs(X[i]-X[j]))
        sum1 += np.log(np.abs(X[i]+1)) + np.log(np.abs(X[i]-1)) + (1/2)*second_sum
    return sum1


def derived_E(X):
    """
        Compute the partials derivative of an electrostatic energy of a system X
    """
    N=len(X)
    d_E=np.zeros(N)
    for i in range (N):
        for j in range(N):
            if (i==j):
                second_sum=0
            else : 
                second_sum=second_sum+(1/(X[i]-X[j]))
        sum1=(1/(X[i]+1))+(1/(X[i]-1))+second_sum
        d_E[i]=sum1
    return d_E



def Jacobian_matrix_d_E(X):
    N=len(X)
    Jacobian_d_E=np.zeros((N,N))
    sum1=0
    for i in range (N):
        for j in range(N):
            if (i==j):
                for k in range(N):
                    if (i==k):
                        second_sum=0
                    else : 
                        second_sum=second_sum+(1/((X[i]-X[k])**2))
                sum1=(-1/((X[i]+1)**2))-(1/((X[i]-1)**2))-second_sum
            else : 
                sum1=-(1/((X[i]-X[j])**2))
            Jacobian_d_E[i][j]=sum1
    return Jacobian_d_E


def compute_jacob_diag_coeff(x, i, N):
    """
        Compute the diagonal coefficients of the Jacobian of ∇E(x1,x2,⋯ ,xN)
    """
    res = 0

    for k in range(N):
        if k != i:
            res -= 1 / (x[i] - x[k])**2

    res -= (1 / (x[i] + 1)**2) + (1 / (x[i] - 1)**2)

    return res

def compute_jacob_extra_coeff(x, i, j):
    """
        Compute the other coefficients (other than the diagonals) of the Jacobian of ∇E(x1,x2,⋯ ,xN)
    """
    return 1 / (x[i] - x[j])**2

def J_(x):
    """
        Compute the Jacobian of ∇E(x1,x2,⋯ ,xN)  
    """
    N = len(x)
    jacob = np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            if i == j:
                jacob[i, j] = compute_jacob_diag_coeff(x, i, N)
            else:
                jacob[i, j] = compute_jacob_extra_coeff(x, i, j)

    return jacob



X=[-0.8,0.2]
print("E(X): \n",E(X),"\n")
print("dE(X):\n",derived_E(X),"\n")
print("Jacobienne de E : \n",J_(X),"\n")

#print(newton_raphson_backtracking(derived_E, Jacobian_matrix_d_E, X, len(X), epsilon=1e-7))

if __name__ == "__main__":
    x0 = np.array([-0.8,-0.5,0.1,0.5])

    # Solve for x using Newton-Raphson with backtracking
    x = newton_raphson_backtracking(derived_E, J_, x0, 100,1e-4)
    #print(x)
    # Add the fixed charges to the array of points
    points = np.concatenate(([-1, 1], x))
    
    # Set up the first plot for equilibrium points and legendre polynomials
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    
    ax1.scatter(points, np.zeros(len(points)), color='black', label='Points d\'équilibre')
    ax1.set_xlim([-1.1, 1.1])
    ax1.set_ylim([-1.2, 1.2])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    
    # Plot legendre polynomials
    x = np.linspace(-1, 1, 1000)
    for order in range(7):
        ax1.plot(x, np.polynomial.legendre.Legendre.basis(order)(x), label=order)
    ax1.set_title('Tracés des courbes des polynômes de Legendre et racines de la Jacobienne de ∇E')
    ax1.legend()
    ax1.grid()
    #plt.savefig('part2_legendre.png')
    # Set up the second plot for the energy function
    fig2, ax3 = plt.subplots(figsize=(10, 8))
    
    xvals = np.linspace(-1, 1, 500)
    Evals = np.array([E(np.array([xval])) for xval in xvals])
    ax3.plot(xvals, Evals)
    ax3.set_xlim([-1.1, 1.1])
    ax3.set_ylim([-5, 1])
    ax3.set_title('Energie électrostatique totale en fonction de x')
    ax3.set_xlabel('x')
    ax3.set_ylabel('E(x)')
    ax3.grid()
    #plt.savefig('part2_max_min.png')
    plt.show()
    
    
        