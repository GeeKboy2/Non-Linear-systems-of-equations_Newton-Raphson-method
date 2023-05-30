import numpy as np

def f(x):
    return np.array([x[0]**2+x[1]**2,x[0]*x[1]])

def J(x):
    return np.array([[2*x[0],2*x[1]],[x[1],x[0]]])

def test_protocole(f):
    print('__FUNC__ f : ')
    X=[1,1]
    f_X=[2,1]
    Y=[2,5]
    f_Y=[29,10]
    Z=[5,-3]
    f_Z=[34,-15]
    assert(f(X)==f_X)
    assert(f(Y)==f_Y)
    assert(f(Z)==f_Z)
    print("OK\n")

    print('__FUNC__ Newton_Raphson :')
    solution_X = Newton_Raphson(f,jacobienne(f),np.array([0]*len(X)),X,200, 10**(-5))
    assert(f(solution_X)==0)
    solution_Y = Newton_Raphson(f,jacobienne(f),np.array([0]*len(Y)),Y,200, 10**(-5))
    assert(f(solution_X)==0)
    solution_Z = Newton_Raphson(f,jacobienne(f),np.array([0]*len(Z)),Z,200, 10**(-5))
    assert(f(solution_X)==0)
    print("OK\n")


