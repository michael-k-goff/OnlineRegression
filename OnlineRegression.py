import numpy as np
import random
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from scipy import linalg
import sympy

# 1-dimensional online regression
def lr(x_avg,y_avg,Sxy,Sx,n,new_x,new_y):
    new_n = n + len(new_x)

    new_x_avg = (x_avg*n + np.sum(new_x))/new_n
    new_y_avg = (y_avg*n + np.sum(new_y))/new_n

    if n > 0:
        x_star = (x_avg*np.sqrt(n) + new_x_avg*np.sqrt(new_n))/(np.sqrt(n)+np.sqrt(new_n))
        y_star = (y_avg*np.sqrt(n) + new_y_avg*np.sqrt(new_n))/(np.sqrt(n)+np.sqrt(new_n))
    elif n == 0:
        x_star = new_x_avg
        y_star = new_y_avg
    else:
        raise ValueError

    new_Sx = Sx + np.sum((new_x-x_star)**2)
    new_Sxy = Sxy + np.sum((new_x-x_star).reshape(-1) * (new_y-y_star).reshape(-1))

    beta = new_Sxy/new_Sx
    alpha = new_y_avg - beta * new_x_avg
    return new_Sxy, new_Sx, new_n, alpha, beta, new_x_avg, new_y_avg

x_avg, y_avg, Sxy, Sx, n = 0,0,0,0,0
random.seed(1234)
X = np.array([random.random() for i in range(10)])
y = np.array([random.random() + 5*X[i] for i in range(10)])

X_total = X
y_total = y

Sxy, Sx, n, alpha, beta, x_avg, y_avg = lr(x_avg,y_avg,Sxy,Sx,n, X,y)

for i in range(100):
    X = np.array([random.random() for i in range(10)])
    X_total = np.append(X_total, X)
    y = np.array([random.random() + 5*X[i] for i in range(10)])
    y_total = np.append(y_total, y)
    Sxy, Sx, n, alpha, beta, x_avg, y_avg = lr(x_avg,y_avg,Sxy,Sx,n, X,y)
    
print(Sxy, Sx, n, alpha, beta, x_avg, y_avg)

regr = linear_model.LinearRegression()
regr.fit(X_total.reshape(-1,1), y_total)
print([regr.intercept_,regr.coef_])

# Multidimensional version

def lr_multi(XX,Xy,X,y, calc_results=False):
    XX = np.add(XX, np.matmul(X.transpose(),X))
    Xy = np.add(Xy, np.matmul(X.transpose(),y))
    if (calc_results):
        lin_ind_cols = sympy.Matrix(XX).T.rref()[1]
        XX_reduced = [
                [XX[i][j] for j in range(len(XX[0])) if j in lin_ind_cols]
            for i in range(len(XX)) if i in lin_ind_cols]
        Xy_reduced = [[Xy[i][0]] for i in range(len(XX)) if i in lin_ind_cols]
        result = np.matmul( np.linalg.inv(XX_reduced), Xy_reduced )
        full_result = np.zeros((len(XX)))
        result_pointer = 0
        for i in range(len(XX)):
            if i in lin_ind_cols:
                full_result[i] = result[result_pointer]
                result_pointer += 1
        return XX, Xy, {"intercept":full_result[0], "coefficients":full_result[1:]}
    else:
        return XX, Xy, None
        
def example():
    y_base = np.array([10,11,12,7,7])
    X_base = np.array([[1,2],[0,1],[3,5],[2,1],[3,3]])
    y = y_base.reshape(1,len(y_base)).transpose()
    X = np.concatenate(([[1]]*len(X_base),X_base), axis=1)
    XX = np.zeros( ( len(X[0]) , len(X[0]) ) )
    Xy = np.zeros( ( len(X[0]) , 1 ) )

    # Split into 2
    X1, X2, y1, y2 = X[:3], X[3:], y[:3], y[3:]
    XX, Xy, _ = lr_multi(XX,Xy,X1,y1)
    XX, Xy, results = lr_multi(XX,Xy,X2,y2, True)
    print(results)

    # Validate with scikit learn. Should match the intercept and coefficients found above.
    regr = linear_model.LinearRegression()
    regr.fit(X_base, y_base)
    print({"intercept":regr.intercept_, "coefficients":regr.coef_})
example()

def singular_example():
    y_base = np.array([10,11,12,7,7])
    # Note that the second column is twice the first, making it redundant for regression purposes.
    X_base = np.array([[1,2],[2,4],[1,2],[3,6],[1,2]]) 
    y = y_base.reshape(1,len(y_base)).transpose()
    X = np.concatenate(([[1]]*len(X_base),X_base), axis=1)
    XX = np.zeros( ( len(X[0]) , len(X[0]) ) )
    Xy = np.zeros( ( len(X[0]) , 1 ) )

    X1,X2,y1,y2 = X[:2],X[2:],y[:2],y[2:]

    XX, Xy, _ = lr_multi(XX,Xy,X1,y1)
    XX, Xy, results = lr_multi(XX,Xy,X2,y2,True)
    print(results)
    # These results are not the same as the results given by scikit-learn's regression.
    # In the event of the X^TX matrix being singular (linear dependency among features), there is not an unambiguous
    # regression that minimizes error.
    regr = linear_model.LinearRegression()
    regr.fit(X_base, y_base)
    print({"intercept":regr.intercept_, "coefficients":regr.coef_})
singular_example()

def large_example():
    # A larger example
    y_base = np.random.rand(1000)
    X_base = np.random.rand(1000,10)
    
    y = y_base.reshape(1,len(y_base)).transpose()
    X = np.concatenate(([[1]]*len(X_base),X_base), axis=1)
    XX = np.zeros( ( len(X[0]) , len(X[0]) ) )
    Xy = np.zeros( ( len(X[0]) , 1 ) )
    
    breakpoints = [0,300,500,800,1000]
    for i in range(len(breakpoints)-2):
        XX, Xy, _ = lr_multi(XX,Xy,X[breakpoints[i]:breakpoints[i+1]], y[breakpoints[i]:breakpoints[i+1]])
    _,_,results = lr_multi(XX,Xy,X[breakpoints[-2]:], y[breakpoints[-2]:],True)
    print(results)
    
    # Validate with scikit-learn's linear regression.
    regr = linear_model.LinearRegression()
    regr.fit(X_base, y_base)
    print({"intercept":regr.intercept_, "coefficients":regr.coef_})
results = large_example()

def degenerate_example():
    y_base = np.array([1,2,3,4,5])
    X_base = np.array([[1,1],[1,1],[1,1],[1,1],[1,1]])
    
    y = y_base.reshape(1,len(y_base)).transpose()
    X = np.concatenate(([[1]]*len(X_base),X_base), axis=1)
    XX = np.zeros( ( len(X[0]) , len(X[0]) ) )
    Xy = np.zeros( ( len(X[0]) , 1 ) )
    _,_,results = lr_multi(XX,Xy,X,y,True)
    print(results)
    
    # Validate with scikit-learn's linear regression.
    regr = linear_model.LinearRegression()
    regr.fit(X_base, y_base)
    print({"intercept":regr.intercept_, "coefficients":regr.coef_})
    
degenerate_example()
