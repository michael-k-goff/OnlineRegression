# Online Linear Regression

In this project, we describe and implement online linear regression.

#### Linear Regression

Linear regression is one of the most basic machine learning models. Given `n` inputs with `k` features each, <img src="https://latex.codecogs.com/gif.latex?X = (X_{1,1},X_{1,2},\ldots,X_{1,k}); \ldots; (X_{n,1},X_{n,2},\ldots,X_{n,k}) " /> and responses <img src="https://latex.codecogs.com/gif.latex?y = y_1,y_2,\ldots,y_n " />, we want to choose an intercept <img src="https://latex.codecogs.com/gif.latex?\beta_0" /> and coefficients <img src="https://latex.codecogs.com/gif.latex?\beta_1, \ldots, \beta_n" /> that minimize the sum of squared errors: <img src="https://latex.codecogs.com/gif.latex?\sum_{i=1}^n (y_i - \beta X_i)^2" />. Linear regression is a fairly simple model and often a good starting point among machine learning algorithms.

#### Online Learning

In the form stated above, linear regression requires that all inputs are known and processed simultaneously. This may be infeasible if there is too much input data to be held in memory, or if data is collected continually. In this scenario, we need to feed data in batches to a learning algorithm in a process known as online learning.

#### Implementation

As demonstrated in the workbook, if regression coefficients are generated only once at the end, the time complexity of online linear regression is <img src="https://latex.codecogs.com/gif.latex?O(nk^2+k^3)" />. If a set of coefficients is desired for every input example, the time complexity is <img src="https://latex.codecogs.com/gif.latex?O(nk^3)" />. These figures assume that naive implementations of matrix multiplication and inversion are used. In either case, the space complexity is <img src="https://latex.codecogs.com/gif.latex?O(n'k^2)" />, where `n'` is the size of the largest batch.

The correctness of this implementation is verified with several examples. Note that when there is a linear dependency among features, there is not a unique value of <img src="https://latex.codecogs.com/gif.latex?\beta" /> that minimizes error. In these cases our results may differ from those produced by scikit-learn, but our algorithm also minimizes error.