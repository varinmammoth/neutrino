#%%
import numpy as np
import matplotlib.pyplot as plt
# %%
class linEq():
    """Note: this class only works where A is an NxN matrix,
    ie. N unknowns with N equations.
    """
    def __init__(self, A, b):
        if type(A) == np.matrix or type(A) == np.ndarray:
            A.tolist()
        if type(b) == np.ndarray:
            b.tolist()

        self.A = A
        self.b = b
        self.L = None
        self.U = None

    #----Helper functions----

    #function to return square matrix with 1's as diaganol
    def ones_diagonal_square(self, N):
        output = []
        for i in range(0,N):
            row = []
            for j in range(0,N):
                if i == j:
                    row.append(1)
                else:
                    row.append(0)
            output.append(row)
        return output

    #function to return square matrix of 0's
    def zeros_square(self, N):
        output = []
        for i in range(0,N):
            row = []
            for j in range(0,N):
                row.append(0)
            output.append(row)
        return output

    #----End of helper functions----

    def LU_decompose(self):
        """Decomposes A into A = LU, where L and U are lower
        and upper diagonal matrices respectively.

        Returns:
            list: two NxN lists which are L and U
        """
        A = self.A
        N = len(A)
        
        #initalise L and U in Doolittle scheme
        L = self.ones_diagonal_square(N)
        U = self.zeros_square(N)

        for j in range(0,N):
            for i in range(0,j+1):
                summ = 0
                for k in range(0,i):
                    summ += L[i][k]*U[k][j]
                U[i][j] = A[i][j] - summ
            
            for i in range(j+1,N):
                summ = 0
                for k in range(0,j):
                    summ += L[i][k]*U[k][j]
                L[i][j] = (A[i][j] - summ)/U[j][j]

        self.L = L
        self.U = U
        
        return L, U

    def forward_sub(self, L, b):
        """Finds the solution to Lx=b, where L is a lower
        diagonal matrix.

        Args:
            L (list): Lower diagonal matrix
            b (list): Vector of constnats

        Returns:
            list: solution vector x, to Lx=b
        """
        x = [0]*len(b)

        x[0] = b[0]/L[0][0]

        for i in range(1,len(x)):
            summ = 0
            for j in range(0, i):
                summ += L[i][j]*x[j]
            x[i] = (b[i] - summ)/L[i][i]
    
        return x

    def backward_sub(self, U, b):
        """Finds the solution to Ux=b, where U is a upper
        diagonal matrix.

        Args:
            U (list): Upper diagonal matrix
            b (list): Vector of constants

        Returns:
            list: solution vector x, to Ux=b
        """
        x = [0]*len(b)

        x[-1] = b[-1]/U[-1][-1]

        i = len(b)-2
        while i >= 0:
            summ = 0
            for j in range(i+1, len(b)):
                summ += U[i][j]*x[j]
            x[i] = (b[i] - summ)/U[i][i]

            i -= 1

        return x

    def solve(self, method='LU'):
        """Solves Ax=b using LU decomposition.

        Returns:
            list: two lists, the solution and errors
        """
        if method == 'LU':
            if self.L == None or self.U == None:
                self.LU_decompose()

            y = self.forward_sub(self.L, self.b)
            x = self.backward_sub(self.U, y)

            self.x_LU = x

            error = np.matmul(np.matrix(self.A), np.array(x)) - np.array(self.b)
            self.x_LU_err = error.tolist()[0]

            return x, error

    def return_sol(self, method='LU', returnError=False):
        """Returns solution obtained using a particuliar method.

        Args:
            method (str, optional): Method. Method could be
            'LU', or ... Defaults to 'LU'.
            returnError (bool, optional): Set to True to return error. Defaults to False.

        Returns:
            _type_: _description_
        """
        if method == 'LU':
            if returnError == True:
                return self.x_LU, self.x_LU_err
            else:
                return self.x_LU

    def invert_A(self, method='LU'):
        """Inverts A using the selected method.

        Args:
            method (str, optional): Can choose from 'LU',.... Defaults to 'LU'.

        Returns:
            list, list: The inverse of A and the error matrix.
        """
        A = self.A
        N = len(A)
        b_T = self.ones_diagonal_square(N) #transpose of b ie. the same thing since b is identity matrix
        x_T = [] #tranpose of x

        if method == 'LU':
            for b in b_T:
                eq = linEq(A, b)
                eq.LU_decompose()
                eq.solve(method='LU')
                x = eq.return_sol(method='LU')
                x_T.append(x)

        x = np.transpose(np.matrix(x_T))
        error = np.matmul(np.matrix(A), x) - np.matrix(b_T)
        x = x.tolist()
        error = error.tolist()

        self.A_inverse_LU = x
        self.A_inverse_LU_err = error
        
        return x, error

    def return_A_inverse(self, method='LU', returnError=False):
        """Returns the inverse of A obtained using the selected method.

        Args:
            method (str, optional): Choose from 'LU', .... Defaults to 'LU'.
            returnError (bool, optional): Set to True to return error matrix. Defaults to False.

        Returns:
            list, list: two lists (actually matrix but not numpy matrix): the inverse (and error matrix)
        """
        if method == 'LU':
            if returnError == True:
                return self.A_inverse_LU, self.A_inverse_LU_err
            else:
                return self.A_inverse_LU

    def accuracy_score(self, method='LU'):
        """Returns accuracy score of the solution and inverse of A
        obtained using the selected method.

        Args:
            method (str, optional): Chosse from 'LU', .... Defaults to 'LU'.
        """
        def vector_accuracy(vector):
            N = len(vector)
            accuracy = 0
            for i in vector:
                accuracy += abs(i)/N
            return accuracy

        def matrix_accuracy(matrix):
            N = len(matrix)
            N_squared = N**2
            accuracy = 0
            for row in range(0,N):
                for col in range(0,N):
                    accuracy += abs(matrix[row][col])/N_squared
            return accuracy

        
        try:
            x_accuracy = self.x_accuracy_LU
        except:
            try:
                err_vector = self.return_sol(method='LU', returnError=True)[1]
                x_accuracy = vector_accuracy(err_vector)
                self.x_accuracy_LU = x_accuracy
            except:
                x_accuracy = 'Not yet solved'

        try:
            A_inverse_accuracy = self.A_inverse_accuracy_LU
        except:
            try:
                err_matrix = self.return_A_inverse(method=method, returnError=True)[1]
                A_inverse_accuracy = matrix_accuracy(err_matrix)
                self.A_inverse_accuracy_LU = A_inverse_accuracy
            except:
                A_inverse_accuracy = 'Not yet solved'

        return x_accuracy, A_inverse_accuracy

class interpolator():
    def __init__(self, x, f):
        #warning: x must be sorted
        self.x = x 
        self.f = f

    def cubicspline(self):
        n = len(self.x) - 1 #indexing from 0 to n
        N = len(self.x) #number of given datapoints

        x = self.x
        f = self.f

        A = np.zeros((N,N))
        b = np.zeros(N)

        #generate the A matrix and b vector:
        A[0][0] = 1 #manually set value of f0'' and fn''
        A[n][n] = 1
        for i in range(1, n): #looping through i=1 to i=n-1 (inclusive of n-1)
            A[i][i-1] = (x[i]-x[i-1])/6
            A[i][i] = (x[i+1]-x[i-1])/3
            A[i][i+1] = (x[i+1]-x[i])/6
            b[i] = (f[i+1]-f[i])/(x[i+1]-x[i]) - (f[i]-f[i-1])/(x[i]-x[i-1])
        
        #Solve for f0'' to fn''
        linear_solver_obj = linEq(A, b) #create the solver instance
        linear_solver_obj.solve(method='LU') #solve using LU decomposition
        f_pp = linear_solver_obj.return_sol(method='LU') #returns the solution found using LU decomposition

        self.A = A
        self.B = b
        self.f_pp = f_pp

        #create the interpolation function to be returned to the user
        def interpolation_function(x_input_ls):
            if min(x_input_ls) < min(x):
                print("At least one of the inputted x-values is below the interpolation range.")
                return 
            elif max(x_input_ls) > max(x):
                print("At least one of the inputted x-values is above the interpolation range.")
                return

            f_interp = []

            for x_input in x_input_ls:
                for i, x_data in enumerate(x):
                    if x_input == x_data:
                        f_interp.append(f[i])
                        break
                    elif x_input > x_data and x_input < x[i+1]:
                        A = (x[i+1]-x_input)/(x[i+1]-x[i])
                        B = 1-A
                        C = (1/6)*(A**3 - A)*((x[i+1]-x[i])**2)
                        D = (1/6)*(B**3 - B)*((x[i+1]-x[i])**2)
                        interpolation_result = A*f[i] + B*f[i+1] + C*f_pp[i] + D*f_pp[i+1]
                        f_interp.append(interpolation_result)
                        break
            
            return np.array(f_interp)

        return interpolation_function
# %%
x = np.linspace(0,2*np.pi,50)
f = np.exp(-x)*(np.sin(10*x))

interpolator_obj = interpolator(x, f)
func = interpolator_obj.cubicspline()

x_intrp = np.linspace(0, 2*np.pi, 1000)
f_intrp = func(x_intrp)

plt.plot(x, f, 'x')
plt.plot(x_intrp, f_intrp, '.')
plt.show()
# %%
