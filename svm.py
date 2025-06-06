import torch as T

class SVM:
    def __init__(self, x, y, alphas=None, C=1, t=1):
        self.x = x
        self.y = y.float()
        self.alphas = alphas if alphas!=None else T.zeros(y.shape).float()
        self.ones = T.ones(y.shape).float()
        self.epsilon = 1e-8
        self.C = C
        self.t = t
    
    def get_bias(self):
        #vectorized form of (Eq. 2) in the readme
        margin_indices = T.where((self.alphas > 0) & (self.alphas <= self.C))[0]
        support_indices = T.where((self.alphas > 0))[0]
        ay = self.alphas[support_indices]*self.y[support_indices]
        ay = ay.repeat((1, margin_indices.shape[0])).T
        x_matmul = T.matmul(self.x[margin_indices],self.x[support_indices].T)
        bias = T.sum(self.y[margin_indices].squeeze(-1) - T.sum(ay*x_matmul, dim=1))
        bias = bias/margin_indices.shape[0]
        return bias
    
    def predict(self, bias, point):
        #vectorized form of (Eq. 3) in the readme
        indices = T.where((self.alphas > 0))[0]  
        pred = T.sum(self.alphas[indices].T*self.y[indices].T*T.matmul(point, self.x[indices].T)) + bias
        return pred

    def predict_vectorized(self, bias, points):
        #further vectorized to allow for prediction on multiple points
        indices = T.where((self.alphas > 0))[0]
        ay = self.alphas[indices]*self.y[indices]
        ay = ay.repeat((1, points.shape[0])) 
        pred = T.sum(ay.T * T.matmul(points, self.x[indices].T), dim=1) + bias
        return pred
    
    def f(self, inputs):
        #implementation of (Eq. 8) in the readme
        yx = T.matmul(self.y, self.y.T)*T.matmul(self.x, self.x.T)
        objective = -T.matmul(self.ones.T, inputs) + \
                    T.matmul(T.matmul(inputs.T, yx), inputs)/2
        inequality_log_terms = T.sum(-T.log(inputs+self.epsilon) \
                               -T.log(self.C-inputs+self.epsilon))
        return self.t*objective + inequality_log_terms

    def gradients_f(self, inputs):
        #implementation of (Eq. 9) in the readme
        yx = T.matmul(self.y, self.y.T)*T.matmul(self.x, self.x.T)
        gradient = self.t*(-self.ones + T.matmul(yx, inputs)) \
                   - (1/(inputs+self.epsilon))\
                   + (1/(self.C-inputs+self.epsilon))
        return gradient

    def hessian_f(self, inputs):
        #implementation of (Eq. 10) in the readme
        diag = T.eye(inputs.shape[0])
        yx = T.matmul(self.y, self.y.T)*T.matmul(self.x, self.x.T)
        hessian = self.t*yx + (1/(inputs**2+self.epsilon))*diag \
                  + (1/((self.C-inputs)**2+self.epsilon))*diag
        return hessian
    
    def get_newton_step(self, gradient, hessian):
        #implementation of (Eq. 6) in the readme
        #constraint: (y.T)(alpha) = 0

        #concat shape (num_points, num_points) with (num_points, 1) to get (num_points, num_points+1)
        kkt_matrix = T.cat((hessian, self.y), dim=1)

        #add a 0 to the end of coeffecients vector -> shape: [1, num_points+1]
        coeff_vec = T.cat((self.y.T, T.tensor([[0.0]])), dim=1)

        #concat with matrix to get shape of [num_points+1, num_points+1]
        kkt_matrix = T.cat((kkt_matrix, coeff_vec), dim=0)

        #right hand side -> negative gradient with a concatenated 0
        #from [num_points, 1] to [num_points+1, 1]
        rhs = T.concat((-gradient, T.tensor([[0.0]])), dim=0)

        #print("det kkt: ", T.linalg.det(kkt_matrix))
        #print("cond kkt: ", T.linalg.cond(kkt_matrix))

        solution = T.linalg.solve(kkt_matrix, rhs)

        newton_step = solution[:self.y.shape[0]] #shape: [num_points, 1]
        w = solution[-1,-1]

        #from (Eq. 7) in the readme
        newton_decrement = T.sqrt(T.matmul(newton_step.T, T.matmul(hessian, newton_step))).squeeze()

        return newton_step, newton_decrement
