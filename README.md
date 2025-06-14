# SVM (Support Vector Machine) - Interior Point Method

This project implements a linear-kernel soft-margin SVM by solving the dual problem using interior-point methods. Inequality constraints are enforced using logarithmic barrier functions, and the resulting equality-constrained optimization problem is solved with Newton's method. The implementation follows techniques outlined in Convex Optimization by Boyd and Vandenberghe and Pattern Recognition and Machine Learning by Bishop.

Lagrangian Dual of the Linear-Kernel Soft-Margin SVM (Eq. 1):

$$\max_{\alpha} \ \sum_{i=1}^N \alpha_i \ - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i\alpha_j y_i y_jx_ix_j^T $$
$$s.t. \ 0 \leq \alpha_i \leq C $$
$$\sum_{i=1}^N \alpha_i y_i = 0$$


X -> Array of N input features vectors

Y -> Vector of labels with length N ($$y_i = 1$$ for positive class, $$y_i = -1$$ for negative class)

$$\alpha_i$$ -> dual variable for the i-th data point

C denotes class overlap. A larger C allows less class overlap.

Above is the equation that must be optimized.

After reaching optimal values for alpha, we can make predictions using the alpha values, the labels, and the input feature vectors using the two following equations:

The bias is obtained by (Eq. 2):

$$b \ = \frac{1}{|M|} \sum_{n\in M} \left( y_n \ - \sum_{i\in S}\alpha_iy_ix_nx_i^T\right)$$

Where M is the vector of indicies of data points whose alphas abide by the first constraint (0<=a_i<=C) and S is the vector of indices of support vectors (a_i>0).

A prediction is obtained by (Eq. 3):

$$y(x) = \sum_{i=1}^N\alpha_iy_ixx_i^T + b$$

where N is the number of support vectors (a_i>0)

Reference: Bishop, Pattern Recognition and Machine Learning (2006), Chapter 7

To optmized the first equation, we model the inequality constraints as log barriers (Eq. 4, Eq. 5):

$$\min_{\alpha} \ - \sum_{i=1}^N \alpha_i \ + \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i\alpha_j y_i y_jx_ix_j^T \ + \frac{1}{t}\left(\ - \sum_{i=1}^N\log\alpha_i\ - \sum_{i=1}^N\log(C-\alpha_i)\right)$$

$$\min_{\alpha} \ - t\sum_{i=1}^N \alpha_i \ + \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i\alpha_j y_i y_jx_ix_j^T \ \ - \sum_{i=1}^N\log\alpha_i\ - \sum_{i=1}^N\log(C-\alpha_i)$$

where t is the barrier parameter. t controls the trade-off between minimizing the original dual objective and enforcing the inequality constraints.

Let
$$f(\alpha) = \ - t\sum_{i=1}^N \alpha_i \ + \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i\alpha_j y_i y_jx_ix_j^T \ \ - \sum_{i=1}^N\log\alpha_i\ - \sum_{i=1}^N\log(C-\alpha_i)$$

Now we have to minimize $$f(\alpha) \ s.t. \  \sum_{i=1}^N \alpha_i y_i = 0$$

This is an equality constrained minimization. We solve using Newton's method.

We calculate the Newton step by solving the following KKT system (Eq. 6):

$$\begin{bmatrix}\nabla^2f(\alpha)  &  Y\\\Y^T & 0\end{bmatrix}\begin{bmatrix}\Delta x_{nt} \\\ w\end{bmatrix}\ =\begin{bmatrix}-\nabla f(\alpha) \\\ 0 \end{bmatrix}$$

(The process of arriving at the above equation is shown in Convex Optimization by Boyd, Stephen, and Lieven Vandenberghe, chapter 10.) 

$$\Delta x_{nt}$$ is the newton step.

Newton decrement (as seen in Convex Optimization) (Eq. 7):

$$\sqrt{\Delta x_{nt}^T\nabla^2 f(\alpha)\Delta x_{nt}}$$

In a loop, we calculate the newton step and update the alphas:
$$A = A + ls_t*newtonStep$$

$$ls_t$$ is the learning rate parameter found through backtracking line search. Alternatively, one can set ls_t manually to avoid the extra computation of the backtracking line search algorithm. (Backtracking line search is covered in Chapter 9 of the Convex Optimization book).

We break out of the loop once newton_decrement squared, divided by 2, is less than the tolerance.

The newton method loop is nested inside the barrier method loop. In each subsequent iteration of the barrier method, we increase t (the barrier parameter) until we reach an optimal solution.

Vectorized form of f(A) (Eq. 8):

$$f(A) = t(-\textbf{1}^TA \ + \frac{1}2{}A^T(YY^T \odot XX^T)A) - \textbf{1}^T\log A \ - \textbf{1}^T\log(C-A)$$

Vectorized form of the gradient (Eq. 9):

$$\nabla f(A) = t(-\textbf{1} \ + (YY^T \odot XX^T)A) - \frac{1}{A} \ +\frac{1}{C-A}$$

Vectorized form of the hessian (Eq. 10):

$$\nabla^2 f(A) = t(YY^T \odot XX^T) + diag\left(\frac{1}{A^2}\right) \ + diag\left(\frac{1}{(C-A)^2}\right)$$
