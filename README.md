# SVM (Support Vector Machine) - Interior Point Method

Lagrangian Dual of the Linear-Kernel Soft-Margin SVM (Eq. 1):

$$\max_{\alpha} \ \sum_{i=1}^N \alpha_i \ - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i\alpha_j y_i y_jx_ix_j^T $$
$$s.t. \ 0 \leq \alpha_i \leq C $$
$$\sum_{i=1}^N \alpha_i y_i = 0$$

C denotes class overlap. A larger C allows less class overlap.

The bias is obtained by (Eq. 2):

$$b \ = \frac{1}{|M|} \sum_{n\in M} \left( y_n \ - \sum_{i\in S}\alpha_iy_ix_nx_i^T\right)$$

Where M is the vector of indicies of data points whose alphas abide by the first constraint (0<=a_i<=C) and S is the vector of indices of support vectors (a_i>0).

A prediction is obtained by (Eq. 3):

$$y(x) = \sum_{i=1}^N\alpha_iy_ixx_i^T + b$$

where N is the number of support vectors (a_i>0)

Reference: Bishop, Pattern Recognition and Machine Learning (2006), Chapter 7

We model the inequality constraints of the first equation as log barriers (Eq. 4, Eq. 5):

$$\min_{\alpha} \ - \sum_{i=1}^N \alpha_i \ + \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i\alpha_j y_i y_jx_ix_j^T \ + \frac{1}{t}\left(\ - \sum_{i=1}^N\log\alpha_i\ - \sum_{i=1}^N\log(C-\alpha_i)\right)$$

$$\min_{\alpha} \ - t\sum_{i=1}^N \alpha_i \ + \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i\alpha_j y_i y_jx_ix_j^T \ \ - \sum_{i=1}^N\log\alpha_i\ - \sum_{i=1}^N\log(C-\alpha_i)$$

where t is the barrier parameter. t controls the trade-off between minimizing the original dual objective and enforcing the inequality constraints.

Let
$$f(\alpha) = \ - t\sum_{i=1}^N \alpha_i \ + \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i\alpha_j y_i y_jx_ix_j^T \ \ - \sum_{i=1}^N\log\alpha_i\ - \sum_{i=1}^N\log(C-\alpha_i)$$

Now we have to minimize $$f(\alpha) \ s.t. \  \sum_{i=1}^N \alpha_i y_i = 0$$

This is an equality constrained minimization. We solve using Newton's method (Eq. 6).

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
