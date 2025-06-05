import torch as T
from utils import accuracy

def backtracking_line_search(svm, newton_step, grads):
    ls_t = 1 #line search t
    ls_alpha = 0.1 #line search alpha
    ls_beta = 0.6 #line search beta
    
    backtracks = 0
    
    left = svm.f(svm.alphas + ls_t*newton_step)
    right = svm.f(svm.alphas) + ls_alpha*ls_t*T.matmul(grads.T, newton_step)
    while left > right:
        ls_t = ls_beta*ls_t
        backtracks += 1
    return ls_t, backtracks

def newtons_method(svm, tolerance, best_acc=0.0, 
                   barrier_method_counter=0, 
                   path="output.txt", fixed_lr=None
                   verbose = False):
    counter = 0
    while True:
        grad = svm.gradients_f(svm.alphas)
        hess = svm.hessian_f(svm.alphas)
        newton_step, newton_decrement = svm.get_newton_step(grad, hess)
        if (newton_decrement**2)/2 < tolerance:
            print("tolerance met. Breaking out of Newton's method. decrement: ", newton_decrement)
            print()
            break
    
        counter+=1
        backtracks = 0
        if(fixed_lr==None):
            ls_t, backtracks = backtracking_line_search(svm, newton_step, grad)
        else:
            ls_t = fixed_lr
        svm.alphas = svm.alphas + ls_t*newton_step
            
        bias = svm.get_bias()
        preds = svm.predict_vectorized(bias, bow_val)
        acc = accuracy(preds, labels_val)
        
        if(verbose and acc<=best_acc):
            print()
            print("barrier method counter: ", barrier_method_counter, "t: ", svm.t)
            print("newton's method iter: ", counter, ", newton decrement: ", newton_decrement, ", backtracks: ", backtracks)
            print("Accuracy: ", acc)
            print()

        if(acc > best_acc):
            print("High score!!")
            print("barrier method counter: ", barrier_method_counter, "t: ", svm.t)
            print("newton's method iter: ", counter, ", newton decrement: ", newton_decrement, ", backtracks: ", backtracks)
            print("Accuracy: ", acc)
            print()
            with open(path, "w") as f:
                f.write(",".join(map(str, svm.alphas.squeeze().tolist())))
            best_acc = acc
            
    return best_acc

def barrier_method(svm, mu=2, path="output.txt", fixed_lr=None, tolerance = 0.002, verbose=False):
    best_acc = 0.0
    for i in range(5000):
        best_acc = newtons_method(svm, tolerance, best_acc, i, path=path, 
                                  fixed_lr=fixed_lr, verbose=verbose)
        svm.t*=mu
        
        print("******************Finished One Iteration of Barrier Method*********************")
        print("barrier iteration: ", i, "t: ", svm.t)
        bias = svm.get_bias()
        preds = svm.predict_vectorized(bias, bow_val)
        acc = accuracy(preds, labels_val)
        print("Accuracy: ", acc)
        print()
