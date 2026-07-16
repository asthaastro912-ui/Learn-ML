import numPy as np

def softmax(x):
    n = len(x)
    x = x-np.max(x)
    exp_x = np.exp(x)
    sum = 0
    for i in range(n):
        sum+=exp_x[i]
    
    for i in range(n):
        exp_x[i] = (exp_x[i]/sum)
    
    return exp_x

"""Why do we use ReLU instead of Sigmoid?
Because of vanishing gradient problem Sigmoid(100) ,Sigmoid(-100) = 1 gradient = 0 

What happens if a neuron always receives negative inputs?
It never updates again. This is called the Dying ReLU Problem.

The common solution is Leaky ReLU, which allows a small negative slope instead of 
outputting zero for all negative inputs."""

def relu(x):
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = 0

    return x

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

""" Sigmoid is still used in binary & multilabel classification
Sigmoid	                           Softmax
Binary classification	    Multi-class classification
Multi-label classification	Exactly one correct class
Independent probabilities	Probabilities sum to 1
"""