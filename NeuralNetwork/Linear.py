import functions.matmul as matmul
import numpy as np

class Linear:

    def __init__(self,in_features,out_features):
        """ input features = in_features
            out features = out_features """
        self.W = np.ones((in_features,out_features))
        self.b = np.ones(out_features)
    """Why can't we initialize all weights to zero?
The network can still compute predictions and the loss, but all
 neurons in the same layer start identically, produce identical outputs,
 receive identical gradients, and remain identical throughout training. 
   This symmetry prevents different neurons from learning different features. 
   Random initialization breaks this symmetry and allows neurons to specialize."""
        

        



    def forward(self,x):
        Y = matmul(x,self.W)+ self.b
        return Y


