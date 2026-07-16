import functions.matmul as matmul ,functions.transpose as transpose
"""
X = [
 [1,0],
 [0,1],
 [1,1]
]

(3,2)

Wq.shape = ?
if Q = XWq and query dimension is 2

Answer: (2,2)

result Q is dimension of input(sequence length) * query dimension (dk)

Q = XWq

K = XWk

V = XWv

Q.shape = (3,2)

K.shape = (3,2)

V.shape = (3,2)

The same word now has three different representations.
Query = what this token wants.
Key = what this token offers.
Value = the information carried by this token.

"""

def mul(Q,K):
    return matmul(Q,transpose(K))

"""
QKᵀ

             Keys
           I love AI

Queries I   1  0  1
        love0  1  1
        AI  1  1  2

Each entry is a similarity score.
QKᵀ computes the similarity between every query vector 
and every key vector. These similarity scores tell the 
model how much attention each token should pay to every 
other token before combining information from the value 
vectors.

Why do we divide by √dₖ?
The dot product between queries and keys grows in magnitude
 as the key dimension increases. Large attention scores cause 
 the softmax to become very sharp, where one token gets almost
 all the probability and the gradients become very small. 
 Dividing by √dₖ normalizes the scale of the scores, preventing 
 softmax saturation and making training more stable."


If K helped us compute attention, why don't we multiply by K?
K is only for deciding where to look.
V contains the information to retrieve.

"""