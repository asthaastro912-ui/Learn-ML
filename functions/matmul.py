import numPy as np



a = np.array([1,2,3])
## (3,)

b = np.array([[1,2,3],
              [4,5,6]])

## (3,3)

c = np.random.randn(10,20)

## (10,20)

d = np.random.randn(32,224,224,3)

## shape ??

def matmul(A, B):
    n,m = A.shape
    o,p = B.shape
    if m!=o:
        raise Exception("multiplication not possible")
    
    result = [[0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for j in range(p):
            sum = 0
            for k in range(m):
                sum+= A[i][k]*B[k][j]
            result[i][j]= sum

    return np.array(result)

a = [1, 2, 3]
b = [4, 5, 6]
def dot_product(a,b):
    """ dot product is for vectors so we expect the shape 
    to be (m,) or so (we assume thats the shape and then 
    we write)"""

    m = len(a)
    if m!= len(b):
        raise Exception("the vectors are not equal length or dim")
    res = 0
    for i in range(m):
        res+= a[i]*b[i]
    return res

"""The dot product measures how aligned two vectors are.
Attention computes
Q • K
Large value
↓
High attention.

Small value
↓
Ignore.

This is why Cosine Similarity exists
Cosine similarity removes the effect of magnitude.
That larger magnitude increases the dot product.


In attention why dont we use cosine similarity
So attention can express both
QKᵀ / √dₖ
alignment (direction)
confidence/strength (magnitude)
Cosine similarity throws away the second piece of information.
But why divide with √dₖ
The probabilities become extremely "peaky."
That causes:
unstable gradients, slower learning, one token dominating everything


"""