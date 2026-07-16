def transpose(A):
    m,n = A.shape
    res = [[0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            res[i][j] = A[j][i]
    return res