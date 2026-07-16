def conv2d(image, kernel,stride,pad):   
    kh, kw = kernel.shape
    image = pad_image(image,pad)
    h, w = image.shape
    out_h = ((h-kh)//stride) + 1
    out_w = ((w-kw)//stride) +1

    output = [[0 for _ in range(out_w)] for _ in range(out_h)]
    for i in range(out_h):
        for j in range(out_w):
            output[i][j] = mul(image,kernel,i*stride,j*stride)
    return output

def mul(image,kernel,start_i,start_j):
    kh, kw = kernel.shape
    sum = 0
    for i in range(0,kh):
        for j in range(0,kw):
            sum+= image[start_i+i][start_j+j]*kernel[i][j]
    return sum

def pad_image(image,pad):
    h,w = image.shape
    output = [[0 for _ in range(w+2*pad)] for _ in range(h+2*pad)]
    for i in range(pad,pad +h):
        for j in range(pad,pad+w):
            output[i][j] = image[i-pad][j-pad]
    return output
    