def B(x, k, i, t, derivative_order=0):
    if k == 0:
        if x == t[-k-1]: x -= 1e-7
        return 1.0 if t[i] <= x < t[i+1] else 0.0
    
    if t[i+k] == t[i]:
        c1 = 0.0
    else:
        if derivative_order == 0:
            c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
        else:
            c1 = k/(t[i+k] - t[i]) * B(x, k-1, i, t, derivative_order-1)

    if t[i+k+1] == t[i+1]:
        c2 = 0.0
    else:
        if derivative_order == 0:
            c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
        else:
            c2 = - k/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t, derivative_order-1)

    return c1 + c2

