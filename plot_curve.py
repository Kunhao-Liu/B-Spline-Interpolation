import numpy as np
import matplotlib.pyplot as plt

h = 25

def x_u(u):
    return 1.5 * (np.exp(1.5*np.sin(6.2*u - 0.027*h)) + 0.1) * np.cos(12.2*u)

def y_u(u):
    return (np.exp(np.sin(6.2*u - 0.027*h)) + 0.1) * np.sin(12.2*u)

# tasks
plot_curve = True
cubic_polynomial = True

if plot_curve:
    # plot the curve
    u = np.linspace(0, 1, 100, endpoint=True)
    x = [x_u(u_i) for u_i in u]
    y = [y_u(u_i) for u_i in u]
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y, 'y', label='r(u)')
    plt.axis('equal')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    # plt.show()

if cubic_polynomial:
    # sample points from the curve
    u = np.linspace(0, 1, 10, endpoint=True)
    x = np.array([[x_u(u_i)] for u_i in u]) # [10, 1]
    y = np.array([[y_u(u_i)] for u_i in u]) # [10, 1]

    A = np.array([[1, u_i, u_i**2, u_i**3] for u_i in u]) # [10, c]

    # get the x cubic polynomial
    coeff_x = np.linalg.solve(A.T @ A, np.squeeze(A.T @ x))
    # get the y cubic polynomial
    coeff_y = np.linalg.solve(A.T @ A, np.squeeze(A.T @ y))
    
    def cubic_polynomial_x(u):
        return coeff_x[0] + coeff_x[1]*u + coeff_x[2]*u**2 + coeff_x[3]*u**3
    
    def cubic_polynomial_y(u):
        return coeff_y[0] + coeff_y[1]*u + coeff_y[2]*u**2 + coeff_y[3]*u**3
    
    # plot the sampled points
    plt.plot(x, y, 'gx', label='sampled points')

    # plot the curve
    u = np.linspace(0, 1, 100, endpoint=True)
    x = [cubic_polynomial_x(u_i) for u_i in u]
    y = [cubic_polynomial_y(u_i) for u_i in u]
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y, 'r', label='cubic polynomial curve')
    plt.axis('equal')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()

