import numpy as np
import matplotlib.pyplot as plt
from b_spline_interpolation import *

h = 25

def x_u(u):
    return 1.5 * (np.exp(1.5*np.sin(6.2*u - 0.027*h)) + 0.1) * np.cos(12.2*u)

def y_u(u):
    return (np.exp(np.sin(6.2*u - 0.027*h)) + 0.1) * np.sin(12.2*u)

# tasks
plot_curve = True
cubic_polynomial = False
b_spline = True

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
    num_points = 10
    u = np.linspace(0, 1, num_points, endpoint=True)
    # u = np.random.uniform(low=0, high=1, size=num_points)
    # u = np.clip(np.random.normal(0.5, 0.2, num_points), 0, 1) 

    u.sort()
    print(u)

    x = np.array([[x_u(u_i)] for u_i in u]) 
    y = np.array([[y_u(u_i)] for u_i in u]) 

    A = np.array([[1, u_i, u_i**2, u_i**3] for u_i in u]) # [10, c]

    # get the x cubic polynomial
    coeff_x = np.linalg.solve(A.T @ A, np.squeeze(A.T @ x))
    # get the y cubic polynomial
    coeff_y = np.linalg.solve(A.T @ A, np.squeeze(A.T @ y))
    
    def cubic_polynomial_x(u):
        return coeff_x[0] + coeff_x[1]*u + coeff_x[2]*u**2 + coeff_x[3]*u**3
    
    def cubic_polynomial_y(u):
        return coeff_y[0] + coeff_y[1]*u + coeff_y[2]*u**2 + coeff_y[3]*u**3
    
    # calculate the error
    error = 0
    for u_i in u:
        error += (cubic_polynomial_x(u_i) - x_u(u_i))**2 + (cubic_polynomial_y(u_i) - y_u(u_i))**2
    error /= num_points
    error = np.sqrt(error)
    print(f'RMSE: {error.round(4)}')

    # plot the sampled points
    plt.plot(x, y, 'gx', label='sampled points')

    # plot the curve
    u = np.linspace(0, 1, 100, endpoint=True)
    x = [cubic_polynomial_x(u_i) for u_i in u]
    y = [cubic_polynomial_y(u_i) for u_i in u]

    plt.title(f'Randomly Sample {num_points} Points from a Normal Distribution')
    # plt.title(f'Unifomly Sample {num_points} Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y, 'r', label='cubic polynomial curve')
    plt.axis('equal')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()

if b_spline:
    degree = 3

    # sample points from the curve
    num_points = 10
    u = np.linspace(0, 1, num_points, endpoint=True)
    # u = np.random.uniform(low=0, high=1, size=num_points)
    # u = np.clip(np.random.normal(0.5, 0.2, num_points), 0, 1) 
    # u = np.insert(u, 0, 0)
    # u = np.insert(u, -1, 1)

    u.sort()

    x = np.array([[x_u(u_i)] for u_i in u]) 
    y = np.array([[y_u(u_i)] for u_i in u]) 
    data_points = np.concatenate((x, y), axis=1)
    plt.plot(x, y, 'gx', label='sampled points')

    # get the b-spline curve
    control_x, control_y, knot_vector = b_spline_interpolation(data_points)

    # visualize the b-spline curve
    u = np.linspace(knot_vector[degree], knot_vector[-degree-1], 200)
    x = [0. for i in range(len(u))]
    y = [0. for i in range(len(u))]
    for i in range(len(u)):
        for j in range(len(control_x)):
            x[i] += control_x[j] * B(u[i], degree, j, knot_vector)
            y[i] += control_y[j] * B(u[i], degree, j, knot_vector)

    plt.title(f'Randomly Sample {num_points} Points from a Normal Distribution')
    # plt.title(f'Unifomly Sample {num_points} Points')
    # plt.title(f'Uniform Parameterization')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y, 'r', label='cubic B-spline curve')
    plt.axis('equal')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()


