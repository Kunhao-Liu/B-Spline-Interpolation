import numpy as np
import matplotlib.pyplot as plt
from basis import B

file = "eg1d5.txt"

# read the file at path
with open(file, 'r') as f:
    lines = f.readlines()
    degree = int(lines[0])
    num_control_points = int(lines[1])
    # next non empty line is the knot vector
    for i in range(2, len(lines)):
        if lines[i] != '\n':
            knot_vector = []
            for k in lines[i].split(' '):
                if k != '':
                    knot_vector.append(float(k))
            break
    # find the start line of the data points
    for j in range(i+1, len(lines)):
        if lines[j] != '\n':
            start_line = j
            break
    # read the control points
    control_points = [(float(line.split(' ')[0]), float(line.split(' ')[1])) for line in lines[start_line:start_line+num_control_points]]


# visualize the b-spline curve
u = np.linspace(knot_vector[degree], knot_vector[-degree-1], 100)
x = [0. for i in range(len(u))]
y = [0. for i in range(len(u))]
for i in range(len(u)):
    for j in range(num_control_points):
        x[i] += control_points[j][0] * B(u[i], degree, j, knot_vector)
        y[i] += control_points[j][1] * B(u[i], degree, j, knot_vector)

# visualize the control points
plt.plot([p[0] for p in control_points], [p[1] for p in control_points], 'ro')
# connect the control points
for i in range(len(control_points)-1):
    plt.plot([control_points[i][0], control_points[i+1][0]], [control_points[i][1], control_points[i+1][1]], 'r-')
plt.plot(x, y)
plt.show()