import math
import numpy as np
from basis import *

degree = 3 # degree of the B-spline curve

file_name = 'input_files/input.txt'
with open(file_name, 'r') as f:
    lines = f.readlines()

data_points = [(float(line.split(' ')[0]), float(line.split(' ')[1])) for line in lines]
num_data_points = len(data_points)
print('read data points:', data_points)

# compute the sum of the distances between the points
total_distance = 0
for i in range(1, len(data_points)):
    total_distance += math.sqrt((data_points[i][0] - data_points[i-1][0])**2 + (data_points[i][1] - data_points[i-1][1])**2)

# construct the knot vector using Chord Length Method
knot_vector = [0.]
for i in range(1, len(data_points)):
    knot_vector.append(knot_vector[i-1] + math.sqrt((data_points[i][0] - data_points[i-1][0])**2 + (data_points[i][1] - data_points[i-1][1])**2)/total_distance)

# append three 0 in the start and two 1 in the end of the knot vector
for i in range(degree):
    knot_vector.insert(0, 0.)
    knot_vector.append(1.)

print('knot vector:', knot_vector)


A = np.zeros((num_data_points, num_data_points+2))
for i in range(num_data_points):
    for j in range(num_data_points+2):
        A[i][j] = B(knot_vector[i+degree], degree, j, knot_vector)
        
# insert the boundary conditions
con1 = np.zeros(num_data_points+2)
con2 = np.zeros(num_data_points+2)
for i in range(num_data_points+2):
    con1[i] = B(knot_vector[degree], degree, i, knot_vector, derivative_order=2)
    con2[i] = B(knot_vector[-1-degree], degree, i, knot_vector, derivative_order=2)

# insert con1 to the scend row of A
A = np.insert(A, 1, con1, axis=0)
# insert con2 to the scend to last row of A
A = np.insert(A, -1, con2, axis=0)
det = np.linalg.det(A)
# Print the result
print(det)
print(A)

# convert x-coordinates and y-coordinates to two numpy vectors
data_x = np.array([p[0] for p in data_points])
data_y = np.array([p[1] for p in data_points])
# insert 0 in the start and end of x and y
data_x = np.insert(data_x, 1, 0)
data_x = np.insert(data_x, -1, 0)
data_y = np.insert(data_y, 1, 0)
data_y = np.insert(data_y, -1, 0)

# solve the linear system
control_x = np.linalg.solve(A, data_x)
control_y = np.linalg.solve(A, data_y)

print('control points', [(control_x[i], control_y[i]) for i in range(len(control_x))] )

# visualize the b-spline curve
u = np.linspace(knot_vector[degree], knot_vector[-degree-1], 100)
x = [0. for i in range(len(u))]
y = [0. for i in range(len(u))]
for i in range(len(u)):
    for j in range(len(control_x)):
        x[i] += control_x[j] * B(u[i], degree, j, knot_vector)
        y[i] += control_y[j] * B(u[i], degree, j, knot_vector)
        
import matplotlib.pyplot as plt
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x, y, 'r', label='B-spline curve')
# add labels


# plot the data points
for i in range(len(data_points)):
    plt.plot(data_points[i][0], data_points[i][1], 'bo', label='data points')

# plot the control points
for i in range(len(control_x)):
    plt.plot(control_x[i], control_y[i], 'gx' ,label='control points')
# connect the control points
for i in range(len(control_x)-1):
    plt.plot([control_x[i], control_x[i+1]], [control_y[i], control_y[i+1]], 'g--')

plt.axis('equal')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()

# write the b-spline curve to a file
with open('curve_'+file_name, 'w') as f:
    # write the degree in a line
    f.write(str(degree)+'\n')
    # write the number of control points in a line
    f.write(str(len(control_x))+'\n\n')

    # write the knot vector in a line, each value rounded to 4 decimal places
    for i in range(len(knot_vector)):
        f.write(str(round(knot_vector[i], 3))+' ')
    f.write('\n\n')

    # write the control points, each point in a line
    for i in range(len(control_x)):
        f.write(str(round(control_x[i], 3))+' '+str(round(control_y[i], 3))+'\n')
