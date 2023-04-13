import math
from basis import *

degree = 3 # degree of the B-spline curve

with open('input.txt', 'r') as f:
    lines = f.readlines()

data_points = [(int(line.split(' ')[0]), int(line.split(' ')[1])) for line in lines]
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

for i in range(num_data_points):
    print('i =', i)
    for j in range(len(knot_vector)-degree-1):
        print('N3({},{}) = {}'.format(knot_vector[i+degree], j, B(knot_vector[i+degree], degree, j, knot_vector)))

# print boundary conditions
print(  B(knot_vector[degree], degree, 0, knot_vector, derivative_order=2),
        B(knot_vector[degree], degree, 1, knot_vector, derivative_order=2),
        B(knot_vector[degree], degree, 2, knot_vector, derivative_order=2), '\n')


print(  B(knot_vector[-1-degree], degree, num_data_points-1, knot_vector, derivative_order=2),
        B(knot_vector[-1-degree], degree, num_data_points, knot_vector, derivative_order=2),
        B(knot_vector[-1-degree], degree, num_data_points+1, knot_vector, derivative_order=2),)