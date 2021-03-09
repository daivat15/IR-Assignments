import numpy as np
import pandas as pd
import time
import math
from numpy.linalg import svd
import operator


# Function to get top k movies
def get_top_k_movies(temp, k):
	movie_index_rating = []
	top_k_movies_for_temp = []
	avg_rating_of_movie = np.zeros(len(temp[0]))
	for j in range(len(temp[0])):
		number_of_users_rated = 0
		num = 0
		for i in range(len(temp)):
			if(temp[i][j] != 0):
				number_of_users_rated += 1
				num += temp[i][j]
		if(number_of_users_rated > 0):
			avg_rating_of_movie[j] = float(num) / number_of_users_rated
			movie_index_rating.append([j, avg_rating_of_movie[j]])

	sorted_movie_index_rating = sorted(movie_index_rating, key = operator.itemgetter(1), reverse = True)

	for i, index in zip(range(k), range(len(sorted_movie_index_rating))):
		top_k_movies_for_temp.append(sorted_movie_index_rating[i][0])

	return top_k_movies_for_temp


# Function to select random rows for CUR
def select_rows(B, r):
	
	indices = [i for i in range(len(B))]
	square_of_frobenius_norm_of_B = 0
	for i in range(len(B)):
		for j in range(len(B[i])):
			square_of_frobenius_norm_of_B += B[i][j] ** 2

	p = np.zeros(len(B))
	for i in range(len(B)):
		sum_of_squared_values_in_row = 0
		for j in range(len(B[i])):
			sum_of_squared_values_in_row += B[i][j] ** 2
		p[i] = sum_of_squared_values_in_row / float(square_of_frobenius_norm_of_B)

	rows_selected = np.random.choice(indices, r, True, p)

	R = np.zeros((r, len(B[0])))
	for i, row in zip(range(r), rows_selected):
		for j in range(len(B[row])):
			R[i][j] = B[row][j]
            
            
            
			R[i][j] = R[i][j] / float(math.sqrt(r*p[row]))

	return rows_selected, R

#CUR function
def cur_func(B, r, k, top_k_movies_for_B):
	print("In CUR function!")
	start_time = time.time()
	# Selecting r rows
	row_indices, R = select_rows(B, r)
	column_indices, C = select_rows(B.T, r)
	C = C.T
		
	W = np.zeros((r, r))
	for i, row in zip(range(len(row_indices)), row_indices):
		for j, column in zip(range(len(column_indices)), column_indices):
			W[i][j] = B[row][column]

	X, eigen_values, YT = svd(W, full_matrices = False)

	sigma = np.zeros((r, r))
	sigma_plus = np.zeros((r, r))

	for i in range(len(eigen_values)):
		sigma[i][i] = math.sqrt(eigen_values[i])
		if(sigma[i][i] != 0):
			sigma_plus[i][i] = 1 / float(sigma[i][i])
    
	U = np.transpose(YT)@(sigma_plus@sigma_plus)@np.transpose(X)
    
	print(C.shape, U.shape, R.shape)
	# CUR matrix
	cur_matrix = (C@U)@R
	
	count = 0
	top_k_movies_for_cur = get_top_k_movies(cur_matrix, k)
	for movie in top_k_movies_for_B:
		if(movie in top_k_movies_for_cur):
			count += 1
	print("count: " + str(count))
	print("k: " + str(k))
	precision_on_top_k = float(count) / k

	squared_error_sum = 0
	mean_absolute_error = 0
	number_of_predictions = 0

	for i in range(len(B)):
		for j in range(len(B[i])):
			if(B[i][j] != 0):
				squared_error_sum += (B[i][j] - cur_matrix[i][j]) ** 2
				mean_absolute_error += abs(B[i][j] - cur_matrix[i][j])
				number_of_predictions += 10

	print(number_of_predictions)
                        
	# Root mean square error
	rmse = math.sqrt(squared_error_sum )/ float(number_of_predictions)
	mae = mean_absolute_error / float(number_of_predictions)
	n, precision_on_top_k, squared_error_sum, rmse = number_of_predictions, precision_on_top_k, squared_error_sum, rmse


	print("RMSE for CUR: " + str(rmse*100))
	print("Mean Average Error for CUR: " + str(mae))
	print("Precision on top k for CUR with rows and columns: " + str(precision_on_top_k))
	print("Time taken for CUR with rows and columns : " + str(time.time() - start_time))

	start_time = time.time()
	row_indices, temp_matrix = select_rows(B, r)
	R = temp_matrix
	column_indices, temp_matrix = select_rows(B.T, r)
	C = temp_matrix.T
    
	print("Exiting CUR function!")
	return

user_ids_index = {}
movie_ids_index = {}
user_count = 0
movie_count = 0
count = 0
max_user_no = 0
max_movie_no = 0
movies_rated_by_user = {}
to_be_predicted = []
k = 50
r = 300

# Reading file for finding max movie id and max user id
with open("movies.data", "r") as data_file:
	for line in data_file:
		count += 1
		line_values = line.split("\t")
		a = int(line_values[0])
		b = int(line_values[1])
		if(a > max_user_no):
			max_user_no = a
		if(b > max_movie_no):
			max_movie_no = b

three_fourth_data_length = int(0.75 * count)
counter = 0
count_thousand_data_points = 0
A = np.zeros((max_user_no + 1, max_movie_no + 1))
temper = np.zeros((max_user_no + 1, max_movie_no + 1))
B = np.zeros((max_user_no + 1, max_movie_no + 1))

# Reading file
with open("movies.data", "r") as data_file:
	for line in data_file:
		line_values = line.split("\t")
		a = int(line_values[0])
		b = int(line_values[1])
		B[a][b] = float(line_values[2])
		if(counter <= three_fourth_data_length):
			A[a][b] = float(line_values[2])
			temper[a][b] = float(line_values[2])
			counter += 1
			if a not in movies_rated_by_user:
				movies_rated_by_user[a] = [b]
			else:
				movies_rated_by_user[a].append(b)
		elif(count_thousand_data_points < 120):
			to_be_predicted.append([b, a])
			count_thousand_data_points += 1
            
data_file.close()


# Getting top k rated movies for B
top_k_movies_for_B = get_top_k_movies(B, k)

# Calling CUR function
cur_func(B, r, k, top_k_movies_for_B)