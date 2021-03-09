import numpy as np
import time
import math
from numpy.linalg import svd
import operator


# Function to get top k movies
def top_movies(temp, k):
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
def select_rows(M, r):
	
	indices = [i for i in range(len(M))]
	frob = 0
	for i in range(len(M)):
		for j in range(len(M[i])):
			frob += M[i][j] ** 2

	p = np.zeros(len(M))
	for i in range(len(M)):
		m = 0
		for j in range(len(M[i])):
			m += M[i][j] ** 2
		p[i] = m / float(frob)

	rows_selected = np.random.choice(indices, r, True, p)

	R = np.zeros((r, len(M[0])))
	
	# Scaling the row selected by sqrt(r*pi)
	for i, row in zip(range(r), rows_selected):
		for j in range(len(M[row])):
			R[i][j] = M[row][j]
			R[i][j] = R[i][j] / float(math.sqrt(r*p[row]))

	return rows_selected, R

# Retains max_energy% of the total energy and returns the corresponding SVD matrices
def svd_reduce(U, Eigen, V, max_energy):
	energy = 0
	count = 0
	total_energy = 0
	for sigma in Eigen:
		total_energy += sigma**2
		
	for sigma in Eigen:		
		if (energy+sigma**2)/total_energy > max_energy:
			break
		else:
			energy += sigma**2
			count += 1
	
	newEigen = Eigen[:count]
	U = np.delete(U, np.s_[:-count], axis=1)
	V = np.delete(V, np.s_[:-count], axis=0)
	
	return U, newEigen, V
	
#CUR function
def CUR(B, r, k, top_k_movies_for_B):
	print("CUR...")
	start_time = time.time()
	# Selecting r rows
	row_indices, R = select_rows(B, r)
	column_indices, C = select_rows(B.T, r)
	C = C.T
		
	W = np.zeros((r, r))
	for i, row in zip(range(len(row_indices)), row_indices):
		for j, column in zip(range(len(column_indices)), column_indices):
			W[i][j] = B[row][column]
	
	# Calculating the SVD and getting the Eigenvalues and Eigen Vectors
	X, eigen_values, YT = svd(W, full_matrices = False)
	print("Original Eigen Values", X.shape, eigen_values.shape, YT.shape)
		
	# Calculating the new Eigenvalues and Eigen Vectors after retaining 90% energy
	X, eigen_values, YT = svd_reduce(X, eigen_values, YT, 0.9)
	print("New Eigen Values", X.shape, eigen_values.shape, YT.shape)
	r = len(eigen_values)
	
	sigma = np.zeros((r, r))
	sigma_plus = np.zeros((r, r))

	for i in range(len(eigen_values)):
		sigma[i][i] = math.sqrt(eigen_values[i])
		if(sigma[i][i] != 0):
			sigma_plus[i][i] = 1 / float(sigma[i][i])
    
	U = np.transpose(YT)@(sigma_plus@sigma_plus)@np.transpose(X)
    
	# CUR matrix
	cur_matrix = (C@U)@R
	count = 0
	top_k_movies_for_cur = top_movies(cur_matrix, k)
	for movie in top_k_movies_for_B:
		if(movie in top_k_movies_for_cur):
			count += 1
			
	print("Matched: " + str(count))
	print("Test Size: " + str(k))
	precision = float(count) / k

	squared_error_sum = 0
	mean_absolute_error = 0
	number_of_predictions = 0

	for i in range(len(B)):
		for j in range(len(B[i])):
			if(B[i][j] != 0):
				squared_error_sum += (B[i][j] - cur_matrix[i][j]) ** 2
				mean_absolute_error += abs(B[i][j] - cur_matrix[i][j])
				number_of_predictions += 1

	# Root mean square error
	rmse = math.sqrt(squared_error_sum/ float(number_of_predictions))
	mae = mean_absolute_error / float(number_of_predictions)

	print("RMSE: " + str(rmse))
	print("MAE: " + str(mae))
	print("Precision: " + str(precision))
	print("Time taken for CUR with 90% energy : " + str(time.time() - start_time))

	return

count = 0
users = 0
movies = 0
# Finding the number of Users and Movies
with open("movies.data", "r") as data_file:
	for line in data_file:
		count += 1
		line_values = line.split("\t")
		a = int(line_values[0])
		b = int(line_values[1])
		if(a > users):
			users = a
		if(b > movies):
			movies = b

Ratings = np.zeros((users + 1, movies + 1))

# Reading file
with open("movies.data", "r") as data_file:
	for line in data_file:
		line_values = line.split("\t")
		a = int(line_values[0])
		b = int(line_values[1])
		Ratings[a][b] = float(line_values[2])
            
data_file.close()


# Calling CUR function
test_size = 350
r = 500

test_k_movies = top_movies(Ratings, test_size)
CUR(Ratings, r, test_size, test_k_movies)