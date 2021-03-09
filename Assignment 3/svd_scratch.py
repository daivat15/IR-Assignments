import pandas as pd
import numpy as np

from numpy import linalg as LA
from timeit import default_timer as timer
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

pd.options.mode.chained_assignment = None

USER_ID = 1               # Id of user to be recommended
NUM_RECOMMENDATIONS = 10    # Number of recommendations
ENERGY_FRACTION = 0       # Percentage energy, put percentage as 0 to use factors instead of percentage
NUMBER_OF_FACTORS = 50     # Number of rows in decomposition


def energy(sigma, reqd_energy):
    if sigma.ndim == 2:
        sigma = np.squeeze(sigma)

    total_energy = np.sum(sigma)
    percentage_reqd_energy = reqd_energy * total_energy / 100.0
    indexSum = sigma.cumsum()
    checkSum = indexSum <= percentage_reqd_energy
    last_singular_val_pos = np.argmin(checkSum)
    num_singular_vals = last_singular_val_pos + 1
    return num_singular_vals


def compute_svd(A, energy_retain, num_factors):
    # get eigen values and vectors
    eig_vals, eig_vecs = LA.eig(np.dot(A.T, A))
    eig_vals = np.absolute(np.real(eig_vals))

    # calculate the number of eigen values to retain
    if energy_retain == 100:
        eig_vals_num = LA.matrix_rank(np.dot(A.T, A))
    elif energy_retain == 0:
        eig_vals_num = num_factors
    else:
        eig_vals_num = energy(np.sort(eig_vals)[::-1], energy_retain)

    # place the eigen vectors according to increasing order of their corresponding eigen values to form V
    eig_vecs_num = np.argsort(eig_vals)[::-1][0:eig_vals_num]
    V = np.real(eig_vecs[:, eig_vecs_num])

    # Calculation of sigma | sort in decreasing order and fill till number of eigen values to retain
    sigma_vals = np.reshape(np.sqrt(np.sort(eig_vals)[::-1])[0:eig_vals_num], eig_vals_num)
    sigma = np.zeros([eig_vals_num, eig_vals_num])
    np.fill_diagonal(sigma, sigma_vals)
    U = np.dot(A, np.dot(V, LA.inv(sigma)))
    Vtr = V.T
    return U, sigma, Vtr


# Read files
ratings = [i.strip().split("::") for i in open('ratings.dat', 'r').readlines()]
users = [i.strip().split("::") for i in open('users.dat', 'r').readlines()]
movies = [i.strip().split("::") for i in open('movies.dat', 'r').readlines()]

ratings_df = pd.DataFrame(ratings, columns=['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype=int)
movies_df = pd.DataFrame(movies, columns=['MovieID', 'Title', 'Genres'])
movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)

start = timer()

# Create user to movie rating matrix, fill unrated movies as 0
um_rating_df = ratings_df.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)
um_rating = um_rating_df.values
um_rating = um_rating.astype(np.int)

# Subtract mean from every user to normalise values
user_rating_mean = np.mean(um_rating, axis=1)
um_rating = um_rating - user_rating_mean.reshape(-1, 1)

# SVD
u, sigma, v = compute_svd(um_rating, ENERGY_FRACTION, NUMBER_OF_FACTORS)

# Generate prediction matrix
predictions = np.dot(np.dot(u, sigma), v) + user_rating_mean.reshape(-1, 1)

end = timer()

# Generate RMSE and MAE
rmse = sqrt(mean_squared_error(um_rating + user_rating_mean.reshape(-1, 1), predictions))
print("RMSE: " + str(rmse))
mae = mean_absolute_error(um_rating + user_rating_mean.reshape(-1, 1), predictions)
print("MAE: " + str(mae))
print("Time taken to generate predictions: " + str(end - start) + "s")

predictions = pd.DataFrame(predictions, columns=um_rating_df.columns)

user_number = USER_ID - 1       # User number starts from 0 and id from 1
predictions = predictions.iloc[user_number].sort_values(ascending=False)    # Sort movies by predicted ratings
user_predictions = ratings_df[ratings_df['UserID'] == str(USER_ID)]         # Get ratings of current user

movies_df['MovieID'] = movies_df['MovieID'].astype(int)
user_predictions['MovieID'] = user_predictions['MovieID'].astype(int)
predictions = pd.DataFrame(predictions).reset_index()
predictions['MovieID'] = predictions['MovieID'].astype(int)

# Merge predictions with movie dataset
user_info = (user_predictions.merge(movies_df, how='left', left_on='MovieID', right_on='MovieID').sort_values(['Rating'], ascending=False))

# Get predictions for user
recommendations = (movies_df[~movies_df['MovieID'].isin(user_info['MovieID'])].merge(predictions, how='left', left_on='MovieID', right_on='MovieID').rename(columns={user_number: 'Predictions'}).sort_values('Predictions', ascending=False).iloc[:NUM_RECOMMENDATIONS, :-1])

print(recommendations['Title'])