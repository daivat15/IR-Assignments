#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 19:31:15 2019

@author: rynaagrover
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from random import randint
import math
from timeit import default_timer as timer

ratings_df = pd.read_csv('ratings_data.csv', usecols=[0, 1, 2])
ratings = ratings_df.values.tolist() 

np.random.shuffle(ratings)

max_vals = list(np.max(np.array(ratings), axis=0))
mat = [[0]*int(max_vals[1])] * int(max_vals[0])
counter_nan = 0
for i in range(0, len(ratings)):
    if(math.isnan(ratings[i][0]) == True):
        print(i, ratings[i])
        counter_nan += 1
        ratings[i][0] = 0
    if(math.isnan(ratings[i][1]) == True):
        print(ratings[i])
        counter_nan += 1
        ratings[i][1] = 0
    if(math.isnan(ratings[i][2]) == True):
        print(i, ratings[i])
        counter_nan += 1
        ratings[i][2] = 0

data_train, data_test = train_test_split(ratings, test_size = 0.3, random_state = 0)

#print(counter_nan)

k = 6
learning_rate = 0.01
lambda_reg = 0.02
epoch = 10000
p = [[1]*k] * int(max_vals[0])
q = [[1]*int(max_vals[1])] * k
overall_mean_rating = 0.0
for i in data_train:
    overall_mean_rating += i[2]
overall_mean_rating /= len(data_train)
biasness = 0
#print(overall_mean_rating)

def baseline_rating(u, m):
    
    user_mean_rating = 0
    count = 0
    for i in data_train:
        if i[0] == u:
            count += 1
            user_mean_rating += i[2]
    if count != 0:
        user_mean_rating /= count
    
    movie_mean_rating = 0
    count = 0
    for i in data_train:
        if i[1] == m:
            count += 1
            movie_mean_rating += i[2]
    if count != 0:
        movie_mean_rating /= count
        
    user_bias = user_mean_rating - overall_mean_rating
    movie_bias = movie_mean_rating - overall_mean_rating
    
    return overall_mean_rating + user_bias + movie_bias

def sgd():
    ss_err = 0
    for iter_val in range(0, epoch):
        if iter_val % 10 == 0:
            print(iter_val)
        val = randint(0, len(ratings))
        biasness = 0
        i = int(ratings[val][0])-1
        j = int(ratings[val][1])-1
        
        r_cap = 0
        for counter in range(0, k):
            r_cap += p[i][counter] * q[counter][j]
            
        frob_p = 0
        for j_var in range(0, k):
            frob_p += p[i][j_var]**2
        
        frob_q = 0
        for i_var in range(0, k):
            frob_q += q[i_var][j]**2
            
        err_sq = (ratings[val][2] - biasness - r_cap) + lambda_reg* (frob_p**0.5 + frob_q**0.5) 
        
        for t in range(0, k):
            p[i][t] = p[i][t] + learning_rate * 2 * ((err_sq) * q[t][j] - lambda_reg * p[i][t])
            q[t][j] = q[t][j] + learning_rate * 2 * ((err_sq) * p[i][t] - lambda_reg * q[t][j])

start = timer()
sgd()        
ss_err = 0
mae = 0
for t in range (0, len(data_test)):
    i = data_test[t][0]-1
    j = data_test[t][1]-1
    r_cap = 0
    for counter in range(0, k):
        r_cap += p[i][counter] * q[counter][j]
    mae += abs(data_test[t][2] - r_cap - biasness)
    ss_err += (data_test[t][2] - r_cap - biasness)**2
end = timer()
print("RMSE - ")
print((ss_err/len(data_test)) ** 0.5)
print("MAE - ")
print(mae / len(data_test))
print("Time taken to generate predictions: " + str(end - start) + "s")
