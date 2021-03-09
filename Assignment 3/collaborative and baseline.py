import numpy as np
import pandas as pd
from scipy import spatial
import math
from time import time
from sklearn.model_selection import train_test_split


def Sort_Tuple2(tp):
    tp.sort(key=lambda x: (x[1], x[0]))
    return tp


# reading User File
User = [i.strip().split('::') for i in open('users.dat').readlines()]
user = pd.DataFrame(User)
user.columns = ['UserID', 'Gender', 'age', 'occupation', 'zip_code']


# reading Movie File
Movie = [i.strip().split('::') for i in open('movies.dat').readlines()]
movie = pd.DataFrame(Movie)
movie.columns = ['MovieID', 'Title', 'Genres']

modifiedList = []
for i in Movie:
    modifiedList.append(i[2].split('|'))
movie['GenreList'] = modifiedList

# reading Rating File
Rating = [i.strip().split('::') for i in open('ratings.dat').readlines()]
rating = pd.DataFrame(Rating)
rating.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

data_Train, data_Test = train_test_split(rating.iloc[:, [0, 1, 2]].values, test_size=0.0001)

DataTrain = pd.DataFrame({'UserId': data_Train[:, 0], 'MovieId': data_Train[:, 1], 'Rating': data_Train[:, 2]})


# checking for duplicates
nusers = DataTrain.UserId.unique().shape[0]
nmovies = movie.MovieID.unique().shape[0]

# creating User Movie Matrix
UserMovieR = np.zeros((nusers, nmovies))

for i in DataTrain.itertuples():
    if int(i[1]) < nusers+1 and int(i[2]) < nmovies+1:
        UserMovieR[int(i[1])-1, int(i[2])-1] = int(i[3])

usermovie = pd.DataFrame(UserMovieR)

normUsermovie = usermovie
print(normUsermovie)
for j in range(nusers):
    print(j)
    uarray = usermovie.iloc[[j], :].values
    sumv = uarray.sum()
    counter = np.count_nonzero(uarray)
    if counter > 0:
        avg = sumv/counter
    else:
        avg = 0
    for i in range(nmovies):
        if normUsermovie[i][j] != 0:
            normUsermovie[i][j] -= avg

mat1 = np.zeros((nusers, nmovies))

for i in DataTrain.itertuples():
    if int(i[1]) < nusers+1 and int(i[2]) < nmovies+1:
        mat1[int(i[1])-1, int(i[2])-1] = int(i[3])

lf = pd.DataFrame(mat1)

# function for Collaborative Recommender System
def collaborative():
    MSE2 = 0
    RMSE2 = 0
    size2 = 0
    start_time_item1 = time()
    for i in data_Test:
        print('User Id: ', end=' ')
        print(i[0], end=' ')
        print('Movie Id: ', end=' ')
        print(i[1])
        print('Actual Rating:', end=' ')
        print(i[2])
        m = int(i[1]) - 1
        u = int(i[0]) - 1
        if m > 3880:
            continue
        start_time_item = time()
        G = []
        for j in range(nmovies):
            if (lf[j][u] > 0):
                G.append((j, 1 - spatial.distance.cosine(normUsermovie.iloc[:, m], normUsermovie.iloc[:, [j]])))
        Gfinal = Sort_Tuple2(G)
        leng = len(Gfinal) - 1
        GUse = []
        if (leng >= 10):
            for j in range(leng - 10, leng):
                GUse.append(Gfinal[j])
        else:
            for j in range(leng):
                GUse.append(Gfinal[j])
        pratn = 0
        pratd = 0
        for j in GUse:
            pratn += j[1] * lf[j[0]][u]
            pratd += j[1]
        print('Predicted Rating:', end=' ')
        if (pratd == 0):
            continue
        pl = pratn / pratd
        print(pl)
        size2 += 1
        MSE2 = abs(lf[j[0]][u]) - abs(pl)
        RMSE2 = (lf[j[0]][u] - pl) ** 2
        print('counter: ', end=' ')
        print(size2)
        print('Time Taken: ', end=' ')
        print(time() - start_time_item)
    print(time() - start_time_item1)
    print(MSE2 / size2)
    print(math.sqrt(RMSE2 / size2))

# Calling Function
collaborative()


pq = DataTrain.iloc[:,[2]].values
sumv = 0
for i in pq:
    sumv += int(i)
globalMean = sumv/len(pq)


# function for Collaborative Recommender_baseline System
def collaborative_baseline():
    MSE3 = 0
    RMSE3 = 0
    size3 = 0
    start_time_item1 = time()
    for i in data_Test:
        print('User Id: ', end=' ')
        print(i[0], end=' ')
        print('Movie Id: ', end=' ')
        print(i[1])
        print('Actual Rating:', end=' ')
        print(i[2])
        m = int(i[1]) - 1
        u = int(i[0]) - 1
        if m > 3880:
            continue
        start_time_item = time()
        G = []
        for j in range(nmovies):
            if (lf[j][u] > 0):
                G.append((j, 1 - spatial.distance.cosine(normUsermovie.iloc[:, m], normUsermovie.iloc[:, [j]])))
        Gfinal = Sort_Tuple2(G)
        leng = len(Gfinal) - 1
        GUse = []
        if (leng >= 10):
            for j in range(leng - 10, leng):
                GUse.append(Gfinal[j])
        else:
            for j in range(leng):
                GUse.append(Gfinal[j])
        pratn = 0
        pratd = 0

        movieavg = 0
        sizev = 0
        for k in range(nusers):
            if (lf[m][k] > 0):
                sizev += 1
                movieavg += lf[m][k]
        if(sizev != 0):
            movief = (movieavg / sizev) - globalMean
        else:
            continue
        useravg = 0
        sizev = 0
        for k in range(nmovies):
            if (lf[k][u] > 0):
                sizev += 1
                useravg += lf[k][u]
        userf = (useravg / sizev) - globalMean

        baseline = globalMean + movief + userf

        for j in GUse:
            for k in range(nusers):
                if (lf[j[0]][k] > 0):
                    sizev += 1
                    movieavg += lf[j[0]][k]
            movief = (movieavg / sizev) - globalMean
            pratn += j[1] * (lf[j[0]][u] - (globalMean + movief + userf))
            pratd += j[1]
        print('Predicted Rating:', end=' ')
        if (pratd == 0):
            continue
        pl = baseline + (pratn / pratd)
        print(pl)
        size3 += 1
        MSE3 = abs(lf[j[0]][u]) - abs(pl)
        RMSE3 = (lf[j[0]][u] - pl) ** 2
        print('counter: ', end=' ')
        print(size3)
        print('Time Taken: ', end=' ')
        print(time() - start_time_item)
    print(time() - start_time_item1)
    print(MSE3 / size3)
    print(math.sqrt(RMSE3 / size3))

# Calling Function
collaborative_baseline()
