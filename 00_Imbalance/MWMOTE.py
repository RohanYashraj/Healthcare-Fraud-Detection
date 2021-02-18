#!/usr/bin/env python
# coding: utf-8

import math, bisect, random, logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s : %(message)s')

class Knn:
    """docstring for Knn"""

    def __init__(self):
        self.data = []
        self.dic = {}

    def fit(self, data):
        self.data = data
        self.real_indices = range(len(data))
        for i in range(len(data)):
            self.dic[(i, i)] = 0.
            for j in range(i):
                self.dic[(i, j)] = math.sqrt(math.fsum(((a - b) ** 2 for a, b in zip(self.data[i], self.data[j]))))
                self.dic[(j, i)] = self.dic[(i, j)]

    def fit_subset(self, indices):
        self.real_indices = indices

    def get_dis(self, a, b):
        return self.dic[(a, b)]

    def kneighbors(self, instance_index, n_neighbors, return_distance=False):
        result = []
        for i in self.real_indices:
            distance = self.dic[(instance_index, i)]
            result.append((distance, i))
        result = sorted(result)[:n_neighbors]

        if return_distance:
            return ([i[1] for i in result], [i[0] for i in result])
        else:
            return [i[1] for i in result]

class WeightedSampleRandomGenerator(object):
    def __init__(self, indices, weights):
        self.totals = []
        # self.indices = indices
        self.indices = list(indices)
        running_total = 0

        for w in weights:
            running_total += w
            self.totals.append(running_total)

    def next(self):
        rnd = random.random() * self.totals[-1]
        b = bisect.bisect_right(self.totals, rnd)

        i = self.indices[b]

        return self.indices[bisect.bisect_right(self.totals, rnd)]

    def __call__(self):
        return self.next()

def clus_dis(A, B, K):
    distance = 0.
    for i in A:
        for j in B:
            distance += K.get_dis(i, j)
    return distance / len(A) / len(B)

def MWMOTE(X, Y, S_min, S_maj, N, k1=5, k2=3, k3=0.5, C_th=5, CMAX=2, C_p=3, return_mode='only'):
    logger.debug('MWMOTE: Starting with %d instances' % len(Y))

    '''
    # Generating indices of S_min, S_maj
    S_min, S_maj = [], []

    #separate the S_min and S_maj
    for index, i in enumerate(Y):
        if i == 1:
            S_min.append(index)
        else:
            S_maj.append(index)
    '''
    if type(k3) == float:
        k3 = int(round(len(S_min) * k3))

    k = Knn()

    logger.debug(' Step   0: Computing Knn table')
    k.fit(X)

    # Step 1~2: Generating S_minf
    S_minf = []
    for i in S_min:
        neighbors = k.kneighbors(i, k1 + 1)  # remove itself from neighbors
        neighbors.remove(i)
        if not all((neighbor in S_maj) for neighbor in neighbors):
            S_minf.append(i)

    logger.debug(' Step 1~2: %d in S_minf' % len(S_minf))

    # Step 3~4: Generating S_bmaj
    k.fit_subset(S_maj)
    S_bmaj = []
    for i in S_minf:
        neighbors = k.kneighbors(i, k2)
        S_bmaj.extend(neighbors)
    S_bmaj = list(set(S_bmaj))
    logger.debug(' Step 3~4: %d in S_bmaj' % len(S_bmaj))

    # Step 5~6: Generating S_imin
    k.fit_subset(S_min)
    S_imin = []
    N_min = {}
    for i in S_bmaj:
        neighbors = k.kneighbors(i, k3)
        S_imin.extend(neighbors)
        N_min[i] = neighbors
    S_imin = list(set(S_imin))
    logger.debug(' Step 5~6: %d in S_imin' % len(S_imin))

    # Step 7~9: Generating I_w, S_w, S_p
    I_w = {}
    for y in S_bmaj:
        sum_C_f = 0.
        for x in S_imin:
            # closeness_factor
            if x not in N_min[y]:
                closeness_factor = 0.
            else:
                distance_n = math.sqrt(math.fsum(((a - b) ** 2 for a, b in zip(X[x], X[y])))) / len(X[x])
                if distance_n == 0:
                    closeness_factor = min(C_th, 0) / C_th * CMAX
                else:
                    closeness_factor = min(C_th, (1 / distance_n)) / C_th * CMAX

            I_w[(y, x)] = closeness_factor
            sum_C_f += I_w[(y, x)]
        for x in S_imin:
            closeness_factor = I_w[(y, x)]
            density_factor = closeness_factor / sum_C_f
            I_w[(y, x)] = closeness_factor * density_factor

    S_w = {}
    for x in S_imin:
        S_w[x] = math.fsum((I_w[(y, x)]) for y in S_bmaj)

    S_p = {}  # actually useless
    WeightSum = math.fsum(S_w.values())
    for x in S_w:
        S_p[x] = float(S_w[x]) / WeightSum
    logger.debug(' Step 7~9: %d in I_w' % len(I_w))

    # Step 10:Generating L, clusters of S_min
    d_avg = 0.
    for i in S_minf:
        tmp = []
        for j in S_minf:
            if i == j:
                continue
            tmp.append(math.sqrt(math.fsum(((a - b) ** 2 for a, b in zip(X[i], X[j])))))
        d_avg += min(tmp)
        d_avg /= len(S_minf)
        T_h = d_avg * C_p

        L = {index: [i] for index, i in enumerate(S_min)}

        clusters_number = list(range(len(S_min)))

        dis_table = [[0 for i in clusters_number] for j in clusters_number]
        for i in clusters_number:
            for j in clusters_number:
                dis_table[i][j] = clus_dis(L[i], L[j], k)
        MAX = max(max(j) for j in dis_table)
        for i in clusters_number:
            dis_table[i][i] = MAX

        for i in S_min:
            MIN = min(min(j) for j in dis_table)
            if MIN > T_h:
                break
            for j in clusters_number:
                if MIN in dis_table[j]:
                    b = dis_table[j].index(MIN)
                    a = j
                    break
            L[a].extend(L[b])

            del L[b]
            clusters_number.remove(b)
            for j in clusters_number:
                tmp = clus_dis(L[a], L[j], k)
                dis_table[a][j] = tmp
                dis_table[j][a] = tmp
            dis_table[a][a] = MAX
            for j in clusters_number:
                dis_table[b][j] = MAX
                dis_table[j][b] = MAX

        which_cluster = {}
        for i, clu in L.items():
            for j in clu:
                which_cluster[j] = i
        logger.debug(' Step  10: %d clusters' % len(L))

        # Step 11: Generating X_gen, Y_gen
        X_gen = []
        some_big_number = 10000000.
        sample = WeightedSampleRandomGenerator(S_w.keys(), S_w.values())
        for z in range(N):
            x = sample()
            print("x:", x)
            index = (L[which_cluster[x]])
            y = random.choice(index)
            alpha = random.randint(0, some_big_number) / some_big_number
            s = [i + alpha * (j - i) for i, j in zip(X[x], X[y])]
            X_gen.append(s)
        Y_gen = [1 for z in range(N)]
        logger.debug(' Step  11: %d over-sample generated' % N)

        # return the desired data
        # X.extend(X_gen)
        # Y.extend(Y_gen)
        if return_mode == 'append':
            return (X, Y)
        elif return_mode == 'shuffled':
            Permutation = range(len(X))
            random.shuffle(Permutation)
            X = [X[i] for i in Permutation]
            Y = [Y[i] for i in Permutation]
            return (X, Y)
        elif return_mode == 'only':
            return (X_gen, Y_gen)
        else:
            pass

import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np

# X_ret,Y_ret = MWMOTE(car_features,car_label,15419,k1 = 5, k2 = 3, k3 = 0.5, C_th = 5, CMAX = 2, C_p = 3, return_mode = 'only')

# divide the dataset into S_min and S_maj
# Divide the data into fraudulent and non-fraudulent data
# get the lable from the dataset

# read the dataset
Data = pd.read_csv('./Data/CarClaims/Pre-Processed_OneHotEncoding.csv')

X = Data.iloc[:, :-1]
y = Data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# car_label = car_features['FraudFound']
# drop the label
# car_features.drop(['FraudFound'], inplace=True, axis=1)

columnlable = list(X_train.columns.values)

# create the dataframe for fraudulent and non-fraudulent data
nonFraudulent = pd.DataFrame(columns=columnlable)
nonFraudulentLable = pd.DataFrame(columns=['FraudFound'])

fraudulent = pd.DataFrame(columns=columnlable)
fraudulentLable = pd.DataFrame(columns=['FraudFound'])

j = 0
k = 0
for i in range(X_train.shape[0]):
    if (y_train.iloc[i] == 0):
        nonFraudulent.loc[j] = X_train.iloc[i]
        nonFraudulentLable.loc[j] = 0
        j += 1
    else:
        fraudulent.loc[k] = X_train.iloc[i]
        fraudulentLable.loc[k] = 1
        k += 1

X = X_train.values
Y = y_train.values
S_min = fraudulent.values
S_maj = nonFraudulent.values

len(S_min)

x_new, label_new = MWMOTE(X, Y, S_min, S_maj, 10793)

# save the generated file.

generatedNewData = pd.DataFrame(x_new)

generatedNewData.shape

generatedNewData.head()

generatedNewData.to_csv(r'./Data/CarClaims/MWMOTE/mwmoteGeneratedOneHotData.csv', index=False)

count = label_new.count(-1)

print(count)

newLabel = pd.DataFrame(label_new)

newLabel.replace(-1, 1)

newLabel.to_csv(r'./Data/CarClaims/MWMOTE/mwmoteGeneratedLabel.csv', index=False )

X_test.to_csv(r"./Data/CarClaims/MWMOTE/X_test.csv", index=False)
y_test.to_csv(r"./Data/CarClaims/MWMOTE/y_test.csv", index=False)
