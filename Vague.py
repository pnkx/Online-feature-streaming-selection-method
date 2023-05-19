# coding=utf-8
import copy
import math
from copy import deepcopy
from scipy import integrate
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KNeighborsClassifier
from ReliefF import ReliefF


def relation_two(X1, X2, Y, relation_row):
    X1 = normalize(X1)
    X2 = normalize(X2)
    N = get_kind(Y)  
    answer = 0
    answer_flag = 0
    for i in range(N):
        now_x1 = []
        now_x2 = []
        now_x = 0
        now_y = 0
        now_y_flag = 0
        for k in range(relation_row):
            if Y[k] == i:
                now_x1.append(X1[k])
                now_x2.append(X2[k])
        now_x = distcorr(now_x1, now_x2)
        for j in range(i + 1, N + 1):
            now_y1 = []
            now_y2 = []
            temp_y = 0

            for k in range(relation_row):
                if Y[k] == j:
                    now_y1.append(X1[k])
                    now_y2.append(X2[k])
            temp_y = distcorr(now_y1, now_y2)
            now_y = now_y + temp_y
            now_y_flag += 1
        now_y = now_y / now_y_flag
        answer = answer + card_integrate(now_x, now_y)
        answer_flag += 1
    return answer / answer_flag


def distcorr(X, Y):
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum() / float(n * n)
    dcov2_xx = (A * A).sum() / float(n * n)
    dcov2_yy = (B * B).sum() / float(n * n)
    dcor = np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor


def func(x):
    return x


def card_integrate(integrate_x, integrate_y):
    result = integrate.quad(func, integrate_x, integrate_y)
    return result[0] + integrate_x


def normalize(_x):  
    n_top = _x[0]
    n_bottom = _x[0]
    for n_i in range(len(_x)):
        temporary_x_value = _x[n_i]
        if temporary_x_value > n_top:
            n_top = temporary_x_value
        if temporary_x_value < n_bottom:
            n_bottom = temporary_x_value
    nor_col = []
    for n_in in range(len(_x)):
        if _x[n_in] - n_bottom == 0:
            temporary = 0
        else:
            temporary = (_x[n_in] - n_bottom) / (n_top - n_bottom)
        nor_col.append(temporary)
    return nor_col


def get_kind(kind_y):
    kind_flag = 1
    for kind_i in range(len(kind_y)):
        if kind_y[kind_i] > kind_flag:
            kind_flag = kind_y[kind_i]
    return kind_flag


def dep_cor(cor_x, cor_y, cor_row):
    _cor_x = np.array(cor_x)
    dim_dis = _cor_x.ndim  
    _D = cor_x.copy()
    cor_flag = get_kind(cor_y) 
    pos_flag = 0  
    pos_sum = 0  
    con = 1
    if dim_dis == 1:
        _d = normalize(_D)
    else:
        con = _cor_x.shape[1]
        _d = np.zeros(cor_row, dtype=float)
        for cor_i in range(con):
            _D_temporary = np.zeros(cor_row)
            for cor_j in range(cor_row):
                _D_temporary[cor_j] = _D[cor_j][cor_i]
            _cor_temporary = normalize(_D_temporary)
            for cor_j in range(cor_row):
                _d[cor_j] = _d[cor_j] + math.pow(_cor_temporary[cor_j], 2)
    for cor_i in range(cor_row):
        _d[cor_i] = math.sqrt(_d[cor_i]) 
    _d.sort()  
    o = sorted(enumerate(cor_x), key=lambda density_x: density_x[1])  
    l = []
    for vague_i in range(len(cor_x)):
        l.append(o[vague_i][0])  
    for cor_m in range(cor_flag):
        beg = 0
        power = 1
        power_1 = 0
        sum_dis_1 = 0
        pos_flag += 1
        neg_flag = 0
        neg_sum = 0
        for cor_i in range(cor_row):
            if cor_y[l[cor_i]] == cor_m:
                beg = cor_i
                break
        for cor_i in range(beg, cor_row):
            if cor_y[l[cor_i]] != cor_m and cor_y[l[cor_i - 1]] == cor_m:
                for cor_j in range(cor_i, cor_row):
                    if cor_y[l[cor_j]] == cor_m:
                        power_1 += 1
                        cor_temporary = 0
                        if dim_dis != 1:
                            for cor_k in range(con):
                                cor_temporary = cor_temporary + math.pow(_cor_x[cor_i][cor_k] - _cor_x[cor_j][cor_k], 2)
                        else:
                            cor_temporary = math.pow(_cor_x[cor_i] - _cor_x[cor_j], 2)
                        sum_dis_1 = sum_dis_1 + (np.sqrt(cor_temporary) / con)
                        break
            if cor_y[l[cor_i]] == cor_m:
                power = power + 1
        if sum_dis_1 != 0:
            sum_dis_1 = (power_1 / power)
        else:
            sum_dis_1 = power_1 / power

        for cor_n in range(cor_m + 1, cor_flag + 1):
            beg = 0
            power_2 = 0
            sum_dis_2 = 0
            neg_flag += 1
            power = 1
            for cor_i in range(cor_row):
                if cor_y[l[cor_i]] == cor_n:
                    beg = cor_i
                    break
            for cor_i in range(beg, cor_row):
                if cor_y[l[cor_i]] != cor_n and cor_y[l[cor_i - 1]] == cor_n:
                    for cor_j in range(cor_i, cor_row):
                        if cor_y[l[cor_j]] == cor_n:
                            power_2 += 1
                            cor_temporary = 0
                            if dim_dis != 1:
                                for cor_k in range(con):
                                    cor_temporary = cor_temporary + math.pow(
                                        _cor_x[cor_i][cor_k] - _cor_x[cor_j][cor_k], 2)
                            else:
                                cor_temporary = math.pow(_cor_x[cor_i] - _cor_x[cor_j], 2)
                            sum_dis_2 = sum_dis_2 + (np.sqrt(cor_temporary) / con)
                            break
                if cor_y[l[cor_i]] == cor_n:
                    power = power + 1
            if sum_dis_2 != 0:
                # sum_dis_2 = (sum_dis_2 / power) * (power_2 / power)
                sum_dis_2 = (power_2 / power)
            else:
                sum_dis_2 = power_2 / power
            neg_sum = neg_sum + sum_dis_2
        if neg_flag != 0:
            neg_sum = neg_sum / neg_flag
        else:
            print('error')
        if sum_dis_1 > neg_sum:
            pos_sum = pos_sum + 1 - card_integrate(neg_sum, sum_dis_1)
        else:
            pos_sum = pos_sum + 1 - card_integrate(sum_dis_1, neg_sum)
    return pos_sum / pos_flag

def feature_selection(X, Y, parameter):
    max_columns = X.shape[1]
    dep_mean = 0
    dep_diff = parameter  
    depArray = np.zeros(max_columns, dtype=float)
    dep_flag = np.zeros(max_columns, dtype=int)
    redfeature = []
    now_row = X.shape[0]
    for i in range(max_columns):
        col = []
        for j in range(now_row):
            col.append((X[j][i])) 
        dep_single = dep_cor(col, Y, now_row)
        array_num = 0
        array_mean = 0
        for j in range(max_columns):
            if dep_flag[j] == 1:
                array_num += 1
                array_mean = array_mean + depArray[j]
        if array_num == 0:
            dep_mean = 0
        else:
            dep_mean = array_mean / array_num
        depArray[i] = dep_single 
        if dep_single >= dep_mean: 
            dep_flag[i] = 1
            flag_num = 0
            for j in range(max_columns):
                if dep_flag[j] == 1:
                    flag_num = flag_num + 1
            if flag_num <= 2: 
                continue
            for j in range(len(dep_flag)):
                if dep_flag[j] == 1 and j != i:
                    cols = np.zeros(now_row, dtype=float)
                    for k in range(now_row):
                        cols[k] = X[k][j]
                    dep_new = relation_two(col, cols, Y, now_row)
                    dep_Set = 0
                    cols_flag = 0
                    for k in range(len(dep_flag)):
                        if dep_flag[k] == 1 and k != i and k != j:
                            cols_flag += 1
                            colss = np.zeros(now_row, dtype=float)
                            for m in range(now_row):
                                colss[m] = X[m][k]
                            dep_Set = dep_Set + relation_two(cols, colss, Y, now_row)
                    dep_Set = dep_Set / cols_flag  
                    if dep_new < dep_Set or abs(dep_new - dep_Set) / dep_Set <= dep_diff:
                        continue
                    else:
                        if dep_cor(col, Y, now_row) > dep_cor(cols, Y, now_row):
                            dep_flag[j] = 0
                        else:
                            dep_flag[i] = 0
                            redfeature.append(i)  
                        break
            continue
        dep_flag[i] = 0
    selectedFeatures = []  
    for i in range(len(dep_flag)):
        if dep_flag[i] == 0:
            continue
        if len(redfeature) == 0:
            break
        before_col = []
        for j in range(now_row):
            before_col.append((X[j][i]))
        if dep_cor(before_col, Y, now_row) < dep_mean:
            for k in range(len(redfeature)):
                last_flag = 0
                after_col = []
                before_value = 0
                after_value = 0
                for j in range(now_row):
                    after_col.append((X[j][redfeature[k]]))  
                if dep_cor(after_col, Y, now_row) < dep_mean or dep_cor(after_col, Y, now_row) < dep_cor(before_col, Y,
                                                                                                         now_row):
                    continue
                for j in range(len(dep_flag)): 
                    if dep_flag[j] == 1 and j != i and j != k:
                        last_flag += 1
                        colss = np.zeros(now_row, dtype=float)
                        for m in range(now_row):
                            colss[m] = X[m][j]
                        before_value = before_value + relation_two(before_col, colss, Y, now_row)
                        after_value = after_value + relation_two(after_col, colss, Y, now_row)
                if last_flag == 0:
                    break
                before_value = before_value / last_flag
                after_value = after_value / last_flag 
                if after_value < before_value or abs(after_value - before_value) / before_value <= dep_diff:
                    dep_flag[i] = 0
                    dep_flag[redfeature[k]] = 1
                    redfeature.remove(redfeature[k])
                    break
    for i in range(len(dep_flag)):
        if dep_flag[i] == 1:
            selectedFeatures.append(i)
    return selectedFeatures

