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
    N = get_kind(Y)  # 计算有多少个特征标签
    answer = 0
    answer_flag = 0
    for i in range(N):
        now_x1 = []
        now_x2 = []
        now_x = 0
        now_y = 0
        now_y_flag = 0
        print(X1)
        print(X2)
        for k in range(relation_row):
            if Y[k] == i:
                now_x1.append(X1[k])
                now_x2.append(X2[k])
        now_x = distcorr(now_x1, now_x2)
        # print(now_x1)
        # print(now_x2)
        # print(now_x)
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
        print(now_y)
        answer = answer + card_integrate(now_x, now_y)
        answer_flag += 1
    return answer / answer_flag


def distcorr(X, Y):
    """
    Compute the distance correlation function
    a = [1,2,3,4,5]
    b = np.array([1,2,9,4,4])
    distcorr(a, b)
    0.762676242417
    """
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


def normalize(_x):  # 归一化函数
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
    # 权重和距离二者综合为相关性
    _cor_x = np.array(cor_x)
    dim_dis = _cor_x.ndim  # 看data是几维矩阵
    _D = cor_x.copy()
    cor_flag = get_kind(cor_y)  # 得到有多少个分类标签值
    pos_flag = 0  # 需要计算多少次
    pos_sum = 0  # 相关性的总和
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
        _d[cor_i] = math.sqrt(_d[cor_i])  # 得到了平均化以后的值
    _d.sort()  # 排序
    o = sorted(enumerate(cor_x), key=lambda density_x: density_x[1])  # 元组排序得到排序完之后的原来的位置
    l = []
    for vague_i in range(len(cor_x)):
        l.append(o[vague_i][0])  # 获得排序后的数组原始的坐标
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
        # 先算正面的相关性
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
            # sum_dis_1 = (sum_dis_1 / power) * (power_1 / power)
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
    # 目前得到了归一化后的值，相关性是有正反二方面来得出使用算数平均的方法来判断其相关性，因此必须知道在里面有多少标签的分类

    # 现在就得到了正反二方面的相关性  后面就是利用积分计算它们的大小

    # 最后得到的综合相关性
    return pos_sum / pos_flag


print("读取数据")
"""""
    进行特征选择
"""""


# 主函数
# parameter 默认为0.05
def feature_selection(X, Y, parameter):
    print(5)
    max_columns = X.shape[1]
    dep_mean = 0
    dep_diff = parameter  # 不同特征间的相关性，
    depArray = np.zeros(max_columns, dtype=float)
    dep_flag = np.zeros(max_columns, dtype=int)
    redfeature = []
    now_row = X.shape[0]
    print(2)
    for i in range(max_columns):

        print('进度' + str((i + 1) / max_columns * 100) + '%')
        print(dep_flag)

        col = []
        for j in range(now_row):
            col.append((X[j][i]))  # col为i列的所有行值
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
        depArray[i] = dep_single  # 记录下来

        print(dep_single)
        print(dep_mean)

        if dep_single >= dep_mean:  # Hugh correlation检测是否具有高相关性，比平均相关性要高
            dep_flag[i] = 1
            flag_num = 0
            for j in range(max_columns):
                if dep_flag[j] == 1:
                    flag_num = flag_num + 1
            if flag_num <= 2:  # 二个特征以下则不进行冗余性判断
                continue
            for j in range(len(dep_flag)):
                if dep_flag[j] == 1 and j != i:
                    cols = np.zeros(now_row, dtype=float)
                    for k in range(now_row):
                        cols[k] = X[k][j]
                    dep_new = relation_two(col, cols, Y, now_row)
                    # 计算其他特征与现计算特征的相关性，来判断现特征的相关性高低，如果低于该特征则认为此特征是最优特征之一，特征选择越多，选择新的最优特征就越难
                    dep_Set = 0
                    cols_flag = 0
                    for k in range(len(dep_flag)):
                        if dep_flag[k] == 1 and k != i and k != j:
                            cols_flag += 1
                            colss = np.zeros(now_row, dtype=float)
                            for m in range(now_row):
                                colss[m] = X[m][k]
                            dep_Set = dep_Set + relation_two(cols, colss, Y, now_row)
                    dep_Set = dep_Set / cols_flag  # 获得当前特征的平均总和相关性，如果当前特征的特征小于目前的的特征则，认为此特征与当前特征没有较强的相关性
                    print()
                    print(dep_new)
                    print(dep_Set)
                    if dep_new < dep_Set or abs(dep_new - dep_Set) / dep_Set <= dep_diff:
                        continue
                    else:
                        if dep_cor(col, Y, now_row) > dep_cor(cols, Y, now_row):
                            dep_flag[j] = 0
                        else:
                            dep_flag[i] = 0
                            redfeature.append(i)  # 比现在的挑选的特征相关性要差，但从全局来看，此特征并不是无用特征，只是弱冗余特征
                        break
            continue
        dep_flag[i] = 0
        # 比平均特征值的相关性要高，但是低于综合相关性，加入弱冗余特征里面

    # 最后再和弱冗余特征进行筛选,将备选的特征与现有特征比较，如果现有的特征相关性更高，则替换原先的特征

    selectedFeatures = []  # 最优特征集合
    for i in range(len(dep_flag)):
        if dep_flag[i] == 0:
            continue
        if len(redfeature) == 0:
            break
        before_col = []
        for j in range(now_row):
            before_col.append((X[j][i]))  # col为i列的所有行值
        if dep_cor(before_col, Y, now_row) < dep_mean:
            for k in range(len(redfeature)):
                last_flag = 0
                after_col = []
                before_value = 0
                after_value = 0
                for j in range(now_row):
                    after_col.append((X[j][redfeature[k]]))  # col为i列的所有行值
                if dep_cor(after_col, Y, now_row) < dep_mean or dep_cor(after_col, Y, now_row) < dep_cor(before_col, Y,
                                                                                                         now_row):
                    continue
                for j in range(len(dep_flag)):  # 依次与所有特征进行综合相关性筛查
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
                after_value = after_value / last_flag  # 获得当前特征的平均总和相关性，如果当前特征的特征小于目前的的特征则，认为此特征与当前特征没有较强的相关性
                if after_value < before_value or abs(after_value - before_value) / before_value <= dep_diff:
                    dep_flag[i] = 0
                    dep_flag[redfeature[k]] = 1
                    redfeature.remove(redfeature[k])
                    break
    # print(dep_flag)
    print(dep_flag)
    for i in range(len(dep_flag)):
        if dep_flag[i] == 1:
            selectedFeatures.append(i)
    # print(selectedFeatures)
    return selectedFeatures


"""
sklearn.cross_validation.cross_val_score(estimator, X, y=None, scoring=None,
cv=None, n_jobs=1, verbose=0, fit_params=None, pre_dispatch=‘2*n_jobs’)

estimator:估计方法对象(分类器)
X：数据特征(Features)
y：数据标签(Labels)
soring：调用方法(包括accuracy和mean_squared_error等等)
cv：几折交叉验证
n_jobs：同时工作的cpu个数（-1代表全部）
"""

# 2022/8/20/18:30
