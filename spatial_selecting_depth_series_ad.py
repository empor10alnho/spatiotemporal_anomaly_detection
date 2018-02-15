# coding:utf-8
# coded : 2016/11/11



import ncUtility
import numpy as np
#import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
# %matplotlib inline
#import matplotlib.dates as mdates
#from matplotlib import cm
import netCDF4
import os
import datetime
import dateutil.parser as parser
import time
import sys
#sys.setrecursionlimit(10000)
#import random
import scipy.spatial.distance
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
#from scipy import interpolate
#from numba import jit
#from multiprocessing import Pool
#import multiprocessing
#from mpl_toolkits.basemap import Basemap
from sklearn.metrics import roc_curve, auc
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import NearestNeighbors
import json

def nanargsort(x):
    '''
    nanargsort under 3-dimensional array

    memo:
    x,y = np.mgrid[0:3:1,0:2:1]
    In [23]: np.c_[x.ravel(),y.ravel()].reshape(3,2,2)[2,1]
    Out[23]: array([2, 1])
    '''
    d = len(x.shape)
    if d == 1:
        indices = np.mgrid[0:x.shape[0]:1]
    elif d == 2:
        i,j = np.mgrid[0:x.shape[0]:1,0:x.shape[1]:1]
        indices = np.c_[i.ravel(),j.ravel()]
    elif d == 3:
        i,j,k = np.mgrid[0:x.shape[0]:1,0:x.shape[1]:1,0:x.shape[2]:1]
        indices = np.c_[i.ravel(),j.ravel(),k.ravel()]
    non_sorted_indices = x[~np.isnan(x)].argsort()
    return indices[(~np.isnan(x)).ravel()][non_sorted_indices]

def make_sliding_windows(interp_press, interp_values, window_width=5):
    '''
    input  : list of interp_press, list of interp_values, windwo_width
    output : list of sliding window press, list of sliding windows
    '''
    window_press = []
    sliding_windows = []
    for i in range(len(interp_press)):
        window_press.append(interp_pres[:-window_width+1])
        sliding_windows.append(np.array([interp_value[j:j+window_width] for j in xrange(len(interp_pres)-window_width+1)]))
    return window_press, sliding_windows

def make_sliding_window(value, window_width=4):
    '''
    return np.array(sliding_window_depth starting with depth[0] (depth[:-window_width+1])), np.array(sliding_window)
    '''
    return np.array([value[j:j+window_width] for j in xrange(len(value)-window_width+1)])

def make_sliding_window_array(depth, value_array, window_width=4):
    value_array.shape = (value_array.shape[0], value_array.shape[1], 1)
    if window_width > 1:
        window_array = np.c_[value_array[:,0:-window_width+1],value_array[:,1:-window_width+2]]
        for i in range(2,window_width):
            window_array = np.c_[window_array, value_array[:, i:len(depth)-window_width+1+i]]
        window_array[np.isnan(window_array).any(axis=2)] = np.nan
        return depth[:-window_width+1], window_array
    elif window_width == 1:
        return depth, value_array
    '''普通に１つずつwindowをとるやり方：ミカン
    window_array = np.full((value_array.shape[0],value_array.shape[1]-window_width+1,window_width),np.nan)
    if window_width > 1:
        for i in range(len(value_array)):
            for j in range(len(depth)-window_width+1):
                window_array[i,j] = value_array[i,j:j+window_width]
        return depth[:-window_width+1], window_array
    elif window_width == 1:
        return depth[:-window_width+1], window_array[:,:,None]
    '''

def value2interp(depths, values, target_depth):
    # interpolation
    training_interp_depths, training_interp_values = [], []
    for depth,value in zip(depths,values):
        training_interp_depths.append(target_depth[(depth.min()<=target_depth) & (target_depth<=depth.max())])
        training_interp_values.append(np.interp(training_interp_depths[-1], depth, value))
    return training_interp_depths, training_interp_values

def values2interpolated_array(depths, values, target_depth):
    # interpolate values to an interpolated array
    interp_array = np.empty((len(values),len(target_depth)))
    interp_array[:] = np.nan
    for i in range(len(depths)):
        interp_array[i][(depths[i].min()<=target_depth) & (target_depth<=depths[i].max())] = np.interp(target_depth[(depths[i].min()<=target_depth) & (target_depth<=depths[i].max())], depths[i], values[i])
    return interp_array


def make_test_windows(depths, values, qcs, target_depth, window_width):
    '''
    input : list of (depths, values, qcs(with'1''2''3''4'labels)), target_depth to interpolate, window_width
    what is done : interpolation and sliding window transformation
    output : list of (depth, values, qcs)
    '''
    window_depths, window_values, window_qcs = [], [], []
    for i in xrange(len(depths)):
        # interpolation
        interp_depth = target_depth[(depths[i][0]<=target_depth) & (target_depth<=depths[i][-1])]
        interp_qc = np.ones_like(interp_depth, dtype = 'S1') # とりあえず'1'を埋める
        interp_value = np.interp(interp_depth, depths[i], values[i])
        raw_len = len(depths[i])
        # labeling for interp_qc
        # ->①逐一ラベル3４前後は全て４ ②(1,4)と(4,1)の間を全て４　③(1,4)の前４、次がラベル１であるとき４，④1と4の中間で1と4をつける
        if (qcs[i][0] == '4'): # or (qc[0] == '3') # initialization
            interp_qc[(depths[i][0] <= interp_depth) & (interp_depth < depths[i][1])] = '4'
        for j in xrange(1, raw_len-1):
            if (qcs[i][j] == '4'): # or (qc[j] == '3')
                interp_qc[(depths[i][j-1] < interp_depth) & (interp_depth < depths[i][j+1])] = '4'
        if (qcs[i][-1] == '4'): # or (qc[-1] == '3') # terminating
            interp_qc[(depths[i][-2] < interp_depth) & (interp_depth <= depths[i][-1])] = '4'

        # sliding window
        if len(interp_depth) < window_width: # window_widthが補間深度よりも幅が広くスライド窓を形成できない場合
            window_depths.append(np.nan)
            window_values.append(np.nan)
            window_qcs.append(np.nan)
        if not window_width == 1:
            window_depth = interp_depth[:-window_width+1]
        else:
            window_depth = interp_depth
        window_value = make_sliding_window(interp_value, window_width)
        window_qc0 = make_sliding_window(interp_qc, window_width) # np.array([('1','1','1','1'), (...),('1','1','1','1')]) shape = (raw_len-window_width+1, window_width)
        window_qc = np.array(['4' if (window=='4').any() else '1' for window in window_qc0])
            #確認用 interp_qc, window_depth, window_value, window_qc
        """#もう一つの window_qcのラベリング
        window_qc = np.ones_like(window_depth, dtype = 'S1') # np.array([('1','1','1','1', ...,'1','1','1','1')])
        for i in xrange(window_depth.shape[0]):
            #print window_qc0[i]
            if (window_qc0[i]=='4').any():
                window_qc[i] = '4'
        """
        window_depths.append(window_depth)
        window_values.append(window_value)
        window_qcs.append(window_qc)
    return window_depths, window_values, window_qcs

def make_test_interp(depths, values, qcs, target_depth):
    '''
    '''
    depth_len = len(target_depth)
    data_n = len(values)
    #window_array = np.empty((data_n,depth_len-window_width+1,window_width)) * np.nan
    test_interps = np.full((data_n,depth_len), np.nan)
    test_interp_qcs = np.full((data_n,depth_len), np.nan)
    test_interp_depth = target_depth
    for i in xrange(data_n):
        # interpolation
        depth_mask = (depths[i][0]<=target_depth) & (target_depth<=depths[i][-1])
        interp_depth = target_depth[depth_mask]
        interp_qc = np.zeros_like(interp_depth) # とりあえずlabel=='1'として0を埋める
        interp_value = np.interp(interp_depth, depths[i], values[i])
        interp_len = len(interp_depth)
        for j in range(interp_len):
            if (qcs[i][depths[i]<=interp_depth[j]][-1] == '4') or (qcs[i][interp_depth[j]<=depths[i]][0] =='4'):
                interp_qc[j] = 1
        test_interps[i,depth_mask] = interp_value
        test_interp_qcs[i,depth_mask] = interp_qc
    return test_interp_depth, test_interps, test_interp_qcs


def make_test_window_array(depths, values, qcs, target_depth, window_width):
    '''こっちが正規？
    input : list of (depths, values, qcs(with'1''2''3''4'labels)), target_depth to interpolate, window_width
    what is done : interpolation and sliding window transformation
    output : array of (depth, values, qcs (not '1','4', but 0, 1)), whose shape is (len(values), len(target_depth), window_width))
    '''
    depth_len = len(target_depth)
    data_n = len(values)
    window_depth = target_depth[:depth_len-window_width+1]
    #window_array = np.empty((data_n,depth_len-window_width+1,window_width)) * np.nan
    window_array = np.full((data_n,depth_len-window_width+1,window_width), np.nan)
    window_qc_array = np.full((data_n,depth_len-window_width+1,window_width), np.nan)
    qc_array = np.full((data_n,depth_len-window_width+1), np.nan)
    for i in xrange(data_n):
        # interpolation
        depth_mask = (depths[i][0]<=target_depth) & (target_depth<=depths[i][-1])
        interp_depth = target_depth[depth_mask]
        interp_qc = np.zeros_like(interp_depth) # とりあえずlabel=='1'として0を埋める
        interp_value = np.interp(interp_depth, depths[i], values[i])
        interp_len = len(interp_depth)
        for j in range(interp_len):
            if (qcs[i][depths[i]<=interp_depth[j]][-1] == '4') or (qcs[i][interp_depth[j]<=depths[i]][0] =='4'):
                interp_qc[j] = 1
        '''ここから作業'''
        # sliding window
        for j in range(window_width):
            window_array[i,np.where(depth_mask)[0][:interp_len+1-window_width],j] = interp_value[j:interp_len-window_width+1+j]
            window_qc_array[i,np.where(depth_mask)[0][:interp_len+1-window_width],j] = interp_qc[j:interp_len-window_width+1+j]
            #window_qc_array[i,np.where(depth_mask)[0][:1-window_width],j] = interp_qc[j:interp_len-window_width+1+j] これ多分間違い

            #window_array[i][depth_mask[:1-window_width]][:,j] = interp_value[j:interp_len-window_width+1+j]
            #window_qc_array[i][depth_mask[:1-window_width]][:,j] = interp_qc[j:interp_len-window_width+1+j]
            #window_qc_array[i][depth_mask[:1-window_width]][:,j:interp_len-window_width+1+j][:,0] = interp_qc[j:interp_len-window_width+1+j]

    #qc_array[np.isnan(window_qc_array).any(axi1s=2)] = np.nan
    qc_array[(window_qc_array==1).any(axis=2)] = 1
    qc_array[(window_qc_array==0).all(axis=2)] = 0
    return window_depth, window_array, qc_array

def make_test_window_array0(target_depth, values, qcs, window_width):
    '''これは作りかけ？
    input : list of (depths, values, qcs(with'1''2''3''4'labels)), target_depth to interpolate, window_width
    what is done : interpolation and sliding window transformation
    output : np.array of (depth, values, qcs), whose shape is (len(values), len(target_depth), window_width))
    '''

    depth = target_depth[:len(target_depth)-window_width+1]
    # interpolate values to an interpolated array
    interp_array = np.empty((len(values),len(target_depth),1))
    interp_array[:] = np.nan
    for i in range(len(depths)):
        interp_array[i][(depths[i].min()<=target_depth[i]) & (target_depth<=depths[i].max())] = np.interp(target_depth[(depths[i].min()<=target_depth) & (target_depth<=depths[i].max())], depths[i], values[i])

    value_array.shape = (value_array.shape[0], value_array.shape[1], 1)
    qc_array.shape = (value_array.shape[0], value_array.shape[1])
    if window_width > 1:
        window_array = np.c_[value_array[:,0:-window_width+1],value_array[:,1:-window_width+2]]
        for i in range(2,window_width):
            window_array = np.c_[window_array, value_array[:, i:len(depth)-window_width+1+i]]
        window_array[np.isnan(window_array).any(axis=2)] = np.nan
        return depth[:-window_width+1], window_array

    window_depth = target_depth[:len(target_depth)-window_width+1]
    window_array = np.full((len(values),len(target_depth)-window_width+1,window_width), np.nan)
    window_qc_array = np.full((len(values),len(target_depth)-window_width+1,window_width), np.nan)
    qc_array = np.full((len(values),len(target_depth)), np.nan)
    window_depths, window_values, window_qcs = [], [], []
    for i in xrange(len(depths)):
        # interpolation
        depth_mask = (depths[i][0]<=target_depth) & (target_depth<=depths[i][-1])
        interp_depth = target_depth[depth_mask]
        interp_qc = np.ones_like(interp_depth, dtype = 'S1') # とりあえず'1'を埋める
        interp_value = np.interp(interp_depth, depths[i], values[i])
        raw_len = len(depths[i])
        for j in range(raw_len):
            if (qcs[i][depths[i]<=interp_depth[j]][-1] == '4') or (qcs[i][interp_depth[j]<=depths[i]][0] =='4'):
                interp_qc[j] = '4'

        # sliding window
        for j in range(window_width):
            window_array[i][:,j] = interp_value[j:raw_len-window_width+1+j]
            window_qc_array[i][:,j] = interp_qc[j:raw_len-window_width+1+j]
    qc_array[(window_qc_array=='4').any(axis=2)] = '4'
    qc_array[(window_qc_array=='1').all(axis=2)] = '1'
    #qc_array = np.where((window_qc_array=='4').any(axis=2), '4', '1')

    for i in xrange(len(depths)):
        # interpolation
        depth_mask = (depths[i][0]<=target_depth) & (target_depth<=depths[i][-1])
        interp_depth = target_depth[depth_mask]
        interp_qc = np.ones_like(interp_depth, dtype = 'S1') # とりあえず'1'を埋める
        interp_value = np.interp(interp_depth, depths[i], values[i])
        raw_len = len(depths[i])
        # labeling for interp_qc
        # ->①逐一ラベル3４前後は全て４ ②(1,4)と(4,1)の間を全て４　③(1,4)の前４、次がラベル１であるとき４，④1と4の中間で1と4をつける
        if (qcs[i][0] == '4'): # or (qc[0] == '3') # initialization
            interp_qc[(depths[i][0] <= interp_depth) & (interp_depth < depths[i][1])] = '4'
        for j in xrange(1, raw_len-1):
            if (qcs[i][j] == '4'): # or (qc[j] == '3')
                interp_qc[(depths[i][j-1] < interp_depth) & (interp_depth < depths[i][j+1])] = '4'
        if (qcs[i][-1] == '4'): # or (qc[-1] == '3') # terminating
            interp_qc[(depths[i][-2] < interp_depth) & (interp_depth <= depths[i][-1])] = '4'

        # sliding window
        if len(interp_depth) < window_width: # window_widthが補間深度よりも幅が広くスライド窓を形成できない場合
            window_depths.append(np.nan)
            window_values.append(np.nan)
            window_qcs.append(np.nan)
        if not window_width == 1:
            window_depth = interp_depth[:-window_width+1]
        else:
            window_depth = interp_depth
        window_value = make_sliding_window(interp_value, window_width)
        window_qc0 = make_sliding_window(interp_qc, window_width) # np.array([('1','1','1','1'), (...),('1','1','1','1')]) shape = (raw_len-window_width+1, window_width)
        window_qc = np.array(['4' if (window=='4').any() else '1' for window in window_qc0])
            #確認用 interp_qc, window_depth, window_value, window_qc
        """#もう一つの window_qcのラベリング
        window_qc = np.ones_like(window_depth, dtype = 'S1') # np.array([('1','1','1','1', ...,'1','1','1','1')])
        for i in xrange(window_depth.shape[0]):
            #print window_qc0[i]
            if (window_qc0[i]=='4').any():
                window_qc[i] = '4'
        """
        window_array[i][depth_mask] = window_value
        qc_array[i][depth_mask] = window_qc
    return window_depth, window_array, qc_array

def sort_along_depth(depths, windows, target_depth):
    '''
    sort the training windows into sets aligned along target_depth
    input : a list of depths, alist of windows, target_depth
    output : np.array of window set aligned along depth
    '''
    depth_sorted_training_set = [[] for i in range(len(target_depth))]
    for i in xrange(len(depths)):
        for j in xrange(len(depths[i])):
            depth_sorted_training_set[np.where(target_depth==depths[i][j])[0]].append(windows[i][j])
    return np.array(depth_sorted_training_set)

algorithm = '''
'''

memo = '''
・ラベルは1と4のみ
・階層的クラスタリングのthresholdの決め方，最大併合距離の1/4 or 併合距離の4分位点
・クラスターの重心計算（単純に緯度経度の平均）
・各層でトレーニングデータが異なるから，異常度をまとめてAUCを出して良いのか？各層別々ですべき？
・'all'以外の時，test_window_qcsとanomaly_scoresとtest_windowのnp.isnan()は一致しない
・
'''

argvs = sys.argv
"""=================================================== setting 1/2 =============================================================================="""
if len(argvs) == 1:
    setting_file = {
    # 手法
    'ad_method' : 'SpatialWeightedKNN', # 'OCSVM', 'k-thNN', 'IsolationForest', 'Mahalanobis', 'LOF', 'SpatialWeightedKNN'
    # 手法パラメータ
    #'ad_method_parameters' : {'nu':0.1, 'gamma':0.1, 'k':10, 'outliers_fraction':0.1, 'n_estimators':100, 'max_samples':'auto', 'influence_radius':1000.}, # wide : (lati_half_length, lon_half_length), # complete, single, average
    'OCSVM_params' : {'nu':0.1, 'gamma':0.1},
    'k-thNN_params' : {'k':1},
    'IsolationForest_params' : {'n_estimators':100, 'max_samples':'auto', 'outliers_fraction':0.1},
    'Mahalanobis_params' : {'all_is' : 'fixed'},
    'LOF_params' : {'k':10, 'contamination':0.1}, # LOFはcontaminationを0にできないので小さい値を設定する
    'SpatialWeightedKNN_params' : {'k' : 'all', 'influence_radius':1000.}, # 'k' : 'all' to use all training data

    # ex2 設定
    'spatial_selecting_method' : 'rectangle', # 'hierarchical_clustering1', 'hierarchical_clustering2', 'rectangle', 'all'
    'selecting_method_parameters' : {'distance_measure':'complete', 'rectangle_range':(4,8), 'least_training_n':500, 'max_distance':1000, 'q_quantile':4, 'wide_range_is':'(10,20)', 'narrow_range_is':'(4,8)'}, # 'wide' and 'narrow' are memo, complete, single, average，rectangle_rangee : (緯度，経度)

    # ex1設定
    'narrow_range_experiment' : False,
    'longitude_range_thisEX'  : (175,195),
    'latitude_range_thisEX'  : (35,45),
    'feature_selection'  : 'with_gradient', # 'with_gradient' : (value, value_gradient), 'raw' : (value), 'gradient' : (gradient)
    'window_width' : 10,

    # 保存ディレクトリ
    'result_dir'  : './result_spatial/ex3_test',

    # miscellaneous
    'training_paths' : './../data/spatial_selecting_depth_series_ad/training_paths_10000.npy',
    'test_paths1'  : './../data/spatial_selecting_depth_series_ad/test_paths_1_2000.npy',
    'test_paths4'  : './../data/spatial_selecting_depth_series_ad/test_paths_4_2000.npy',
    'longitude_range'  : (140,220),
    'latitude_range'  : (10,50),
    'interp_range' : (0,2000),
    'interp_width' : 5.0,
    'do_psal' : True,
    'do_temp' : False,
    'profiles_distance_measure' : 'l2', # 'l1', 'l2'
    'partitioning_depths' : [0, 250, 650, 2000],

    'AA__memo__AA' : 'ex3_test'
    }
    setting_file_name = 'setting_file'
    """================================================================================================================="""
else:
    f = open(sys.argv[1], 'r')
    setting_file = json.load(f)
    f.close()
    setting_file_name = sys.argv[1]
# read files
target_depth = np.arange(5., 2000. + setting_file['interp_width']/2., setting_file['interp_width'])
training_paths = np.load(setting_file['training_paths'])
test_paths1 = np.load(setting_file['test_paths1'])
test_paths4 = np.load(setting_file['test_paths4'])
test_paths = list(test_paths1) + list(test_paths4)


"""================================================== PreProcessing =================================================="""
print '0[min] Implementing ' + setting_file_name
# time
times = [time.time()]
time_interval = []
times.append(time.time())

ncUtil = ncUtility.NcUtil()

""" reading and preprocessing data """
"""tr_tpress0, tr_temps0, tr_ppress0, tr_psals0 = ncUtil.paths22vec(setting_file['training_paths'] , which_pres='_ADJUSTED', which_pres_qc='_ADJUSTED'\
            , pres_qc_choice=(True,False,False,False), which_atr='_ADJUSTED', which_atr_qc='_ADJUSTED', vec_shape=['1']\
            , erase_0shape=False)"""
training_tpress0, trainingraining_temps0, training_tqc0, training_ppress0, training_psals0, training_pqc0 = ncUtil.paths2vecnlabel(training_paths, which_pres=''\
    , which_pres_qc='_ADJUSTED', pres_qc_choice='1', which_atr='', which_atr_qc='_ADJUSTED', qc_choice='1', erase_0shape=False)
test_tpress0, t_temps0, test_tqc0, test_ppress0, test_psals0, test_pqc0 = ncUtil.paths2vecnlabel(test_paths, which_pres=''\
    , which_pres_qc='_ADJUSTED', pres_qc_choice='1', which_atr='', which_atr_qc='_ADJUSTED', qc_choice='14', erase_0shape=False)

# longitude, latitudeとる
training_lats, training_lons = [], []
for path in training_paths:
    nc = netCDF4.Dataset(path, 'r')
    training_lats.append(nc.variables['LATITUDE'][0])
    training_lons.append(nc.variables['LONGITUDE'][0] if nc.variables['LONGITUDE'][0] > 0 else nc.variables['LONGITUDE'][0]+360)
    nc.close()
test_lats, test_lons = [], []
for path in test_paths:
    nc = netCDF4.Dataset(path, 'r')
    test_lats.append(nc.variables['LATITUDE'][0])
    test_lons.append(nc.variables['LONGITUDE'][0] if nc.variables['LONGITUDE'][0] > 0 else nc.variables['LONGITUDE'][0]+360)
    nc.close()
# np.arrayに変換
training_tpress0, trainingraining_temps0, training_tqc0, training_ppress0, training_psals0, training_pqc0 = np.array(training_tpress0), np.array(trainingraining_temps0), np.array(training_tqc0), np.array(training_ppress0), np.array(training_psals0), np.array(training_pqc0)
test_tpress0, t_temps0, test_tqc0, test_ppress0, test_psals0, test_pqc0 = np.array(test_tpress0), np.array(t_temps0), np.array(test_tqc0), np.array(test_ppress0), np.array(test_psals0), np.array(test_pqc0)
training_lats, training_lons, test_lats, test_lons = np.array(training_lats), np.array(training_lons), np.array(test_lats), np.array(test_lons)

# 緯度経度区切って小規模実験用
if setting_file['narrow_range_experiment']:
    tr_lons_mask = (setting_file['longitude_range_thisEX'][0] <= training_lons) & (training_lons <= setting_file['longitude_range_thisEX'][1])
    tr_lats_mask = (setting_file['latitude_range_thisEX'][0] <= training_lats) & (training_lats <= setting_file['latitude_range_thisEX'][1])
    tr_laonts_mask = tr_lons_mask & tr_lats_mask
    te_lons_mask = (setting_file['longitude_range_thisEX'][0] <= test_lons) & (test_lons <= setting_file['longitude_range_thisEX'][1])
    te_lats_mask = (setting_file['latitude_range_thisEX'][0] <= test_lats) & (test_lats <= setting_file['latitude_range_thisEX'][1])
    te_laonts_mask = te_lons_mask & te_lats_mask

    training_lats, training_lons, test_lats, test_lons = training_lats[tr_laonts_mask], training_lons[tr_laonts_mask], test_lats[te_laonts_mask], test_lons[te_laonts_mask]
    training_ppress0, training_psals0 = training_ppress0[tr_laonts_mask], training_psals0[tr_laonts_mask]
    test_ppress0, test_psals0, test_pqc0 = test_ppress0[te_laonts_mask], test_psals0[te_laonts_mask], test_pqc0[te_laonts_mask]

# trainingとtestのprofile数
training_n = len(training_ppress0)
test_n = len(test_ppress0)
# trainingとtestのwindow depthのlen
window_depth_len = len(target_depth) - setting_file['window_width'] + 1

# selection temp or psal
if setting_file['do_psal']:
    training_press0 = training_ppress0
    training_depths0 = ncUtil.press2depths(training_press0, training_lats)
    training_values0 = training_psals0
    test_press0 = test_ppress0
    test_depths0 = ncUtil.press2depths(test_press0, test_lats)
    test_values0 = test_psals0
    test_qcs0 = test_pqc0
elif setting_file['do_temp']:
    training_press0 = training_tpress0
    training_depths0 = ncUtil.press2depths(training_press0, training_lats)
    training_values0 = training_temps0
    test_press0 = test_tpress0
    test_depths0 = ncUtil.press2depths(test_press0, test_lats)
    test_values0 = test_temps0
    test_qcs0 = test_tqc0

# feature selection
# output:
#    training_window_depth whose length is len(target_depth) - setting_file['window_width'] + 1, training_windows
#    test_window_depth, test_windows, test_window_qcs
if setting_file['feature_selection']  == 'with_gradient': # array of (value, gradient)
    # preprocessing for training data
    # calculate gradients
    training_gradient_depths0, training_gradients0 = ncUtil.calculate_gradients(training_depths0, training_values0)
    # interpolation for value and gradient array
    training_interp_values = values2interpolated_array(training_depths0, training_values0, target_depth)
    training_interp_gradients = values2interpolated_array(training_gradient_depths0, training_gradients0, target_depth)
    # sliding window for value and gradient
    training_window_depth, training_windows0 = make_sliding_window_array(target_depth, training_interp_values, setting_file['window_width'])
    training_gradient_window_depth, training_gradient_windows = make_sliding_window_array(target_depth, training_interp_gradients, setting_file['window_width'])
    # combine
    training_windows = np.c_[training_windows0,training_gradient_windows]
    training_windows[np.isnan(training_windows).any(axis=2)] = np.nan # gradとvalueのどちらかがnp.nanのところのデータをnp.nan

    # preprocessing for test data
    # calculate gradients
    test_gradient_depths0, test_gradients0 = ncUtil.calculate_gradients(test_depths0, test_values0)
    test_gradient_qcs0 = ncUtil.gradient_labeling(test_qcs0)
    # sliding window through interpolation
    test_window_depth0, test_windows0, test_window_qcs0 = make_test_window_array(test_depths0, test_values0, test_qcs0, target_depth, setting_file['window_width'])
    test_gradient_window_depth, test_gradient_windows, test_gradient_window_qcs = make_test_window_array(test_gradient_depths0, test_gradients0, test_gradient_qcs0, target_depth=target_depth, window_width=setting_file['window_width'])
    # combine
    test_window_depth = test_gradient_window_depth # is also test_window_depth0
    test_windows = np.c_[test_windows0,test_gradient_windows]
    test_windows[np.isnan(test_windows).any(axis=2)] = np.nan # gradとvalueのどちらかがnp.nanのところのデータをnp.nan
    #test_window_qcs = np.where(((~np.isnan(test_window_qcs0))&(~np.isnan(test_gradient_window_qcs))) & ((test_window_qcs0==1)|(test_gradient_window_qcs)), 1, 0) # これやとあかん
    test_window_qcs = test_window_qcs0 + test_gradient_window_qcs
    test_window_qcs[test_window_qcs==2] = 1
    # 1.  vf = np.vectorize(lambda x: np.nan if np.isnan(x) else 1 if x >0 else 0)
    #     test_window_qcs = vf(test_window_qcs)
    # 2.  mask = ~np.isnan(test_window_qcs)
    #     mask[mask] &= test_window_qcs[mask] > 0
    #     test_window_qcs[mask] = 1
elif setting_file['feature_selection']  == 'raw': # array of value
    # preprocessing for training data
    # interpolation for value and gradient array
    training_interp_values = values2interpolated_array(training_depths0, training_values0, target_depth)
    # sliding window for value and gradient
    training_window_depth, training_windows = make_sliding_window_array(target_depth, training_interp_values, setting_file['window_width'])

    #sys.exit()

    # preprocessing for test data
    # sliding window through interpolation
    test_window_depth, test_windows, test_window_qcs = make_test_window_array(test_depths0, test_values0, test_qcs0, target_depth, setting_file['window_width'])
elif setting_file['feature_selection']  == 'gradient': # list of gradient_value
    # preprocessing for training data
    # calculate gradients
    training_gradient_depths0, training_gradients0 = ncUtil.calculate_gradients(training_depths0, training_values0)
    # interpolation for value and gradient array
    training_interp_gradients = values2interpolated_array(training_gradient_depths0, training_gradients0, target_depth)
    # sliding window for value and gradient
    training_window_depth, training_windows = make_sliding_window_array(target_depth, training_interp_gradients, setting_file['window_width'])

    # preprocessing for test data
    # calculate gradients
    test_gradient_depths0, test_gradients0 = ncUtil.calculate_gradients(test_depths0, test_values0)
    test_gradient_qcs0 = ncUtil.gradient_labeling(test_qcs0)
    # sliding window through interpolation
    test_window_depth, test_windows, test_window_qcs = make_test_window_array(test_gradient_depths0, test_gradients0, test_gradient_qcs0, target_depth=target_depth, window_width=setting_file['window_width'])

'''
from sklearn.neighbors import NearestNeighbors
samples = [[0, 0, 2], [1, 0, 0], [0, 0, 1]]
neigh = NearestNeighbors(2, 0.4)
neigh.fit(samples)
neigh.kneighbors([[0, 0, 1.3]], 2, return_distance=1)
    (array([[ 0.3,  0.7]]), array([[2, 0]]))


'''

# パラメータkを持つ手法の，データ最低数の設定 on k and 1(Z標準化)
if setting_file['ad_method'] == 'k-thNN':
    least_training_n = max(2, setting_file['k-thNN_params']['k'])
elif setting_file['ad_method'] == 'OCSVM':
    least_training_n = 2
elif setting_file['ad_method'] == 'IsolationForest':
    least_training_n = 2
elif setting_file['ad_method'] == 'Mahalanobis':
    least_training_n = 2
elif setting_file['ad_method'] == 'LOF':
    least_training_n = max(2, setting_file['LOF_params']['k'])
elif setting_file['ad_method'] == 'SpatialWeightedKNN':
    if setting_file['SpatialWeightedKNN_params']['k'] == 'all':
        least_training_n = 2
    else:
        least_training_n = max(2, setting_file['SpatialWeightedKNN_params']['k'])


# local_training_windowsにall dataを挿入して，データが1点しか無い層はZ標準化のためにnp.nanに置き換える．

# データ点数が１しかない層はnp.nanで埋める
for i in xrange(window_depth_len):
    if (~np.isnan(training_windows[:,i])).any(axis=1).sum() == 1:
        training_windows[:,i] = np.nan

#sys.exit()

# for pre learning before test for (one)
if setting_file['spatial_selecting_method'] == 'all':
    # z-normalization
    ori_test_windows = test_windows.copy() # 念のための保存，ori_test_windowsはZ標準化でnp.nanになったデータも含む
    local_training_means = np.nanmean(training_windows, axis=0) # これでnanが含まれないmean
    local_training_stds =  np.nanstd(training_windows, axis=0)
    local_training_windows = (training_windows - local_training_means) / local_training_stds
    test_windows = (test_windows - local_training_means) / local_training_stds
    # 消されたtest_windowsのためにqcも消す
    for i in xrange(window_depth_len):
        if np.isnan(local_training_means[i]).any():
            test_window_qcs[:,i] = np.nan
    local_training_mask = np.ones(training_n, dtype=bool)

    if setting_file['ad_method'] == 'k-thNN':
        clfs = [] # final len is len(target_depth)
        for i in xrange(window_depth_len):
            if (~np.isnan(local_training_windows[:,i])).all(axis=1).sum() < least_training_n: # ある深度インデックスiのデータ数がk以下の時
                clfs.append(np.nan)
            else:
                clfs.append(NearestNeighbors(setting_file['k-thNN_params']['k']))
                clfs[-1].fit(local_training_windows[:,i][~np.isnan(local_training_windows[:,i]).any(axis=1)])
    elif setting_file['ad_method'] == 'OCSVM':
        clfs = [] # final len is len(target_depth)
        for i in xrange(window_depth_len):
            if (~np.isnan(local_training_windows[:,i])).all(axis=1).sum() < least_training_n: # ある深度インデックスiのデータが全てnanのとき
                clfs.append(np.nan)
            else:
                clfs.append(svm.OneClassSVM(nu=setting_file['OCSVM_params']['nu'], kernel='rbf', gamma=setting_file['OCSVM_params']['gamma']))
                clfs[-1].fit(local_training_windows[:,i][~np.isnan(local_training_windows[:,i]).any(axis=1)])
                # -1 * decision_function(test_data)[0]) # returns np.array([anomaly score])
    elif setting_file['ad_method'] == 'Mahalanobis':  # decision_functionのmahalanobis distanceはちゃんとnormalizationされてるみたい
        clfs = [] # final len is len(target_depth)
        for i in xrange(window_depth_len):
            if (~np.isnan(local_training_windows[:,i])).all(axis=1).sum() < least_training_n: # ある深度インデックスiのデータが全てnanのときnp.nanをclfsに挿入
                clfs.append(np.nan)
            else:
                clfs.append(EllipticEnvelope(support_fraction=1., contamination=0.))
                    # EllipticEnvelope(support_fraction=1., contamination=0.261) # Empirical Covariance, support_fraction=setting_file['ad_method_parameters'] ['support_fraction']
                    # EllipticEnvelope(contamination=0.261) # Robust Covariance (Minimum Covariance Determinant)
                clfs[-1].fit(local_training_windows[:,i][~np.isnan(local_training_windows[:,i]).any(axis=1)])
                # -1 * decision_function(test_data)[0]) # returns np.array([anomaly score])
    elif setting_file['ad_method'] == 'IsolationForest':
        '''
        iFのパラメータセッティング
        ・max_samples = 'auto', min(256, n_samples), 256 (recommended in original paper)
        ・n_estimators = 100 (recommended in original paper)
        ・contamination = parameter
        '''
        clfs = [] # final len is len(target_depth)
        for i in xrange(window_depth_len):
            if (~np.isnan(local_training_windows[:,i])).all(axis=1).sum() < least_training_n: # ある深度インデックスiのデータが全てnanのとき
                clfs.append(np.nan)
            else:
                clfs.append(IsolationForest(n_estimators=setting_file['IsolationForest_params']['n_estimators'],max_samples=setting_file['IsolationForest_params']['max_samples'],contamination=setting_file['IsolationForest_params']['outliers_fraction']))
                clfs[-1].fit(local_training_windows[:,i][~np.isnan(local_training_windows[:,i]).any(axis=1)])
                # -1 * decision_function(test_data)[0]) # returns np.array([anomaly score])
    elif setting_file['ad_method'] == 'LOF':
        '''
        LOFはcontaminationを0にできないので小さい値を設定する
        '''
        clfs = [] # final len is len(target_depth)
        for i in xrange(window_depth_len):
            if (~np.isnan(local_training_windows[:,i])).all(axis=1).sum() < least_training_n: # ある深度インデックスiのデータ数がk以下の時
                clfs.append(np.nan)
            else:
                clfs.append(LocalOutlierFactor(n_neighbors=setting_file['LOF_params']['k'], contamination=setting_file['LOF_params']['contamination']))
                clfs[-1].fit(local_training_windows[:,i][~np.isnan(local_training_windows[:,i]).any(axis=1)])
                # -1 * decision_function(test_data)[0]) # returns np.array([anomaly score])
    elif setting_file['ad_method'] == 'SpatialWeightedKNN':
        clfs = [] # final len is len(target_depth)
        for i in xrange(window_depth_len):
            if (~np.isnan(local_training_windows[:,i])).all(axis=1).sum() < least_training_n: # ある深度インデックスiのデータ数がk以下の時
                clfs.append(np.nan)
            else:
                if setting_file['SpatialWeightedKNN_params']['k'] == 'all':
                    clfs.append(NearestNeighbors(len(local_training_windows[:,i][~np.isnan(local_training_windows[:,i]).any(axis=1)])))
                    clfs[-1].fit(local_training_windows[:,i][~np.isnan(local_training_windows[:,i]).any(axis=1)])
                else:
                    clfs.append(NearestNeighbors(setting_file['SpatialWeightedKNN_params']['k']))
                    clfs[-1].fit(local_training_windows[:,i][~np.isnan(local_training_windows[:,i]).any(axis=1)])

# time
times.append(time.time())
time_interval.append('{0:.1f}'.format((times[-1]-times[-2])/60.))

# sys.exit()

# clustering if selecting method is clustering, with training_windows before z-normalization
if setting_file['spatial_selecting_method'] in ('hierarchical_clustering1', 'hierarchical_clustering2'):
    import scipy.spatial.distance as distance
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    # training_window_depths, training_windowsをつかう
    profile_distance_matrix = np.zeros((training_n, training_n))
    # training_windows.shape = (training_n, len(target_depth), setting_file['window_width'])だから，windowにしてないものに変換する
    if setting_file['feature_selection']  == 'with_gradient':
        raw_clustering_data = np.c_[training_windows0[:,:,0], training_windows0[:,-1,1:]] # shape = (training_n, len(depth))
        gradient_clustering_data = np.c_[training_gradient_windows[:,:,0], training_gradient_windows[:,-1,1:]] # shape = (training_n, len(depth))
        clustering_data = np.c_[raw_clustering_data, gradient_clustering_data] # shape = (training_n, len(depth))
    elif setting_file['feature_selection'] in ('gradient','raw'):
        clustering_data = np.c_[training_windows[:,:,0], training_windows[:,-1,1:]] # shape = (training_n, len(depth))
    for i in xrange(training_n-1):
        for j in xrange(i+1, training_n):
            shared_depth_mask = (~np.isnan(clustering_data[i,:])) & (~np.isnan(clustering_data[j,:]))
            if shared_depth_mask.any():
                v1 = clustering_data[i,:][shared_depth_mask]
                v2 = clustering_data[j,:][shared_depth_mask]
                profile_distance_matrix[i,j] = profile_distance_matrix[j,i] = np.sqrt(np.dot(v1-v2,v1-v2))**(1./2) / len(v1)
            else: # 共通の深度が無い時 でもこれでクラスタリングできる？
                profile_distance_matrix[i,j] = profile_distance_matrix[j,i] = np.nan
    profile_distance_matrix[np.isnan(profile_distance_matrix)] = np.nanmax(profile_distance_matrix) + 1.0 # 距離定義できないとこを埋めるため．前処理
    Z = linkage(distance.squareform(profile_distance_matrix), method=setting_file['selecting_method_parameters']['distance_measure'])
    cluster_labels = fcluster(Z, Z[len(Z)/setting_file['selecting_method_parameters']['q_quantile'], 2], criterion='distance') # thresholdの決め方，最大併合距離の1/4 or 併合距離の4分位点
    if setting_file['spatial_selecting_method'] == 'hierarchical_clustering1':
        # クラスターの重心計算（単純に緯度経度の平均）
        cluster_centers = np.array([[training_lats[cluster_labels==i].mean(), training_lons[cluster_labels==i].mean()] for i in set(cluster_labels)]) # クラスタ1から[(lati, lon), (),... ()]
        clustered_lons = np.take(cluster_centers[:,1], cluster_labels-1)
        clustered_lats = np.take(cluster_centers[:,0], cluster_labels-1)
    #elif setting_file['spatial_selecting_method'] == 'hierarchical_clustering2':

# time
times.append(time.time())
time_interval.append('{0:.1f}'.format((times[-1]-times[-2])/60.))

'''メモ
x1 =
from matplotlib import cm
for i in range(1,max(fc)+1):
    plt.plot(x1[:,0][fc==i], x1[:,1][fc==i], 'o', c=cm.hsv(float(range(0,max(fc))[i-1]) / (max(fc)-1)))

'''
#sys.exit()

"""======================================================== TEST ==================================================================================="""
print '{0:.1f}[min] Testing '.format((times[-1]-times[0])/60.) + setting_file_name
#layer_anomaly_scores = []
anomaly_scores = np.full((test_n, window_depth_len), np.nan)
#profile_anomaly_scores = np.full(test_n, np.nan)
#local_training_windows_list = [] # 保存 for spatial clustering収集
local_training_masks = []
for i in xrange(test_n):
    # local data の選択
    # local_training_windows
    # local_training_distances = []
    # できればlocal_training_data_indices # len is test_n
    # local_training_lats = []
    # local_training_lons = []
    # 教師選択方法　setting_file['spatial_selecting_method'] = 'hierarchical_clustering1' # 'hierarchical_clustering1', 'rectangle', 'all'
    if setting_file['spatial_selecting_method'] == 'hierarchical_clustering1':
        '''
        select local training data
        ・testから10度以内にないクラスタは遠いから候補から消す
        ・クラスター重心をtestに近い順に並び替え #（この時longitudeは1/2倍するべき？）
        ・１クラスターずつ併合し，合計が500以上になるか，距離が1000km（約緯度10度の時の経度10度分の距離）をこすと併合終わり．
        使うデータ
        training_lats, training_lons, test_lats, test_lons
        cluster_centers # クラスタ番号順1~n
        cluster_labels # データindex順
        cluster_indices # クラスタのインデックス
        '''
        #latlong_mask = (np.abs(training_lats-test_lats[i])>10) & (np.abs(training_lons-test_lons[i])>10) # training１つ１つの緯度経度が10度以上は1000km以上だから取り除く
        latlong_mask = (np.abs(cluster_centers[:,0]-test_lats[i])<=10) & (np.abs(cluster_centers[:,1]-test_lons[i])<=10)
        masked_cluster_centers = cluster_centers[latlong_mask]
        masked_cluster_labels = np.arange(1,max(cluster_labels)+1)[latlong_mask]

        cluster_distances = np.array([ncUtil.calculate_distance((test_lats[i],test_lons[i]), masked_cluster_centers[j], scale='km') for j in xrange(len(masked_cluster_centers))])
        sorted_distances = np.sort(cluster_distances)
        sorted_distance_indices = np.argsort(cluster_distances)
        local_training_n = 0
        local_training_cluster_labels = []
        local_training_mask = np.zeros(training_n, dtype=bool)
        for j in xrange(len(cluster_distances)):
            # 終了条件
            if (local_training_n >= setting_file['selecting_method_parameters']['least_training_n']) or (sorted_distances[j] >= setting_file['selecting_method_parameters']['max_distance']): #(緯度経度10, 20)以内にデータ無い
                break
            # local training clusterの追加
            local_training_cluster_labels.append(masked_cluster_labels[sorted_distance_indices[j]])
            local_training_mask = local_training_mask | (cluster_labels==local_training_cluster_labels[-1])
            local_training_n += (cluster_labels==local_training_cluster_labels[-1]).sum()

        local_training_masks.append(local_training_mask)
        # １つも条件に当てはまらなければtest dataを飛ばす
        if local_training_n < least_training_n:
            continue
        # クラスター単位をprofile単位の深度，データ，距離に変換
        local_training_windows = training_windows[local_training_mask]
    elif setting_file['spatial_selecting_method'] == 'hierarchical_clustering2':
        '''
        ・spatial distances(test, training)が近いインスタンスのクラスタを併合
        '''
        spatial_distances = np.array([ncUtil.calculate_distance((test_lats[i],test_lons[i]), (training_lats[j], training_lons[j]), scale='km') for j in xrange(training_n)])
        sorted_distance_indices = np.argsort(spatial_distances)
        sorted_distances = spatial_distances[sorted_distance_indices]
        # 追加開始
        local_training_n = 0
        local_training_cluster_labels = []
        local_training_mask = np.zeros(training_n, dtype=bool)
        for j in xrange(training_n):
            # 終了条件
            if (local_training_n >= setting_file['selecting_method_parameters']['least_training_n']) or (sorted_distances[j] >= setting_file['selecting_method_parameters']['max_distance']): #(緯度経度10, 20)以内にデータ無い
                break
            if cluster_labels[sorted_distance_indices[j]] in local_training_cluster_labels:
                continue
            # local training clusterの追加
            local_training_cluster_labels.append(cluster_labels[sorted_distance_indices[j]])
            local_training_mask = local_training_mask | (cluster_labels==local_training_cluster_labels[-1])
            local_training_n += (cluster_labels==local_training_cluster_labels[-1]).sum()

        local_training_masks.append(local_training_mask)
        # １つも条件に当てはまらなければtest dataを飛ばす
        if local_training_n < least_training_n:
            continue
        local_training_windows = training_windows[local_training_mask]
    elif setting_file['spatial_selecting_method'] == 'rectangle':
        local_training_mask = (np.abs(training_lats-test_lats[i])<=(setting_file['selecting_method_parameters']['rectangle_range'][0]/2.0)) & (np.abs(training_lons-test_lons[i])<=(setting_file['selecting_method_parameters']['rectangle_range'][1]/2.0))
        local_training_masks.append(local_training_mask)
        if local_training_mask.sum() < least_training_n:
            continue
        local_training_windows = training_windows[local_training_mask]
    #elif setting_file['spatial_selecting_method'] is 'all': # 'all' のときは何もしない
        # training_windowsをlocal_training_windowsとする
        #if  setting_file['ad_method'] is 'SpatialWeightedKNN':

            #local_training_weights = local_training_distances/local_training_distances.sum() 規格化はkが決まってないと！！

    # local_training_windowsのz標準化：std=0で標準化しないように注意
    if not setting_file['spatial_selecting_method'] == 'all':
        local_training_means,local_training_stds = [],[]
        for j in xrange(window_depth_len):
            if (~np.isnan(local_training_windows[:,j])).any(axis=1).sum() < least_training_n: # データ数１or 0の場合
                local_training_means.append(np.nan)
                local_training_stds.append(np.nan)
                local_training_windows[:,j] = np.nan
            else:
                local_training_means.append(np.nanmean(local_training_windows[:,j])) # local_training_windows[:,j][~np.isnan(local_training_windows[:,j])].mean()
                local_training_stds.append(np.nanstd(local_training_windows[:,j])) # local_training_windows[:,j][~np.isnan(local_training_windows[:,j])].std()
                local_training_windows[:,j] = (local_training_windows[:,j]-local_training_means[-1]) / local_training_stds[-1]
        # local training dataの保存用の作成でもnp.saveで保存できないから放置
        # local_training_windows_list.append(local_training_windows)

    # error detection選択と実行（profileのanomaly scoreとlayerのanomaly scoreを出す）
    if setting_file['ad_method'] == 'OCSVM':
        # 全教師選択でないときは学習
        if setting_file['spatial_selecting_method'] == 'all':
            for j in xrange(window_depth_len): # 全深度
                if (not clfs[j] is np.nan) and (not np.isnan(test_windows[i,j]).any()): # trainingがあり，testがある時，clfsはnp.nan使えない
                    # trainingの数がk-thNNのk以下の場合等は最早考慮しない
                    anomaly_scores[i,j] = -1 * clfs[j].decision_function([test_windows[i,j]])[0,0]
        else:
            for j in xrange(window_depth_len):
                if (not np.isnan(local_training_means[j])) and (not np.isnan(test_windows[i,j]).any()):
                    clf = svm.OneClassSVM(nu=setting_file['OCSVM_params']['nu'], kernel='rbf', gamma=setting_file['OCSVM_params']['gamma'])
                    clf.fit(local_training_windows[:,j][~np.isnan(local_training_windows[:,j]).any(axis=1)])
                    anomaly_scores[i,j] = -1 * clf.decision_function([(test_windows[i,j]-local_training_means[j])/local_training_stds[j]])[0,0]
    elif setting_file['ad_method'] == 'k-thNN':
        '''# sklearn.NN使わない場合
        if setting_file['spatial_selecting_method'] is 'all':
            for j in xrange(window_depth_len):
                if (not np.isnan(training_windows[:,j]).all()) and (not np.isnan(test_windows[i,j]).any()):
                    anomaly_scores[i,j] = np.sort(np.sum(np.abs(training_windows[:,j][~np.isnan(training_windows[:,j])]-(test_windows[i,j]-local_training_means[j,:])/local_training_stds[j,:])**2,axis=-1)**(1./2))[setting_file['k-thNN_params']['k']-1]
        else:
            for j in xrange(window_depth_len):
                if (not np.isnan(training_windows[:,j]).all()) and (not np.isnan(test_windows[i,j]).any()):
                    anomaly_scores[i,j] = np.sort(np.sum(np.abs(local_training_windows[:,j][~np.isnan(local_training_windows[:,j])]-(test_windows[i,j]-local_training_means[j])/local_training_stds[j])**2,axis=-1)**(1./2))[setting_file['k-thNN_params']['k']-1]
                    '''
        if setting_file['spatial_selecting_method'] == 'all':
            for j in xrange(window_depth_len): # 全深度
                if (not clfs[j] is np.nan) and (not np.isnan(test_windows[i,j]).any()):
                    anomaly_scores[i,j] = clfs[j].kneighbors([test_windows[i,j]], setting_file['k-thNN_params']['k'], return_distance=1)[0][0,-1]
        else:
            for j in xrange(window_depth_len):
                if (not np.isnan(local_training_means[j])) and (not np.isnan(test_windows[i,j]).any()):
                    clf = NearestNeighbors(setting_file['k-thNN_params']['k'])
                    clf.fit(local_training_windows[:,j][~np.isnan(local_training_windows[:,j]).any(axis=1)])
                    anomaly_scores[i,j] = clf.kneighbors([(test_windows[i,j]-local_training_means[j])/local_training_stds[j]], setting_file['k-thNN_params']['k'], return_distance=1)[0][0,-1]
                    # neigh.keighbors returns (([d[x,k1], d[x,k2], ...]), (ik1, ik2, ...)) like (array([[ 0.3,  0.7]]), array([[2, 0]]))
    elif setting_file['ad_method'] == 'IsolationForest':
        if setting_file['spatial_selecting_method'] == 'all':
            for j in xrange(window_depth_len): # 全深度
                if (not clfs[j] is np.nan) and (not np.isnan(test_windows[i,j]).any()):
                    anomaly_scores[i,j] = -1 * clfs[j].decision_function([test_windows[i,j]])[0]
        else:
            for j in xrange(window_depth_len):
                if (not np.isnan(local_training_means[j])) and (not np.isnan(test_windows[i,j]).any()):
                    clf = IsolationForest(n_estimators=setting_file['IsolationForest_params']['n_estimators'],max_samples=setting_file['IsolationForest_params']['max_samples'],contamination=setting_file['IsolationForest_params']['outliers_fraction'])
                    clf.fit(local_training_windows[:,j][~np.isnan(local_training_windows[:,j]).any(axis=1)])
                    anomaly_scores[i,j] = -1 * clf.decision_function([(test_windows[i,j]-local_training_means[j])/local_training_stds[j]])[0]
    elif setting_file['ad_method'] == 'Mahalanobis':
        if setting_file['spatial_selecting_method'] == 'all':
            for j in xrange(window_depth_len): # 全深度
                if (not clfs[j] is np.nan) and (not np.isnan(test_windows[i,j]).any()):
                    anomaly_scores[i,j] = -1 * clfs[j].decision_function([test_windows[i,j]], raw_values=True)[0]
        else:
            for j in xrange(window_depth_len):
                if (not np.isnan(local_training_means[j])) and (not np.isnan(test_windows[i,j]).any()):
                    clf = EllipticEnvelope(support_fraction=1., contamination=0.)
                        # EllipticEnvelope(support_fraction=1., contamination=0.261) # Empirical Covariance, support_fraction=setting_file['ad_method_parameters'] ['support_fraction']
                        # EllipticEnvelope(contamination=0.261) # Robust Covariance (Minimum Covariance Determinant)
                    clf.fit(local_training_windows[:,j][~np.isnan(local_training_windows[:,j]).any(axis=1)])
                    anomaly_scores[i,j] = -1 * clf.decision_function([(test_windows[i,j]-local_training_means[j])/local_training_stds[j]], raw_values=True)[0]
    elif setting_file['ad_method'] == 'LOF': # contamination 0はだめ
        if setting_file['spatial_selecting_method'] == 'all':
            for j in xrange(window_depth_len): # 全深度
                if (not clfs[j] is np.nan) and (not np.isnan(test_windows[i,j]).any()): # trainingがあり，testがある時
                    # trainingの数がk-thNNのk以下の場合等は最早考慮しない
                    anomaly_scores[i,j] = -1 * clfs[j]._decision_function([test_windows[i,j]])[0]
        else:
            for j in xrange(window_depth_len):
                if (not np.isnan(local_training_means[j])) and (not np.isnan(test_windows[i,j]).any()):
                    clf = LocalOutlierFactor(n_neighbors=setting_file['LOF_params']['k'], contamination=setting_file['LOF_params']['contamination'])
                    clf.fit(local_training_windows[:,j][(~np.isnan(local_training_windows[:,j])).all(axis=1)])
                    anomaly_scores[i,j] = -1 * clf._decision_function([(test_windows[i,j]-local_training_means[j])/local_training_stds[j]])[0]
    elif setting_file['ad_method'] == 'SpatialWeightedKNN':
        # 距離重み計算
        local_training_distances = np.array([ncUtil.calculate_distance((test_lats[i],test_lons[i]), (lat,lon), scale='km') for lat,lon in zip(training_lats[local_training_mask],training_lons[local_training_mask])])
        local_training_distances = np.exp(-local_training_distances/setting_file['SpatialWeightedKNN_params']['influence_radius']) # 距離→RBF距離重み
        # 異常度計算
        if setting_file['spatial_selecting_method'] == 'all':
            '''
            ・anomaly score = Σ_k 正規化されたガウス距離 × k-thNNの距離
            '''
            for j in xrange(window_depth_len):
                if (not clfs[j] is np.nan) and (not np.isnan(test_windows[i,j]).any()):
                    if setting_file['SpatialWeightedKNN_params']['k'] == 'all':
                        value_distances2k, indices2k = clfs[j].kneighbors([test_windows[i,j]], len(local_training_windows[:,j][~np.isnan(local_training_windows[:,j]).any(axis=1)]), return_distance=1)#[0,-1]
                    else:
                        value_distances2k, indices2k = clfs[j].kneighbors([test_windows[i,j]], setting_file['SpatialWeightedKNN_params']['k'], return_distance=1)#[0,-1]
                    local_training_weights = local_training_distances[indices2k[0]] / local_training_distances[indices2k[0]].sum() # 重みの規格化
                    anomaly_scores[i,j] = (value_distances2k[0] * local_training_weights).sum()
        else:
            for j in xrange(window_depth_len):
                if (not np.isnan(local_training_means[j])) and (not np.isnan(test_windows[i,j]).any()):
                    """sklearn.KNN使わない
                    value_distances = np.sum(np.abs(local_training_windows[:,j][~np.isnan(local_training_windows[:,j])]-(test_windows[i,j]-local_training_means[j,:])/local_training_stds[j,:])**2,axis=-1)**(1./2) # testとtrainigの距離
                    sorted_indices = np.argsort(value_distances) # 空間距離順に並び替えするindices
                    local_training_weights = local_training_distances[indices2k[0]][:setting_file['SpatialWeightedKNN_params']['k']]/local_training_distances[sorted_indices][:setting_file['SpatialWeightedKNN_params']['k']].sum() # 重みの規格化
                    anomaly_scores[i,j] = (value_distances[indices2k[0]][:setting_file['SpatialWeightedKNN_params']['k']] * local_training_weights).sum()
                    """
                    if setting_file['SpatialWeightedKNN_params']['k'] == 'all':
                        clf = NearestNeighbors(len(local_training_windows[:,j][~np.isnan(local_training_windows[:,j]).all(axis=1)]))
                        clf.fit(local_training_windows[:,j][~np.isnan(local_training_windows[:,j]).all(axis=1)])
                        value_distances2k, indices2k = clf.kneighbors([(test_windows[i,j]-local_training_means[j])/local_training_stds[j]], len(local_training_windows[:,j][~np.isnan(local_training_windows[:,j]).all(axis=1)]), return_distance=1)#[0,-1]
                    else:
                        clf = NearestNeighbors(setting_file['SpatialWeightedKNN_params']['k'])
                        clf.fit(local_training_windows[:,j][~np.isnan(local_training_windows[:,j]).all(axis=1)])
                        value_distances2k, indices2k = clf.kneighbors([(test_windows[i,j]-local_training_means[j])/local_training_stds[j]], setting_file['SpatialWeightedKNN_params']['k'], return_distance=1)#[0,-1]
                    local_training_weights = local_training_distances[indices2k[0]] / local_training_distances[indices2k[0]].sum() # 重みの規格化
                    anomaly_scores[i,j] = (local_training_weights * value_distances2k[0]).sum()
                    # neigh.keighbors returns (([d[x,k1], d[x,k2], ...]), (ik1, ik2, ...)) like (array([[ 0.3,  0.7]]), array([[2, 0]]))

# ndarray化
local_training_masks = np.array(local_training_masks)

# time
times.append(time.time())
time_interval.append('{0:.1f}'.format((times[-1]-times[-2])/60.))

"""============================================ EVALUATION and SAVE DATA =================================================="""
print '{0:.1f}[min] Evaluating '.format((times[-1]-times[0])/60.) + setting_file_name
# for layer(分ける深度毎) and profile for training
#   ・データ数，ラベル（正，負例）
#   ・tp, tn, fp, fn
#   ・AUCを評価（ROC図は保存しない）
# ・時刻
# ・実験条件jsonを同ディレクトリに保存
# ・データ保存
#    ・異常度，補間値，fpr，tpr，
# を保存

# 分けるレイヤーマスク
partitioning_layer_masks = [(setting_file['partitioning_depths'][i]<=test_window_depth)&(test_window_depth<setting_file['partitioning_depths'][i+1]) for i in range(len(setting_file['partitioning_depths'])-1)]

# layer
# データ数（正，負例）
tr_layer_n = (~np.isnan(training_windows).any(axis=2)).sum()
te_layer_n = (~np.isnan(anomaly_scores)).sum()# (~np.isnan(test_windows).any(axis=2)).sum()
te_layer_1_n = (test_window_qcs==0).sum()
te_layer_4_n = (test_window_qcs==1).sum()
# データ数（正，負例） for 分けられた層毎
tr_layer_partitioned_n = [(~np.isnan(training_windows[:,partitioning_layer_mask]).all(axis=2)).sum() for partitioning_layer_mask in partitioning_layer_masks]
te_layer_partitioned_n = [(~np.isnan(test_windows[:,partitioning_layer_mask]).all(axis=2)).sum() for partitioning_layer_mask in partitioning_layer_masks]
te_layer_partitioned_1_n = [(test_window_qcs[:,partitioning_layer_mask]==0).sum() for partitioning_layer_mask in partitioning_layer_masks]
te_layer_partitioned_4_n = [(test_window_qcs[:,partitioning_layer_mask]==1).sum() for partitioning_layer_mask in partitioning_layer_masks]
# fpr, tpr, AUC
layer_fpr, layer_tpr, layer_threshold = roc_curve(test_window_qcs[~np.isnan(anomaly_scores)], anomaly_scores[~np.isnan(anomaly_scores)])
layer_auc = auc(layer_fpr, layer_tpr)
# fpr, tpr, AUC for 分けられた層毎
layer_partitioned_fprs, layer_partitioned_tprs, layer_partitioned_thresholds, layer_partitioned_aucs = [np.nan] * len(partitioning_layer_masks), [np.nan] * len(partitioning_layer_masks), [np.nan] * len(partitioning_layer_masks), [np.nan] * len(partitioning_layer_masks)
for i in range(len(partitioning_layer_masks)):
    #if not np.isnan(anomaly_scores[:,partitioning_layer_masks[i]]).all(): # 層に１つもエラーが無いときは，anomaly_scoresがnp.nanになる
    partitioned_test_window_qc = test_window_qcs[:,partitioning_layer_masks[i]]
    if (partitioned_test_window_qc==1).any() and (partitioned_test_window_qc==0).any():
        partitioned_anomaly_scores = anomaly_scores[:,partitioning_layer_masks[i]]
        layer_partitioned_fprs[i], layer_partitioned_tprs[i], layer_partitioned_thresholds[i] = roc_curve(partitioned_test_window_qc[~np.isnan(partitioned_anomaly_scores)], partitioned_anomaly_scores[~np.isnan(partitioned_anomaly_scores)])
        layer_partitioned_aucs[i] = auc(layer_partitioned_fprs[i], layer_partitioned_tprs[i])

# sys.exit()

'''ROCのプロット
layer_auc_all = auc(layer_fpr, layer_tpr)
plt.plot(layer_fpr, layer_tpr, label='All areas', c='k')

# make roc pictures for layer
fig2, ax2 = plt.subplots() # for window unit
#ax2.set_title('ROC Curve (Window)')##############
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_xlim(0,1)
ax2.set_ylim(0,1)
ax2.plot(layer_fpr, layer_tpr, c='k')#, label='All areas')
#ax2.legend(loc='lower right')#, fontsize=12)##############
fig2.savefig(save_dir + '/roc_curve_layer.png', format = 'png', dpi=300)
# close
plt.close('all')
'''


''' # profile単位の評価は置いとく
# profile
# データ数（正，負例）
tr_profile_n = (~np.isnan(training_windows).any(axis=2)).any(axis=1)
te_profile_n = (~np.isnan(test_windows).any(axis=2)).any(axis=1)

test_profile_qcs = (test_window_qcs==1)
profile_anomaly_scores =

profile_1_n = (test_window_qcs==0).sum()
profile_4_n = (test_window_qcs==1).sum()
# データ数（正，負例） for 分けられた層毎
tr_profile_partitioned_n = [(~np.isnan(training_windows[:,partitioning_layer_mask]).any(axis=2)).sum() for partitioning_layer_mask in partitioning_layer_masks]
te_profile_partitioned_n = [(~np.isnan(test_windows[:,partitioning_layer_mask]).any(axis=2)).sum() for partitioning_layer_mask in partitioning_layer_masks]
profile_1_n = [(test_window_qcs[:,partitioning_layer_mask]==0).sum() for partitioning_layer_mask in partitioning_layer_masks]
profile_4_n = [(test_window_qcs[:,partitioning_layer_mask]==1).sum() for partitioning_layer_mask in partitioning_layer_masks]
# fpr, tpr, AUC
profile_fpr, profile_tpr, profile_threshold = roc_curve(test_window_qcs[~np.isnan(test_window_qcs)], anomaly_scores[~np.isnan(anomaly_scores)])
profile_auc = auc(layer_fpr, layer_tpr)
# fpr, tpr, AUC for 分けられた層毎
layer_partitioned_fpr, layer_partitioned_tpr, layer_partitioned_threshold layer_auc = [0] * len(partitioning_layer_masks)
for i in range(len(partitioning_layer_masks)):
    partitioned_test_window_qc = test_window_qcs[:,partitioning_layer_mask[i]]
    partitioned_anomaly_scores = anomaly_scores[:,partitioning_layer_mask[i]]
    layer_partitioned_fpr[i], layer_partitioned_tpr[i], layer_partitioned_threshold[i] = roc_curve(partitioned_test_window_qc[~np.isnan(partitioned_test_window_qc)], partitioned_anomaly_scores[~np.isnan(partitioned_anomaly_scores)])
    layer_auc[i] = auc(layer_partitioned_fpr[i], layer_partitioned_tpr[i])

'''


# making result dir
to_day = datetime.datetime.today()
dir_name = str(to_day).replace(':','_').replace(' ','_').replace('-','_').replace('.','_') + '_'
    # selecting名を追加
if setting_file['spatial_selecting_method'] == 'hierarchical_clustering1':
    dir_name += 'hierarchical_clustering1_' + setting_file['selecting_method_parameters']['distance_measure']
elif setting_file['spatial_selecting_method'] == 'hierarchical_clustering2':
    dir_name += 'hierarchical_clustering2_' + setting_file['selecting_method_parameters']['distance_measure']
elif setting_file['spatial_selecting_method'] == 'rectangle':
    dir_name += 'rectangle_' + str(setting_file['selecting_method_parameters']['rectangle_range'][0]) + str(setting_file['selecting_method_parameters']['rectangle_range'][1])
elif setting_file['spatial_selecting_method'] == 'all':
    dir_name += 'all'
    # method名を追加
if setting_file['ad_method'] == 'OCSVM': # 'OCSVM'
    dir_name += '_OCSVM_' + 'nu_' + str(setting_file['OCSVM_params']['nu']).replace('.','') + '_gamma' + str(setting_file['OCSVM_params']['gamma']).replace('.','')
elif setting_file['ad_method'] == 'k-thNN':
    dir_name += '_kNN_' + 'k_' + str(setting_file['k-thNN_params']['k'])
elif setting_file['ad_method'] == 'IsolationForest':
    dir_name += '_IF_' + 'n_estimators_' + str(setting_file['IsolationForest_params']['n_estimators']) + '_max_samples_' + str(setting_file['IsolationForest_params']['max_samples']) + '_contamination_' + str(setting_file['IsolationForest_params']['outliers_fraction']).replace('.','')
elif setting_file['ad_method'] == 'Mahalanobis':
    dir_name += '_Mahalanobis'
elif setting_file['ad_method'] == 'LOF':
    dir_name += '_LOF_' + 'k_' + str(setting_file['LOF_params']['k']) + '_contamination_' + str(setting_file['LOF_params']['contamination']).replace('.','')
elif setting_file['ad_method'] == 'SpatialWeightedKNN':
    dir_name += '_SpatialWeightedKNN_' + 'k_' + str(setting_file['SpatialWeightedKNN_params']['k']) + '_InfluenceRadius_' + str(setting_file['SpatialWeightedKNN_params']['influence_radius']).replace('.','')
save_dir = os.path.join(setting_file['result_dir'] , dir_name + '_' + setting_file_name.split('/')[-1].split('.')[0])
os.makedirs(save_dir)

#sys.exit()

# numpyデータの保存
# save anomly profile layer scores
np.save(os.path.join(save_dir, 'training_windows.npy'), training_windows) # Z標準化前training_windows
np.save(os.path.join(save_dir, 'test_windows.npy'), test_windows) # Z標準化前 if 'all' 以外，Z標準化後 if 'all'
np.save(os.path.join(save_dir, 'test_window_qcs.npy'), test_window_qcs)
np.save(os.path.join(save_dir, 'anomaly_scores.npy'), anomaly_scores)
np.save(os.path.join(save_dir, 'layer_fpr.npy'), layer_fpr)
np.save(os.path.join(save_dir, 'layer_tpr.npy'), layer_tpr)
for i in xrange(len(layer_partitioned_fprs)):
    if isinstance(layer_partitioned_fprs[i], np.ndarray):
        np.save(os.path.join(save_dir, 'layer' + str(i+1) + '_partitioned_fprs.npy'), layer_partitioned_fprs[i])
        np.save(os.path.join(save_dir, 'layer' + str(i+1) + '_partitioned_tprs.npy'), layer_partitioned_tprs[i])
if setting_file['spatial_selecting_method'] == 'all':
    np.save(os.path.join(save_dir, 'local_training_windows.npy'), local_training_windows)
    np.save(os.path.join(save_dir, 'ori_test_windows.npy'), ori_test_windows) # Z標準化済み
else:
    np.save(os.path.join(save_dir, 'local_training_masks.npy'), local_training_masks)

''' # interpolated not windowはとりあえず保存しない
if setting_file['feature_selection']  is 'with_gradient': # array of (value, gradient)
    np.save(os.path.join(save_dir, 'training_interp_values.npy'), training_interp_values)
    np.save(os.path.join(save_dir, 'training_interp_gradients.npy'), training_interp_gradients)
elif setting_file['feature_selection']  is 'raw': # array of value

elif setting_file['feature_selection']  is 'gradient':
'''

# setting_fileの保存
with open(os.path.join(save_dir, 'setting_file.json'), 'w') as f:
    json.dump(setting_file, f, sort_keys=False, indent=4)

# result ファイルの保存
try:
    program_name = __file__
except:
    program_name = '???.py'

# time
times.append(time.time())
time_interval.append('{0:.1f}'.format((times[-1]-times[-2])/60.))

result = [
{'partitioning_depths':setting_file['partitioning_depths']},
{'=============== layer partitioned result':'=================='},
{'partitioned layer auc':layer_partitioned_aucs},
{'all layer auc':layer_auc},

{'=============== info about test data' : '=================='},
{'actual test_layer_partitioned_n(anomaly_scores)' : (~np.isnan(anomaly_scores)).sum()},
{'actual all layer_partitioned_n(anomaly_scores)' : [(~np.isnan(anomaly_scores))[:,partitioning_layer_mask].sum() for partitioning_layer_mask in partitioning_layer_masks]},
{'test_layer_partitioned_n' : te_layer_partitioned_n},
{'test layer partitioned normal n ' : te_layer_partitioned_1_n},
{'test layer partitioned error n ' : te_layer_partitioned_4_n},
{'test all layer n' : te_layer_n},
{'test all layer_normal_n ' : te_layer_1_n },
{'test all layer_error_n ' : te_layer_4_n },
{'test profile n ' : test_n},
{'test profile normal n ' : (test_window_qcs==0).all(axis=1).sum()},
{'test profile error n ' : (test_window_qcs==1).any(axis=1).sum()},
{'test layered profile normal n ' : [(test_window_qcs[:,partitioning_layer_mask]==0).all(axis=1).sum() for partitioning_layer_mask in partitioning_layer_masks]},
{'test layered profile error n ' : [(test_window_qcs[:,partitioning_layer_mask]==1).any(axis=1).sum() for partitioning_layer_mask in partitioning_layer_masks]},
{'test layered profile n ' : [(~np.isnan(test_window_qcs[:,partitioning_layer_mask])).any(axis=1).sum() for partitioning_layer_mask in partitioning_layer_masks]},

{'=============== info about test data' : '=================='},
{'tr_layer_partitioned_n' : tr_layer_partitioned_n},
{'training all layer number' : tr_layer_n},
{'training profile n ' : training_n},
{'training layered profile n ' : [(~np.isnan(training_windows[:,partitioning_layer_mask])).all(axis=2).any(axis=1).sum() for partitioning_layer_mask in partitioning_layer_masks]},

{'============== miscellaneous':'====================='},
{'times[min] (Preprocessing, clustering, Test, Evaluation&Save)':time_interval},
{'today':str(to_day)},
{'memo':memo}
]

if not setting_file['spatial_selecting_method'] == 'all': #in ('hierarchical_clustering1', 'hierarchical_clustering2'):
    # resultに各層でのトレーニングデータ数の平均を収納
    result += [
    {'============== local training mask':'====================='},
    {'training mean n':local_training_masks.sum(axis=1).mean()}
    ]


with open(os.path.join(save_dir, 'result.json'), 'w') as f:
    json.dump(result, f, sort_keys=False, indent=4, ensure_ascii=False)

print '{0:.1f}[min] Program Finished '.format((times[-1]-times[0])/60.) + setting_file_name


'''================ここから作業@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
・保存すべきデータ何か？
・make_test_window_arrayとmake_sliding_window_arrayは絶対イケてるかどうか分からんけど，とりあえずやる．（前一回できたことを確認した）
・ないっちゃないとおもうけど，Z標準化の時に，0,1軸の数が一緒の時に，1軸からmean，std引きたい時に，0軸から弾いてしまうときがありえる．
・最終的にtestとtrainingのデータ数を数えるとき，データ１つの深度はstdが0になって計算できないから，anomaly_scoresをnp.nanにするため，データ数からひかないといけない
    ・実際のtestデータ数はanomlay_scoresのnp.isnan()で判断すべきだがとりあえず放置
・EllipticEnvelopeが自分で計算したマハラノビス距離と一致するか確認
・OCSVMにおけるグリッドサーチ実装
・original training_test_windowsを raw_...に変える．変数名をぐちゃぐちゃ市内
・training_latsとかで，どういう風なクラスタ併合が行われたか確認できるようにしたいが面倒なので，放置

ex2のクラスタリングと固定幅
・クラスタリングはk-meansとかで既存パッケージ使える？
・計算量大変だから，PCA使うのも手？
・それか，スペクトラルクラスタリングか．

現在作業
・層ごとに違う全部異なる手法パラメータで行えるように設計．
・shallowのex3 25,26がエラーだから，juypterでsys.exit()いれて確認 after 現在の python ... ; python ...: ...
・
'''



'''
#グリッドサーチ用コード
tuned_parameters = [
    {'nu': [0.1, 0.25, 0.5], 'gamma': [0.1, 0.01, 0.001, 0.0001]},
    {'nu': [0.1, 0.25, 0.5], 'gamma': [0.1, 0.01, 0.001, 0.0001]},
    {'nu': [0.1, 0.25, 0.5], 'gamma': [0.1, 0.01, 0.001, 0.0001]},
    {'nu': [0.1, 0.25, 0.5], 'gamma': [0.1, 0.01, 0.001, 0.0001]},
    ]
#  If gamma is ‘auto’ then 1/n_features will be used instead.

clf = GridSearchCV(
    svm.OneClassSVM(), # 識別器
    tuned_parameters, # 最適化したいパラメータセット
    cv=5, # 交差検定の回数
    scoring='roc_auc')

clf.fit(X_train, y_train)

clf.grid_scores_ # 各思考のスコア

clf.best_params_ # 最適



svm.OneClassSVM(nu=setting_file['OCSVM_params']['nu'], kernel='kernel', gamma=setting_file['OCSVM_params'] ['gamma']))


'''
