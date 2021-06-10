# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def probability_density(df, indicator, ax = None):
    ax1 = plt.subplots()[1] if (ax is None) else ax
    if indicator == 'duration': 
        bin_width = 60 # 1 minute
        min_value = 0
        max_value = 60 * 121 # 60 minutes
        xlabel = "Duration"
    if indicator == "length":
        bin_width = 100 # meters
        min_value = 0
        max_value = 10000
        xlabel = "Length"
    if indicator == "speed":
        bin_width = 1 # meters
        min_value = 0
        max_value = 25
        xlabel = "Speed"
    res_pr = ax1.hist(df[indicator], bins = np.arange(min_value, max_value, bin_width), density = True,
                 edgecolor = 'grey', color = 'C1', linewidth = 0.3, cumulative = False)
    res_cdf = ax1.hist(df[indicator], bins = np.arange(min_value, max_value, bin_width), density = True,
                 edgecolor = 'grey', color = 'C1', linewidth = 0.3, cumulative = -1)
    return res_pr,res_cdf

def plot_indicator(df, indicator, ax = None, out_file = None):
    """
    """
    df = df.copy()
    ax1 = plt.subplots()[1] if (ax is None) else ax
    if indicator == 'duration': 
        bin_width = 60 # 1 minute
        min_value = 0
        max_value = 60 * 121 # 60 minutes
        xlabel = "Duration"
    if indicator == "length":
        bin_width = 100 # meters
        min_value = 0
        max_value = 10000
        xlabel = "Length"
    if indicator == "speed":
        bin_width = 1 # meters
        min_value = 0
        max_value = 25
        xlabel = "Speed"
    
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 20,
    }
    font2 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 16,
    }
    res = ax1.hist(df[indicator], bins = np.arange(min_value, max_value, bin_width), density = True,
                 edgecolor = 'grey', color = 'C1', linewidth = 0.3, cumulative = False)
    _ = ax1.set_yticks(ax1.get_yticks())
    _ = ax1.set_yticklabels([round(x,2) for x in ax1.get_yticks() * bin_width],**font2)
    _ = ax1.set_ylabel("Probability",**font1)

    if indicator == "duration":
        _ = ax1.set_xlabel("Duration [min]",**font1) 
        _ = ax1.set_xticks([0, 60 * 15, 60 * 30, 60 * 45, 60 * 60, 60 * 75, 60 * 90
                            , 60 * 105, 60 * 120])
        _ = ax1.set_xticklabels([0, 15, 30, 45, 60, 75, 90 ,105, 120],**font2)
    if indicator == 'length':
        _ = ax1.set_xlabel("Distance [km]",**font1) 
        _ = ax1.set_xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000
                            ,9000, 10000] )
        _ = ax1.set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],**font2)
    
    if indicator == 'speed':
        _ = ax1.set_xlabel("Speed [km/h]",**font1) 
        _ = ax1.set_xticks([0, 1*3, 1*6, 1*9, 1*12, 1*15, 1*18, 1*21, 1*24,
                            1*27] )
        _ = ax1.set_xticklabels([0, 3, 6, 9, 12, 15, 18, 21, 24, 27],**font2)
    
    ax2 = ax1.twinx()
    new_xs = []
    for i in range(1, len(res[1])):
        new_xs.append((res[1][i-1] + res[1][i]) * 0.5 )
    new_ys = [0] * len(res[0])
    new_ys[0] = res[0][0] * bin_width
    for i in range(1, len(res[0])):
        new_ys[i] = res[0][i] * bin_width + new_ys[i-1]
    ax2.plot(new_xs, new_ys, color='b')
    ax2.set_ylim(0, 1.03)
    _ = ax2.set_ylabel("Cumulative probability",**font1)
    _ = ax2.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0],**font2)

    if out_file: plt.savefig(out_file, bbox_inches='tight', dpi = 300)

def get_density(df, indicator):
    """
    The cumulative density.
    $ F_X(x) = P(X <= x) $
    The inverse cumulative density. 
    $ F_X(x) = P(X \geq x) $

    Parameters
    ----------
    df     :
        The data
    indicator : 
    
    Returns
    ----------
    df_tmp: The dataframe that records differnt types probability
          density, cumulative density, inverse cumulative density
    """
    df_tmp = df[indicator].value_counts().reset_index()
    df_tmp.columns = [indicator, 'count']
    df_tmp = df_tmp.sort_values(indicator).reset_index().drop(columns = ['index'])
    df_tmp['density'] = df_tmp['count'] / df_tmp['count'].sum()
    cur_density = df_tmp['density'].iloc[0]
    list_res    = [cur_density]
    for i in range(1, df_tmp.shape[0]):
        list_res.append(cur_density + df_tmp['density'].values[i])
        cur_density += df_tmp['density'].values[i]
    df_tmp['cum_density'] = list_res

    cur_density = 1
    list_res = [cur_density]
    for i in range(1, df_tmp.shape[0]):
        list_res.append(cur_density - df_tmp['density'].values[i-1])
        cur_density = cur_density - df_tmp['density'].values[i-1]
    df_tmp['inv_cum_density'] = list_res
    return df_tmp

# def plot_icd(df, indicator, xlog=False, ylog=False, ax = None, out_file = None):
#     """
#     plot inverse cumulative density
    
#     Parameters
#     ----------
#     df     :
#         Data
#     indicator : 
        
#     xlog:
#         Log the indicator values
#     ylog:
#         Log the probability values
#     """
#     df_icd = inverse_cumulative_density(df, indicator)
#     if xlog: df_icd['indicator'] = np.log10(df_icd['indicator'])
#     if ylog: df_icd['inv_cum_density'] = np.log10(df_icd['inv_cum_density'])
    
#     ax = plt.subplots()[1] if (ax is None) else ax
#     ax.scatter(df_icd['indicator'], df_icd['inv_cum_density'], s = 1)
#     ax.set_ylabel("Inverse cumulative density")
#     if out_file: plt.savefig(out_file, bbox_inches='tight', dpi = 300)

if __name__ == "__main__":
    ######Statistical distribution
    columns = ['vehicle_id', 'stime', 'etime', 'slng', 'slat', 'elng', 'elat']
    in_file   = r"D:/Code/Python/SamplingRate/FinalTrips/D1/trips_02.pickle"
    data1 = pickle.load(open(in_file, 'rb'))
    #X1=data.drop(columns,axis=1)  
    # X1.to_csv('TripsD1.csv')
   
    in_file   = r"D:/Code/Python/SamplingRate/FinalTrips/D2/trips_02.pickle"
    data2 = pickle.load(open(in_file, 'rb'))
    # X2=data2.drop(columns,axis=1)  
    # # X2.to_csv('TripsD2.csv')

    in_file   = r"D:/Code/Python/SamplingRate/FinalTrips/D3/trips_02.pickle"
    data3 = pickle.load(open(in_file, 'rb'))
    # X3=data3.drop(columns,axis=1)  
    # # X3.to_csv('TripsD3.csv')

    in_file   = r"D:/Code/Python/SamplingRate/FinalTrips/D4/trips_02.pickle"
    data4 = pickle.load(open(in_file, 'rb'))
    # X4=data4.drop(columns,axis=1)  
    # # X4.to_csv('TripsD4.csv')

    in_file   = r"D:/Code/Python/SamplingRate/FinalTrips/D5/trips_02.pickle"
    data5 = pickle.load(open(in_file, 'rb'))
    # X5=data5.drop(columns,axis=1)  
    # X5.to_csv('TripsD5.csv')

    in_file   = r"D:/Code/Python/SamplingRate/FinalTrips/D6/trips_02.pickle"
    data6 = pickle.load(open(in_file, 'rb'))
    # X6=data6.drop(columns,axis=1)  
    # # X6.to_csv('TripsD6.csv')

    in_file   = r"D:/Code/Python/SamplingRate/FinalTrips/D7/trips_02.pickle"
    data7 = pickle.load(open(in_file, 'rb'))
    # X7=data7.drop(columns,axis=1)  
    # # X7.to_csv('TripsD7.csv')

    in_file   = r"D:/Code/Python/SamplingRate/FinalTrips/D8/trips_02.pickle"
    data8 = pickle.load(open(in_file, 'rb'))
    # X8=data8.drop(columns,axis=1)  
    # # X8.to_csv('TripsD8.csv')

    in_file   = r"D:/Code/Python/SamplingRate/FinalTrips/D9/trips_02.pickle"
    data9 = pickle.load(open(in_file, 'rb'))
    # X9=data9.drop(columns,axis=1)  
    # # X9.to_csv('TripsD9.csv')

    in_file   = r"D:/Code/Python/SamplingRate/FinalTrips/D10/trips_02.pickle"
    data10 = pickle.load(open(in_file, 'rb'))
    # X10=data10.drop(columns,axis=1)  
    # # X10.to_csv('TripsD10.csv')

#######Plot the probability density distribution and cumulative probability distribution
    plot_indicator(data1, 'duration', ax = None, out_file = True)
    # plot_indicator(data, 'length', ax = None, out_file = True)
    # plot_indicator(data10, 'speed', ax = None, out_file = True)
    # test1 = data[data['duration']>5400]
    # test2 = data[data['length']>8000]

########calculate the probability density
    # df_duration = pd.DataFrame()
    # res_pr,res_cdf = probability_density(data1, 'duration', ax = None)
    # df_duration['prb_1'] = res_pr[0]
    # df_duration['cdf_1'] = res_cdf[0]
    # res_pr,res_cdf = probability_density(data2, 'duration', ax = None)
    # df_duration['prb_2'] = res_pr[0]
    # df_duration['cdf_2'] = res_cdf[0]
    # res_pr,res_cdf = probability_density(data3, 'duration', ax = None)
    # df_duration['prb_3'] = res_pr[0]
    # df_duration['cdf_3'] = res_cdf[0]
    # res_pr,res_cdf = probability_density(data4, 'duration', ax = None)
    # df_duration['prb_4'] = res_pr[0]
    # df_duration['cdf_4'] = res_cdf[0]
    # res_pr,res_cdf = probability_density(data5, 'duration', ax = None)
    # df_duration['prb_5'] = res_pr[0]
    # df_duration['cdf_5'] = res_cdf[0]
    # res_pr,res_cdf = probability_density(data6, 'duration', ax = None)
    # df_duration['prb_6'] = res_pr[0]
    # df_duration['cdf_6'] = res_cdf[0]
    # res_pr,res_cdf = probability_density(data7, 'duration', ax = None)
    # df_duration['prb_7'] = res_pr[0]
    # df_duration['cdf_7'] = res_cdf[0]
    # res_pr,res_cdf = probability_density(data8, 'duration', ax = None)
    # df_duration['prb_8'] = res_pr[0]
    # df_duration['cdf_8'] = res_cdf[0]
    # res_pr,res_cdf = probability_density(data9, 'duration', ax = None)
    # df_duration['prb_9'] = res_pr[0]
    # df_duration['cdf_9'] = res_cdf[0]
    # res_pr,res_cdf = probability_density(data10, 'duration', ax = None)
    # df_duration['prb_10'] = res_pr[0]
    # df_duration['cdf_10'] = res_cdf[0]
    # df_duration.to_csv('pr_duration.csv')
    
    
 
