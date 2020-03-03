#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 09:50:51 2020

@author: oliver_sun
"""


import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss


#############
# 读取数据
feature_train = ['x_min','x_max','x_mean','x_1/4', 'x_1/2', 
                 'y_min','y_max','y_mean','y_3/4',
                 'xy_cov',
                 'a',
                 'v_mean','v_std','v_3/4',
                 'd_mean', 'static_ratio', 'medium_v_ratio',
                 'low_lon_ratio', 'medium_lon_ratio', 
                 'type']
feature_test = ['ship',
                'x_min','x_max','x_mean','x_1/4', 'x_1/2', 
                'y_min','y_max','y_mean','y_3/4',
                'xy_cov',
                'a',
                'v_mean','v_std','v_3/4',
                'd_mean', 'static_ratio', 'medium_v_ratio',
                'low_lon_ratio', 'medium_lon_ratio']

train_data = pd.read_csv('train_data.csv', header = 0)
test_data = pd.read_csv('test_data.csv', header = 0)

train_data = train_data[feature_train]
test_data = test_data[feature_test]

target = train_data.type
train_data.drop(['type'], axis=1, inplace=True)
test_data.drop(['ship'], axis=1,inplace=True)
############

############
# 配置5个xgb模型
params = []
params.append({'objective': 'multi:softprob',
               'num_round':150,
               'eta':0.39,
               'num_class':3,
               'max_depth':6,
               'alpha':0.004,
               'lambda':0.002,
               'seed':0})
params.append({'objective': 'multi:softprob',
               'num_round':150,
               'eta':0.39,
               'num_class':3,
               'max_depth':6,
               'alpha':0.004,
               'lambda':0.002,
               'seed':128})
params.append({'objective': 'multi:softprob',
               'num_round':150,
               'eta':0.39,
               'num_class':3,
               'max_depth':6,
               'alpha':0.004,
               'lambda':0.002,
               'seed':512})
params.append({'objective': 'multi:softprob',
               'num_round':150,
               'eta':0.39,
               'num_class':3,
               'max_depth':6,
               'alpha':0.004,
               'lambda':0.002,
               'seed':1024})
params.append({'objective': 'multi:softprob',
               'num_round':150,
               'eta':0.39,
               'num_class':3,
               'max_depth':6,
               'alpha':0.004,
               'lambda':0.002,
               'seed':2048})

#clfs.append(xgb.XGBClassifier(n_estimators = 150, learning_rate = 0.39, max_depth = 6, 
#                          reg_alpha = 0.004, reg_lambda = 0.002, importance_type = 'total_cover',
#                          n_jobs = -1, random_state = 0))
##################

##################
# stacking
train_stackers = []
for RS in [0, 1, 2, 64, 128, 256, 380, 512, 1024, 2048, 4096]:
    skf = StratifiedKFold(n_splits=20, random_state=RS, shuffle=True)
    train_stacker = [[0.0 for s in range(3)] for k in range(0, (train_data.shape[0]))]
    cv_scores = {i:[] for i in range(len(params))}
    cv_scores['Avg'] = []
    cnt = 0
    print("Begin 20-flod cross validation")
    for train_idx, val_idx in skf.split(train_data, target):
        cnt += 1
        X_train, y_train = train_data.iloc[train_idx], target.iloc[train_idx]
        X_val, y_val = train_data.iloc[val_idx], target.iloc[val_idx]
        preds = []
        k = 0
        for param in params:
            xg_train = xgb.DMatrix(X_train, label=y_train)
            clf = xgb.train(param, xg_train, param['num_round'])
            y_val_pred = clf.predict(xgb.DMatrix(X_val))
            loss = log_loss(y_val, y_val_pred)
            preds.append(y_val_pred)
            cv_scores[k].append(loss)
            k += 1
            print("Clf_{} iteration {}'s loss: {}".format(k, cnt, loss))
        preds = np.array(preds)
        avg_pred = np.mean(preds, axis=0)
        avg_loss = log_loss(y_val, avg_pred)
        cv_scores["Avg"].append(avg_loss)
        print("Iteration {}'s Avg loss: {}".format(cnt, avg_loss))
        no = 0
        for real_idx in val_idx:
            for i in range(3):
                train_stacker[real_idx][i] = avg_pred[no][i]
            no += 1
    for i in range(len(params)):
        print("clf_{} validation loss : {}".format(i, np.mean(cv_scores[i])))
    print("Average validation loss : {}".format(np.mean(cv_scores["Avg"])))
    train_stackers.append(train_stacker)
train_stacker = np.mean(train_stackers, axis=0)
print("*** Validation finished ***\n")

print("Begin predicting")
test_stacker = [[0.0 for s in range(3)]   for k in range (0,(test_data.shape[0]))]
preds = []
for i in range(len(params)):
    print("Clf_{} fiting".format(i))
    xg = xgb.DMatrix(train_data, label=target)
    clf = xgb.train(params[i], xg, params[i]['num_round'])
    print("Clf_{} predicting".format(i))
    pred = clf.predict(xgb.DMatrix(test_data))
    preds.append(pred)
preds = np.mean(preds, axis=0)
for pr in range(0, len(preds)):
    for d in range(0, 3):
        test_stacker[pr][d] = preds[pr][d]
    
    
    
############
#print("Stacking")
#    train = train_data.copy()
#    test = test_data.copy()
#    y = train["interest_level"].apply(lambda x: target_num_map[x])
#    del train["interest_level"]
#    train_stackers = []
#    for RS in [0, 1, 2, 64, 128, 256, 512, 1024, 2048, 4096]:
#        skf = StratifiedKFold(n_splits=10, random_state=RS, shuffle=True)
#        #Create Arrays for meta
#        train_stacker = [[0.0 for s in range(3)]  for k in range (0,(train.shape[0]))]
#        cv_scores = {i:[] for i in range(len(clfs))}
#        cv_scores["Avg"] = []
#        print("Begin 10-flod cross validation")
#        cnt = 0
#        for train_idx, val_idx in skf.split(train, y):
#            cnt += 1
#            X = train.copy()
#            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
#            X_train, X_val, feats = coreProcess(X, y_train, train_idx, val_idx)
#            X_train.toarray()
#            preds = []
#            k = 0
#            for clf in clfs:
#                clf.fit(X_train, y_train)
#                y_val_pred = clf.predict_proba(X_val)
#                loss = log_loss(y_val, y_val_pred)
#                preds.append(y_val_pred)
#                cv_scores[k].append(loss)
#                k += 1
#                print("Clf_{} iteration {}'s loss: {}".format(k, cnt, loss))
#            preds = np.array(preds)
#            avg_pred = np.mean(preds, axis=0)
#            avg_loss = log_loss(y_val, avg_pred)
#            cv_scores["Avg"].append(avg_loss)
#            print("Iteration {}'s Avg loss: {}".format(cnt, avg_loss))
#            no = 0
#            for real_idx in val_idx:
#                for i in range(3):
#                    train_stacker[real_idx][i] = avg_pred[no][i]
#                no += 1
#        for i in range(len(clfs)):
#            print("clf_{} validation loss : {}".format(i, np.mean(cv_scores[i])))
#        print("Average validation loss : {}".format(np.mean(cv_scores["Avg"])))
#        train_stackers.append(train_stacker)
#    train_stacker = np.mean(train_stackers, axis=0)
#    print("*** Validation finished ***\n")
#
#    test_stacker = [[0.0 for s in range(3)]   for k in range (0,(test.shape[0]))]
#    train_idx = [i for i in range(train.shape[0])]
#    test_idx = [i + train.shape[0] for i in range(test.shape[0])]
#    data = pd.concat([train, test]).reset_index()
#    X_train, X_test, feats = coreProcess(data, y, train_idx, test_idx)
#    print(X_train.shape, len(train_stacker))
#    print("Begin predicting")
#    preds = []
#    for i in range(len(clfs)):
#        print("Clf_{} fiting".format(i))
#        clfs[i].fit(X_train, y)
#        print("Clf_{} predicting".format(i))
#        pred = clfs[i].predict_proba(X_test)
#        preds.append(pred)
#    preds = np.mean(preds, axis=0)
#    for pr in range (0, len(preds)):  
#            for d in range (0,3):            
#                test_stacker[pr][d]=(preds[pr][d])   
#    print ("merging columns")   
#    #stack xgboost predictions
#    X_train = np.column_stack((X_train.toarray(),train_stacker))
#    # stack id to test
#    X_test = np.column_stack((X_test.toarray(),test_stacker))         
#    # stack target to train
#    X = np.column_stack((y,X_train))
#    ids = test.listing_id.values
#    X_test = np.column_stack((ids, X_test))
#    np.savetxt("./train_stacknet.csv", X, delimiter=",", fmt='%.5f')
#    np.savetxt("./test_stacknet.csv", X_test, delimiter=",", fmt='%.5f') 
#    print("Write results...")
#    output_file = "submission_{}.csv".format(np.mean(cv_scores["Avg"]))
#    print("Writing submission to %s" % output_file)
#    f = open(output_file, "w")   
#    f.write("listing_id,high,medium,low\n")# the header   
#    for g in range(0, len(test_stacker))  :
#      f.write("%s" % (ids[g]))
#      for prediction in test_stacker[g]:
#         f.write(",%f" % (prediction))    
#      f.write("\n")
#    f.close()
#    print("Done.")