# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

###############
# 读取训练集数据

train_data = pd.read_csv('train_data.csv', header = 0)
train_data.drop(['x_max_x_min'], axis=1,inplace=True)
###############

###############
# 读取测试集数据

test_data = pd.read_csv('test_data.csv', header = 0)
test_data.drop(['x_max_x_min'], axis=1, inplace=True)
###############

###############
# 分离特征与标签

target = train_data.type
train_data.drop(['type'],axis=1,inplace=True)
test_data.drop(['ship'],axis=1,inplace=True)
################

################
# 配置 XGBClassifier 参数

model = xgb.XGBClassifier(n_estimators = 150, learning_rate = 0.39, max_depth = 6, 
                          reg_alpha = 0.004, reg_lambda = 0.002, importance_type = 'total_cover',
                          n_jobs = -1, random_state = 0)
################

################
# 20折交叉验证

scores = []
prediction = np.zeros((1, 3))
fold = StratifiedKFold(n_splits = 20, shuffle = True, random_state = 380)
for index, (train_idx, test_idx) in enumerate(fold.split(train_data,target)):
    x_train = train_data.iloc[train_idx]
    y_train = target.iloc[train_idx]
    x_test = train_data.iloc[test_idx]
    y_test = target.iloc[test_idx]
    
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    score = f1_score(y_test, pred, average='macro')
    scores.append(score)
    print(index, 'F1 Score: ', score)
    
    prediction = prediction + model.predict_proba(test_data)

prediction = prediction/20
    
print('Tag 2.1, XGB mean F1 Score: ' + str(np.mean(scores)))
################

################
# 保存结果

np.savetxt('result_2.txt', prediction)
################