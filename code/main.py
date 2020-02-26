# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import xgboost as xgb

###########
# 预处理函数

def preprocess(df):
    df.drop_duplicates(['time'], keep = 'first')
    df['time'] = pd.to_datetime(df['time'],format='%m%d %H:%M:%S')
    df['minute'] = df['time'].dt.minute
    df = del_dup(df)
    df = del_v(df)
    del df['minute']
    return df

# 处理重复数据
def del_dup(df):
    res = []
    df['minute'] = pd.to_numeric(df['minute'])
    for i in range(1, len(df)):
        if (df['lat'].iloc[i] == df['lat'].iloc[i - 1] and df['lon'].iloc[i] == df['lon'].iloc[i - 1] and df['速度'].iloc[i] != 0.0):
            df['速度'].iloc[i] = 0.0
        if ((df['minute'].iloc[i] // 10) == (df['minute'].iloc[i-1] // 10)):
            res.append(i)
    df = df.drop(df.index[res])
    return df

# 处理速度
def del_v(df):
    df['v_stage'] = 0
    for i in range(len(df)):
        if df['速度'].iloc[i] == 0:
            df['v_stage'].iloc[i] = 0
        elif 0 < df['速度'].iloc[i] <= 2:
            df['v_stage'].iloc[i] = 1
        elif 2 < df['速度'].iloc[i] <= 6:
            df['v_stage'].iloc[i] = 2
        elif 6 < df['速度'].iloc[i]:
            df['v_stage'].iloc[i] = 3
    return df
###########
    
###########
# 特征提取
    
def feature_engineer(df, flag=True):
    
    df = preprocess(df)
    
    if flag == False:
        features.append(df['渔船ID'].iloc[0])      # 渔船ID
    features.append(df['lat'].min())              #x_min
    features.append(df['lat'].max())              #x_max
    features.append(df['lat'].mean())             #x_mean
    features.append(df['lat'].quantile(0.25))     #x_1/4
    
    features.append(df['lon'].min())              #y_min
    features.append(df['lon'].max())              #y_max
    features.append(df['lon'].mean())             #y_mean
    features.append(df['lon'].quantile(0.75))     #y_3/4
    
    features.append(df['lat'].cov(df['lon']))       #xy_cov
    
    df['time']=pd.to_datetime(df['time'],format='%Y%m%d %H:%M:%S')
    t_diff=df['time'].diff().iloc[1:].dt.total_seconds()
    
    x_diff=df['lat'].diff().iloc[1:].abs()
    y_diff=df['lat'].diff().iloc[1:].abs()
    x_a = (x_diff/t_diff).mean()
    y_a = (y_diff/t_diff).mean()
    
    features.append(np.sqrt(x_a ** 2 + y_a ** 2))   #a
    
    features.append(df['速度'].mean())           #v_mean
    features.append(df['速度'].std())            #v_std
    features.append(df['速度'].quantile(0.75))   #v_3/4
         
    features.append(df['方向'].mean())           #d_mean

    features.append(len(df[df['v_stage'] == 0]) / len(df)) # 静止率
    if len(df[df['v_stage'] != 0]) == 0:
        features.append(0)          # 中速率
    else:
        features.append(len(df[df['v_stage'] == 2]) / len(df[df['v_stage'] != 0])) # 中速率

    
    if(flag):
        if df['type'].iloc[0] == '围网':
            features.append(0)
        elif df['type'].iloc[0] == '刺网':
            features.append(1)
        elif df['type'].iloc[0] == '拖网':
            features.append(2)
##########

# TODO
# 先用处理好的数据
data_path = r'../data'
train_data = pd.read_csv(os.path.join(data_path, 'train_data.csv'), header = 0)
##########
# 处理训练集
            
#features = []
## TODO
## 提交前修改为 tcdata
#train_path = r'../data/hy_round2_train_20200225'
#data_path = r'../data'
#train_files = os.listdir(train_path)
#train_files_len = len(train_files)
#
#for file in tqdm(train_files):
#    df = pd.read_csv(os.path.join(train_path, file), header=0, keep_default_na=False)
#    feature_engineer(df, flag=True)
#
#train_data = pd.DataFrame(np.array(features).reshape(train_files_len, int(len(features) / train_files_len)))
#train_data.columns = ['x_min','x_max','x_mean','x_1/4',
#                     'y_min','y_max','y_mean','y_3/4',
#                     'xy_cov',
#                     'a',
#                     'v_mean','v_std','v_3/4',
#                     'd_mean', 'static_ratio', 'medium_v_ratio',
#                     'type']
##########

##########
# 处理测试集

#features = []
#test_path = r'./tcdata/hy_round2_testA_20200225'
#test_files = os.listdir(test_path)
#test_files_len = len(test_files)
#
#for file in tqdm(test_files):
#    df = pd.read_csv(os.path.join(test_files, file), header=0, keep_default_na=False)
#    feature_engineer(df, flag=False)
#test_data = pd.DataFrame(np.array(features).reshape(test_files_len, int(len(features) / test_files_len)))
#test_data.columns = ['ship',
#                     'x_min','x_max','x_mean','x_1/4',
#                     'y_min','y_max','y_mean','y_3/4',
#                     'xy_cov',
#                     'a',
#                     'v_mean','v_std','v_3/4',
#                     'd_mean', 'static_ratio', 'medium_v_ratio']
##########

##########
# ##########
# 分离特征与标签

target = train_data.type
train_data.drop(['type'],axis=1,inplace=True)
# TODO
# 需要设置 test_data
##########

##########
# 配置 XGBClassifier 参数并训练
## 未标明随机数，如遇问题可电联 18500242957

model = xgb.XGBClassifier(n_estimators = 150, learning_rate = 0.39, max_depth = 6, 
                          reg_alpha = 0.004, reg_lambda = 0.002, importance_type = 'total_cover',
                          n_jobs = -1, random_state = 0)
##########

##########
# 20折交叉验证

result = []
scores = []
models = []
## 未标明随机数，如遇问题可电联 18500242957
fold = StratifiedKFold(n_splits = 20, shuffle = True, random_state = 0)
for index, (train_idx, test_idx) in enumerate(fold.split(train_data,target)):
    x_train = train_data.iloc[train_idx]
    y_train = target.iloc[train_idx]
    x_test = train_data.iloc[test_idx]
    y_test = target.iloc[test_idx]
    
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    score = f1_score(y_test, pred, average='macro')
    models.append(model)
    scores.append(score)
    print(index, 'F1 Score: ', score)
    
#    prediction = model.predict_proba(test_data)
#    result.append(np.argmax(prediction, axis=1))
    
print('XGB mean F1 Score: ' + str(np.mean(scores)))
##########

##########
# 保存结果

#submit_path = r'../submit'
#res = []
#for i in range(2000):
#    tmp = np.bincount(np.array(result,dtype='int')[:,i])
#    res.append(np.argmax(tmp))
#
#ans = pd.DataFrame(np.arange(9000,11000,1))
#ans['type'] = pd.Series(res).map({0:'围网', 1:'刺网', 2:'拖网'})
#ans.to_csv(os.path.join(submit_path, 'submit_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.csv'), index = None, header = None, encoding = 'utf-8')
#
#print(ans['type'].value_counts()/2000)
##########