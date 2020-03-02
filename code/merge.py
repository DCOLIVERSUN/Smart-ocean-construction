# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

##############
# 读取各模型结果
res1 = np.loadtxt('result_1.txt')
res2 = np.loadtxt('result_2.txt')
res3 = np.loadtxt('result_3.txt')
##############

##############
# 读取测试渔船 ID
res = pd.read_csv('test_data.csv', header = 0)
res = res[['ship']].astype('int')
##############

##############
# 模型融合

result = res1 * 0.3 + res2 * 0.5 + res3 * 0.3
##############

##############
# 计算结果
result = np.argmax(result, axis = 1)
##############

##############
# 结果映射与保存
res['type'] = result
res.type = res.type.map({0:'围网', 1:'刺网', 2:'拖网'})
res.sort_values(by='ship', inplace=True)
res.to_csv('result.csv', index = None, header = None, encoding = 'utf-8')