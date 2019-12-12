#此脚本文件放于对应商店文件夹中

import time
s = time.time()
import sys
import os
import warnings
import re
warnings.filterwarnings('ignore')

p = os.path.abspath(sys.argv[0])
for i in range(4):
    p = re.findall(r'.+[\\/]', p)[0][:-1]
sys.path.append(p)
import common_predict

#使用某模型预测
cbtm = True
im = False
if cbtm:
    heat_model_path = os.path.join(p, 'data', 'cbtm', 'result', 'reg_tree_heat.pickle')
    cer_model_path = os.path.join(p, 'data', 'cbtm', 'result', 'reg_tree_cer.pickle')
elif im:
    heat_model_path = os.path.join(p, 'data', 'IM', 'result', 'reg_tree_heat.pickle')
    cer_model_path = os.path.join(p, 'data', 'IM', 'result', 'reg_tree_cer.pickle')
predict_data_path = os.path.join(p, 'predict', 'tm', 'bb', 'bb.csv') #xxx为文件全名
outfile = os.path.join(p, 'predict', 'tm', 'bb', 'result', 'bb.csv') #yyy为结果保存文件
if not os.path.exists(os.path.join(p, 'predict', 'tm', 'bb', 'result')):
    os.mkdir(os.path.join(p, 'predict', 'tm', 'bb', 'result'))
common_predict.predict(predict_data_path, outfile, heat_model_path, cer_model_path)
e = time.time()
print('````````````````````')
print('总用时：', round(e-s, 4), 's')
