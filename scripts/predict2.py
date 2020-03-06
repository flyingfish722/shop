"""
此脚本文件放于predict文件夹下对应商店文件夹中
"""
import time
s = time.time()
import sys
import os
import warnings
import re
import json
warnings.filterwarnings('ignore')

p = os.path.abspath(sys.argv[0])
p = re.findall(r'.+[\\/]', p)[0][:-1]
rootdir = p
for i in range(2):
    rootdir = re.findall(r'.+[\\/]', rootdir)[0][:-1]
sys.path.append(rootdir)
import common_predict2

# xxx是模型所在文件夹的名字，一般也就是店铺的名字-----------
shop_name = 'xxx'
#--------------------------------
model_path = os.path.join(rootdir, 'model', shop_name)
model_path = os.path.join(rootdir, 'model', shop_name)
    
heat_model_path = os.path.join(model_path, 'reg_tree_heat.pickle')
cer_model_path = os.path.join(model_path, 'reg_tree_cer.pickle')

config_file = os.path.join(model_path, 'config.json')
with open(config_file, 'r') as cf:
    config = json.load(cf)
columns_heat = config['columns_heat']
columns_cer = config['columns_cer']
# -----------------------------------
predict_data_path = os.path.join(p, 'test.csv') #xxx为文件全名
outfile = os.path.join(p, 'result', 'test_predicted_by_idx.csv') #yyy为结果保存文件
#------------------------------------
if not os.path.exists(os.path.join(p, 'result')):
    os.mkdir(os.path.join(p, 'result'))
common_predict2.predict(predict_data_path, outfile, heat_model_path, cer_model_path, columns_heat, columns_cer, t=15,
                        t_7_30=config['t_7_30'],
                        t_7_15_60=config['t_7_15_60'],
                        t_all=config['t_all'],
                        dealt=os.path.join(p, 'test_middle.csv'))
e = time.time()
print('````````````````````')
print('总用时：', round(e-s, 4), 's')
