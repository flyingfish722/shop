#此脚本文件放于对应商店文件夹中

import time
s = time.time()
import sys
import os
import warnings
import re
warnings.filterwarnings('ignore')

p = os.path.abspath(sys.argv[0])
p = re.findall(r'.+[\\/]', p)[0][:-1]
rootdir = p
for i in range(3):
    rootdir = re.findall(r'.+[\\/]', rootdir)[0][:-1]
sys.path.append(rootdir)
import common_predict2

#使用某模型预测
cbtm = False
im = False
bb = True
if cbtm:
    heat_model_path = os.path.join(rootdir, 'data', 'cbtm', 'result', 'reg_tree_heat.pickle')
    cer_model_path = os.path.join(rootdir, 'data', 'cbtm', 'result', 'reg_tree_cer.pickle')
elif im:
    heat_model_path = os.path.join(rootdir, 'data', 'IM', 'result', 'reg_tree_heat.pickle')
    cer_model_path = os.path.join(rootdir, 'data', 'IM', 'result', 'reg_tree_cer.pickle')
elif bb:
    heat_model_path = os.path.join(rootdir, 'data', 'new modle', 'result', 'reg_tree_heat.pickle')
    cer_model_path = os.path.join(rootdir, 'data', 'new modle', 'result', 'reg_tree_cer.pickle')

predict_data_path = os.path.join(p, 'bb.csv') #xxx为文件全名
outfile = os.path.join(p, 'result', 'bb.csv') #yyy为结果保存文件
if not os.path.exists(os.path.join(p, 'result')):
    os.mkdir(os.path.join(p, 'result'))

columns_cer = columns_heat = [
    'uv', 'transrare', 'atc_num', 'collection_num',
    'GMV', 'discount_rate', 'pay_items', 'inv_num', 'PMV', 'inv_trans',

    'uv_rate', 'transrare_rate', 'atc_num_rate', 'collection_num_rate',
    'discount_rate_rate', 'pay_items_rate', 'inv_num_rate', 'inv_trans_rate',

    'uv_percentile', 'transrare_percentile', 'atc_num_percentile', 'collection_num_percentile',
    'discount_rate_percentile', 'pay_items_percentile', 'inv_num_percentile', 'inv_trans_percentile',

    'uv_t7', 'transrare_t7', 'atc_num_t7', 'collection_num_t7',
    'GMV_t7', 'discount_rate_t7', 'pay_items_t7', 'inv_num_t7', 'PMV_t7',

    'uv_t15', 'transrare_t15', 'atc_num_t15', 'collection_num_t15',
    'GMV_t15', 'discount_rate_t15', 'pay_items_t15', 'inv_num_t15', 'PMV_t15',

    'uv_t30', 'transrare_t30', 'atc_num_t30', 'collection_num_t30',
    'GMV_t30', 'discount_rate_t30', 'pay_items_t30', 'inv_num_t30', 'PMV_t30',

    'uv_t60', 'transrare_t60', 'atc_num_t60', 'collection_num_t60',
    'GMV_t60', 'discount_rate_t60', 'pay_items_t60', 'inv_num_t60', 'PMV_t60'

]
common_predict2.predict(predict_data_path, outfile, heat_model_path, cer_model_path, columns_heat, columns_cer, t=15,
                        t_all=True)
e = time.time()
print('````````````````````')
print('总用时：', round(e-s, 4), 's')
