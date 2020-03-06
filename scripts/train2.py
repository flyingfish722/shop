# 用数据文件名替换xxx
import sys
import os
import re
import json

# 获取脚本所在目录绝对路径p
p = os.path.abspath(sys.argv[0])
p = re.findall(r'.+[\\/]', p)[0][:-1]

# 获取工程根目录
rootdir = p
for i in range(2):
    rootdir = re.findall(r'.+[\\/]', rootdir)[0][:-1]
sys.path.append(rootdir)
import common_train2

# xxx 是预测数据的文件名-------
train_data_path = os.path.join(p, 'xxx.csv')
#------------------------------
result_dir = os.path.join(p, 'result')
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
    
# aaa 设置为保存模型的文件夹------------
model_sv_dir = os.path.join(rootdir, 'model', 'aaa')
#--------------------------------------
if not os.path.exists(model_sv_dir):
    os.mkdir(model_sv_dir)
outfilewithnewfeature = os.path.join(p, 'bbwithnewfeature.csv')
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

config_file = os.path.join(model_sv_dir, 'config.json')
config = {}
config['columns_heat'] = columns_heat
config['columns_cer'] = columns_cer
# 以下三选一，设为True
config['t_all'] = True
config['t_7_30'] = False
config['t_7_15_60'] = False
# ---------------------
with open(config_file, 'w') as cf:
    json.dump(config, cf)

common_train2.train(train_data_path, model_sv_dir, columns_heat, columns_cer, model_sv_dir,
                    heat_model_max_depth=10,
                    heat_model_max_leaf_nodes=None,
                    cer_model_max_depth=10,
                    cer_model_max_leaf_nodes=None,
                    outfilewithnewfeature=outfilewithnewfeature,
                    t_7_30=config['t_7_30'],
                    t_7_15_60=config['t_7_15_60'],
                    t_all=config['t_all'])
'''
train() 参数说明:
def train(train_data_path, result_dir,
          heat_model_max_depth = None, 
          heat_model_max_leaf_nodes = None,
          cer_model_max_depth = None,
            cer_model_max_leaf_nodes = None,
          outfilewithnewfeature=None,
          dealt=None,
          learning_depths_heat=None,
          learning_depths_cer=None,
          complexity_depths_heat=None,
          complexity_depths_cer=None,
          grid_search_heat=None,
          grid_search_cer=None
          )
    train_data_path：
        训练数据文件路径
    result_dir：
        结果文件夹
    columns_heat:
        热度模型特征
    columns_cer：
        效率模型特征
    model_save_dir：
        模型保存地址
    heat_model_max_depth=None：
        热度模型最大深度参数
    heat_model_max_leaf_nodes=None：
        热度模型最大叶子数参数
    cer_model_max_depth=None：
        效度模型最大深度参数
    cer_model_max_leaf_nodes=None：
        效度模型最大叶子数参数
    outfilewithnewfeature =None：
        中间文件，增加了新的特征值。默认为None。
    dealt=None：
        默认为None，如果已经生成过中间文件，则可以将文件路径传入，省去数据处理时间。
    learning_depths_heat=None:
        热度模型学习曲线深度取值，为一个四元组，包含四个深度值，如[9, 12, 15, 18]
    learning_depths_cer=None:
        效率，同上
    complexity_depths_heat=None:
        热度模型复杂度曲线深度取值范围，为一个列表，如list(range(10,20))
    complexity_depths_cer=None:
        同上
    grid_search_heat=None:
        热度模型参数搜索的参数值取值，默认为None，不进行参数搜索。常用的比如max_leaf_nodes, max_depth, min_samples_split等参数。
        格式为字典，比如{max_depth:[8,9,10,11], max_leaf_nodes:list(range(10,100,10))}
    grid_search_cer=None:
        热度模型参数搜索的参数值取值，默认为None，同上。
    t_7_30=False:
        true表示只选择t t7 t30特征
    t_7_15_60=False：
        true表示只选择t7 t15 t60特征
    t_all=False：
        true表示选择所有t特征   
        
    

    其中train_data_path, result_dir为必选参数，其他根据需要选择。
    例如，如果是生成中间文件并进行参数搜索：
        train(train_data_path=..., result_dir=..., 
            outfilewithnewfeature=中间文件存储路径,
            grid_search_heat={max_depth:list(range(8,20,2)), max_leaf_nodes:list(range(50,300,10))},
            grid_search_cer={max_depth:list(range(10,24,2)), max_leaf_nodes:list(range(100,500,10))})
    如果已经生成过中间文件，进行参数搜索：
        train(train_data_path=..., result_dir=..., 
            dealt=中间文件路径,
            grid_search_heat={max_depth:list(range(8,20,2)), max_leaf_nodes:list(range(50,300,10))},
            grid_search_cer={max_depth:list(range(10,24,2)), max_leaf_nodes:list(range(100,500,10))})
    如果已经搜索确定了参数并且生成过中间文件,假设热度模型最大深度为7，最大叶子数为200，效率模型最大深度为8，最大叶子数为250：
        train(train_data_path=..., result_dir=...,
          heat_model_max_depth = 7, 
          heat_model_max_leaf_nodes = 200,
          cer_model_max_depth = 8,
            cer_model_max_leaf_nodes = 250,
            dealt=...)
    如果已经搜索确定了参数并且生成过中间文件,需要画出学习曲线和复杂度曲线：
        train(train_data_path=..., result_dir=...,
          heat_model_max_depth = 7, 
          heat_model_max_leaf_nodes = 200,
          cer_model_max_depth = 8,
            cer_model_max_leaf_nodes = 250,
            dealt=...,
            learning_depths_heat=[9,12,15,18],
          learning_depths_cer=[8,12,16,20],
          complexity_depths_heat=list(range(8,20)),
          complexity_depths_cer=list(range(8,20)))
'''
