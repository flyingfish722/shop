#用数据文件名替换xxx

import sys
import os
import re

p = os.path.abspath(sys.argv[0])
p = re.findall(r'.+[\\/]',p)[0][:-1]
if not os.path.exists(os.path.join(p, 'result')):
    os.mkdir(os.path.join(p, 'result'))

rootdir = p
for i in range(2):
    rootdir = re.findall(r'.+[\\/]',rootdir)[0][:-1]
sys.path.append(rootdir)
import common_train

train_data_path = os.path.join(p, 'xxx') #xxx 是预测数据的文件名
result_dir = os.path.join(p, 'result')
outfilewithnewfeature = os.path.join(p, 'cbtmwithnewfeature.csv')
common_train.train(train_data_path, result_dir, 7, None, 7, None,
    outfilewithnewfeature=outfilewithnewfeature)
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
