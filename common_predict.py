import sklearn.tree
import pickle
import os
import pandas
import dataprocess
import time
import random
encoding = 'utf8'


def performance_score(heat, cer):
    for i in range(heat.size):
        a = round(random.uniform(0,0.05), 4)
        b = round(random.uniform(0,0.1), 4)
        heat[i] = heat[i]*(1-a)
        cer[i] = cer[i]*(1+b)


def predict(predict_data_path,
            outfile,
            heat_model_path,
            cer_model_path,
            outfilewithnewfeature =None,
            dealt=None,
            new_date_start=None,
            p_score=True):
    '''
    predict_data_path：
        预测数据文件路径
    outfile：
        输出文件路径
    heat_model_path：
        热度模型存储路径
    cer_model_path：
        效度模型存储路径
    outfilewithnewfeature =None：
        中间文件，增加了新的特征值。默认为None，如果已经生成过此文件，则可以将文件路径
    传入，省去数据处理时间。
    dealt=None：
        默认为None，如果已经生成过中间文件，则可以将文件路径传入，省去数据处理时间。
    new_date_start=None：
        预测数据日期。例如预测2019/9/15或者2019/9/15-2019/9/21，则new_date_start = '2019/9/15'
    p_score:
        是否将预测值发散，默认发散。
    '''
            
    # 去掉热度0，效率50的新商品
    o_data = pandas.read_csv(predict_data_path, encoding=encoding)
    data = o_data[~((o_data.heat == 0) & (o_data.cer == 50))]
    data.index = range(len(data))
    #去掉吊牌价为0的数据

    data = data[data.retail_amount != 0]
    data.index = range(len(data))
    # 获取数据的日期
    old_date = data['data_date'].tolist()
    old_date_without_repeat = []
    for each in old_date:
        if each not in old_date_without_repeat:
            old_date_without_repeat.append(each)
    if not new_date_start:
        new_date = [old_date_without_repeat[-1]]
    else:
        new_date = old_date_without_repeat[old_date_without_repeat.index(new_date_start):]
        
    data1 = o_data[((o_data.heat == 0) & (o_data.cer == 50)) & (o_data.data_date == new_date[0])]
    data2 = o_data[(o_data.retail_amount == 0) & (o_data.data_date == new_date[0])]

        
    if not dealt:
        old_data_path = predict_data_path[0:predict_data_path.rfind('.')] + 'withoutnewitems' + '.csv'
        data.to_csv(old_data_path, encoding=encoding, index=None)

        #构建特征值
        newdata = dataprocess.process(old_data_path, old_date_without_repeat, new_date, outfilewithnewfeature)
    else:
        newdata = pandas.read_csv(dealt, encoding=encoding)
    # columns_cer = columns_heat = ['uv', 'transrare', 'atc_num', 'collection_num', 'GMV', 'discount_rate',
    #                               'pay_items', 'inv_num', 'PMV', 'division=APP',
    #                               'uv_t7', 'GMV_t7', 'discount_rate_t7', 'pay_items_t7',
    #                               'uv_t30', 'GMV_t30', 'discount_rate_t30', 'pay_items_t30',
    #                               'pay_items_rate', 'inv_num_rate', 'uv_rate', 'collection_num_rate',
    #                               'uv_t7_rate', 'pay_items_t7_rate', 'pay_items_t30_rate', 'uv_t30_rate']
#OLD MODEL
    '''
    columns_cer = columns_heat = ['uv', 'transrare', 'atc_num', 'collection_num', 'GMV', 'discount_rate',
                                  'pay_items', 'inv_num', 'PMV', 'division=APP',
                                  'uv_t7', 'GMV_t7', 'discount_rate_t7', 'pay_items_t7',
                                  'uv_t30', 'GMV_t30', 'discount_rate_t30', 'pay_items_t30',
                                  'pay_items_rate', 'inv_num_rate', 'uv_rate', 'collection_num_rate']
    '''
#NEW MODEL

    columns_cer = columns_heat = ['uv', 'transrare', 'atc_num', 'collection_num', 'GMV', 'discount_rate',
                                  'pay_items', 'inv_num', 'PMV',
                                  'uv_t7', 'GMV_t7', 'discount_rate_t7', 'pay_items_t7',
                                  'uv_t30', 'GMV_t30', 'discount_rate_t30', 'pay_items_t30',
                                  'pay_items_rate', 'inv_num_rate', 'uv_rate', 'collection_num_rate']

    X_heat = newdata[columns_heat]
    X_cer = newdata[columns_cer]
    heat_f = open(heat_model_path, 'rb')
    cer_f = open(cer_model_path, 'rb')
    reg_tree_heat = pickle.load(heat_f)
    reg_tree_cer = pickle.load(cer_f)

    # 做出预测
    print('开始预测...')
    start = time.time()
    result_heat = reg_tree_heat.predict(X_heat)
    result_cer = reg_tree_cer.predict(X_cer)
    if p_score:
        performance_score(result_heat, result_cer)

    st = data[data.data_date == new_date[0]].index.tolist()[0]
    d = data.loc[st:]
    d['heat_predict'] = result_heat
    d['cer_predict'] = result_cer
    data1['heat_predict'] = None
    data1['cer_predict'] = None
    data2['heat_predict'] = None
    data2['cer_predict'] = None
    d = d.append(data1)
    d = d.append(data2)
    d.to_csv(outfile, encoding=encoding, index=None)
    end = time.time()
    print('预测结束，用时：', round(end - start, 3), 's')

##    测试用
##    y_heat = newdata['heat']
##    y_cer = newdata['cer']
##    # score
##    print('热度回归树模型········')
##    print('深度：', reg_tree_heat.get_depth())
##    print('叶子节点数：', reg_tree_heat.get_n_leaves())
##    print('开始测试...')
##    start = time.time()
##    print('测试得分: ', round(reg_tree_heat.score(X_heat, y_heat), 4))
##    end = time.time()
##    print('用时：', round(end - start, 3), 's')
##
##    print('效率回归树模型········')
##    print('深度：', reg_tree_cer.get_depth())
##    print('叶子节点数：', reg_tree_cer.get_n_leaves())
##    print('开始测试...')
##    start = time.time()
##    print('测试得分: ', round(reg_tree_cer.score(X_cer, y_cer), 4))
##    end = time.time()
##    print('用时：', round(end - start, 3), 's')
    
