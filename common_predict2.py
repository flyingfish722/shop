import sklearn.tree
import pickle
import os
import pandas
import dataprocess2
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
            columns_heat,
            columns_cer,
            outfilewithnewfeature =None,
            dealt=None,
            t_7_30=False,
            t_7_15_60=False,
            t_all=False,
            p_score=True,
            t = 0,
            a = 50,
            r = 0.8,
            warning = False,
            show = False):
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
    t:
        保护周期
    warning:
        预警
    show:
        显示内容
    '''
              
    if not dealt:
        #构建特征值
        newdata = dataprocess2.process(predict_data_path, 'predict', outfilewithnewfeature, t,
                                       t_all=t_all, t_7_30=t_7_30, t_7_15_60=t_7_15_60)
    else:
        newdata = pandas.read_csv(dealt, encoding=encoding)

    #预警
    if warning:
        for each in ['uv', 'collection_num', 'atc_num', 'pay_items', 'inv_trans']:
            if newdata[each].sum() == 0:
                print('总', each, '为0')
            if newdata[each].sum() / newdata[each+'_t7'].sum() <= 0.1:
                print('总',each,'占七天总',each,'的比例不大于0.1')
        if newdata['GMV'][0] == 0:
            print('总销售额为0')
        if newdata['GMV'][0] / newdata['GMV_t7'][0] <= 0.1:
            print('当日总销售额占七天总销售额和的比例不大于0.1')
    if show:
        for each in ['uv', 'pay_items', 'inv_trans', 'GMV']:
            print('当日总{0}{1}，与前七天总{0}平均值的比例{2:.4f}'\
                  .fortmat(each, newdata[each].sum(), newdata[each].sum()/newdata)[each+'t_7'].sum()*7)
        for each in ['discount_rate', 'transrare']:
            weighted_avg = newdata['pay_items']*newdata[each]/newdata['pay_items'].sum()
            weighted_avg_7 = newdata['pay_items_t7']*newdata[each+'_t7']/newdata['pay_items_t7'].sum()/7
            print('当日加权平均{0}{1:.2%},与前七天加权平均值的比例{2:.2f}'.format(each, weighted_avg, weighted_avg/weighted_avg_7))

        data = pandas.read_csv(predict_data_path, encoding=encoding)
        old_date = data['data_date'].tolist()
        olddate = []
        for each in old_date:
            if each not in olddate:
                olddate.append(each)
        thisday = data[data.data_date == olddate[-1]]
        ra0 = len(data[(data.data_date==thisday) & (data.retail_amount==0)])
        inv0 = len(data[(data.data_date==thisday) & (data.inv_num==0)])
        sells = len(data[(data.data_date==thisday) & (data.pay_items!=0)])\
                /len(data[(data.data_date==thisday)])

        pre7_ra0 = pre7_inv0 = pre7_sells = 0
        for i in range(-8,-1):
            date = olddate[i]
            pre7_ra0 += len(data[(data.data_date==date) & (data.retail_amount==0)])
            pre7_inv0 += len(data[(data.data_date==date) & (data.inv_num==0)])
            pre7_sells += len(data[(data.data_date==date) & (data.pay_items!=0)])\
                /len(data[(data.data_date==date)])
        print('当日吊牌价是0的商品数目{0}，与前七天该值平均数的比例{1:.4f}\n'.format(ra0, ra0/pre7_ra0*7),
              '当日库存是0的商品数目{0}，与前七天该值平均数的比例{1:.4f}\n'.format(inv0, inv0/pre7_inv0*7),
              '当日有销量的商品占总商品数的比例{0:.4f}，与前七天该值平均数的比例{1:.4f}'.format(sells, sells/pre7_sells*7))

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

    #计算分
    # cal = newdata[newdata.t != -1].index
    # if not cal.empty:
    #     result_cer[cal] = result_cer[cal]+a*(1-r**(t-newdata.loc[cal, 't']))

    if p_score:
        performance_score(result_heat, result_cer)
    d = pandas.read_csv(predict_data_path[0:predict_data_path.rfind('.')] + '-clean' + '.csv', encoding=encoding)
    d['heat_predict'] = result_heat
    d['cer_predict'] = result_cer
    d.to_csv(outfile, encoding=encoding, index=None)
    end = time.time()
    print('预测结束，用时：', round(end - start, 3), 's')

   # 测试用
   # y_heat = newdata['heat']
   # y_cer = newdata['cer']
   # # score
   # print('热度回归树模型········')
   # print('深度：', reg_tree_heat.get_depth())
   # print('叶子节点数：', reg_tree_heat.get_n_leaves())
   # print('开始测试...')
   # start = time.time()
   # print('测试得分: ', round(reg_tree_heat.score(X_heat, y_heat), 4))
   # end = time.time()
   # print('用时：', round(end - start, 3), 's')
   #
   # print('效率回归树模型········')
   # print('深度：', reg_tree_cer.get_depth())
   # print('叶子节点数：', reg_tree_cer.get_n_leaves())
   # print('开始测试...')
   # start = time.time()
   # print('测试得分: ', round(reg_tree_cer.score(X_cer, y_cer), 4))
   # end = time.time()
   # print('用时：', round(end - start, 3), 's')
    
