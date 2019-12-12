import pandas
import os
import time
import numpy

coding = 'utf8'


def add_feature(df, feature, dates):
    for each in dates:
        tmp = df[df.data_date == each]
        tmp_index = tmp.index.tolist()
        feature_sum = tmp[feature].sum()
        df.loc[tmp_index, feature + "_rate"] = tmp[feature] / feature_sum
        order = tmp[feature].sort_values().index
        p = pandas.Series(list(range(tmp[feature].size))) / tmp[feature].size * 100
        p.index = order
        df.loc[tmp_index, feature + "_percentile"] = p
    print(feature, ':完成')


def process(olddatapath, purpose, outfile=None, t=0, t_7_30=False, t_7_15_60=False, t_all=False):
    start = end = 0
    ts = te = 0
    ts = time.time()
    print('开始读取数据...')
    data = pandas.read_csv(olddatapath, encoding=coding, engine='python')
    # 获取数据的日期
    old_date = data['data_date'].tolist()
    olddate = []
    for each in old_date:
        if each not in olddate:
            olddate.append(each)
    if purpose == 'train':
        newdate = olddate[60]
    elif purpose == 'predict':
        newdate = olddate[-1]
    else:
        print('请检查purpose参数设置')
        return
    print('-----------------------------------')

    print('去掉吊牌价为0的数据')
    data = data[data.retail_amount != 0]
    print('-----------------------------------')

    print('去除uv,库存和销量都为0且效率低于5的数据')
    tmp = data[(data.uv == 0) & (data.inv_num == 0) & (data.pay_items == 0) & (data.cer <= 5)]
    print('这种数据有', tmp.shape[0], '条')
    data = data[(data.uv != 0) | (data.inv_num != 0) | (data.pay_items != 0) | (data.cer > 5)]
    data.index = range(len(data))
    print('-----------------------------------')

    if purpose == 'predict':
        start_index = data[data.data_date == newdate].index[0]
        cleandata = olddatapath[0:olddatapath.rfind('.')] + '-clean' + '.csv'
        data.loc[start_index:].to_csv(cleandata, encoding=coding, index=None)

    print('生成基本特征')
    start = time.time()
    data.loc[data[data.uv != 0].index, 'transrare'] = data['pay_items'] / data['uv']
    data['transrare'] = data['transrare'].fillna(0)
    data = data[data.retail_amount != 0]
    data.index = range(len(data))
    data['discount_rate'] = data['pay_amount'] / data['retail_amount']
    for each in olddate:
        aday = data[data.data_date == each].index.tolist()
        data.loc[aday, 'GMV'] = data.loc[aday, 'pay_amount'].sum()
    data['PMV'] = data['pay_amount'] / data['GMV']
    data['PMV'] = data['PMV'].fillna(0)
    end = time.time()
    print('耗时：', round(end - start, 3))
    print('--------------------------------------')
    # 更新onshelf_date
    print('更新onshelf_date')
    start = time.time()
    start_index = data[data.data_date == olddate[30]].index[0]
    for i in range(start_index, len(data)):

        if data.loc[i, 'data_date'] != data.loc[i - 1, 'data_date'] or i == start_index:
            dateIndexInOlddate = olddate.index(data.loc[i, 'data_date'])
            dstart = data[data.data_date == olddate[dateIndexInOlddate - 30]].index[0]
            dend = data[data.data_date == olddate[dateIndexInOlddate - 1]].index.tolist()[-1]
            tmp = data.loc[dstart:dend]
        if tmp[tmp.spu == data.loc[i, 'spu']].empty:
            tmp1 = data.loc[i:]
            data.loc[tmp1[tmp1.spu == tmp1.loc[i, 'spu']].index, 'onshelf_date'] = data.loc[i, 'data_date']
            # print('更新',  tmp1[tmp1.spu == tmp1.loc[i,'spu']].index.tolist())
    end = time.time()
    print('耗时：', round(end - start, 3), 's')
    print('------------------------------------')
    # data['onshelf_date'] = data['onshelf_date'].fillna('1/1/2018')

    print('生成_t特征')
    start = time.time()
    start_index = data[data.data_date == newdate].index[0]
    dict_indexs = {}
    if purpose == 'predict':
        data['t'] = -1
    day = ''
    for i in range(start_index, len(data)):
        i_spu = data.loc[i, 'spu']
        if i_spu not in dict_indexs.keys():
            i_spu_list = data[data.spu == i_spu].index.tolist()
            i_index = i_spu_list.index(i)
            dict_indexs[i_spu] = [i_spu_list, i_index]

        else:
            dict_indexs[i_spu][1] += 1
            i_index = dict_indexs[i_spu][1]
        if day != data.loc[i, 'data_date']:
            day = data.loc[i, 'data_date']
            dateIndexInOlddate = olddate.index(day)

        dend = data[data.data_date == olddate[dateIndexInOlddate - 1]].index.tolist()[-1]
        if data.loc[i, 'onshelf_date'] in olddate:
            days = dateIndexInOlddate - olddate.index(data.loc[i, 'onshelf_date'])
            if t_all:
                if days >= 60:
                    for k in [7, 15, 30, 60]:
                        dstart = data[data.data_date == olddate[dateIndexInOlddate - k]].index[0]
                        tmp = data.loc[dstart:dend]
                        data.loc[i, 'GMV_t' + str(k)] = sum(set(tmp['GMV']))
                        tmp = tmp[tmp.spu == data.loc[i, 'spu']]

                        data.loc[i, 'uv_t' + str(k)] = tmp['uv'].sum()
                        data.loc[i, 'collection_num_t' + str(k)] = tmp['collection_num'].sum()
                        data.loc[i, 'atc_num_t' + str(k)] = tmp['atc_num'].sum()
                        data.loc[i, 'transrare_t' + str(k)] = tmp['transrare'].sum()
                        data.loc[i, 'pay_items_t' + str(k)] = tmp['pay_items'].sum()
                        data.loc[i, 'inv_num_t' + str(k)] = tmp['inv_num'].sum()
                        data.loc[i, 'discount_rate_t' + str(k)] = tmp['discount_rate'].sum()

                        data.loc[i, 'PMV_t' + str(k)] = tmp['PMV'].sum()
                elif days >= 30:
                    for k in [7, 15, 30]:
                        dstart = data[data.data_date == olddate[dateIndexInOlddate - k]].index[0]
                        tmp = data.loc[dstart:dend]
                        data.loc[i, 'GMV_t' + str(k)] = sum(set(tmp['GMV']))
                        tmp = tmp[tmp.spu == data.loc[i, 'spu']]

                        data.loc[i, 'uv_t' + str(k)] = tmp['uv'].sum()
                        data.loc[i, 'collection_num_t' + str(k)] = tmp['collection_num'].sum()
                        data.loc[i, 'atc_num_t' + str(k)] = tmp['atc_num'].sum()
                        data.loc[i, 'transrare_t' + str(k)] = tmp['transrare'].sum()
                        data.loc[i, 'pay_items_t' + str(k)] = tmp['pay_items'].sum()
                        data.loc[i, 'inv_num_t' + str(k)] = tmp['inv_num'].sum()
                        data.loc[i, 'discount_rate_t' + str(k)] = tmp['discount_rate'].sum()
                        data.loc[i, 'PMV_t' + str(k)] = tmp['PMV'].sum()

                    dstart = data[data.data_date == data.loc[i, 'onshelf_date']].index[0]
                    tmp = data.loc[dstart:dend]
                    data.loc[i, 'GMV_t60'] = sum(set(tmp['GMV'])) / days * 60
                    tmp = tmp[tmp.spu == data.loc[i, 'spu']]

                    data.loc[i, 'uv_t60'] = tmp['uv'].sum() / days * 60
                    data.loc[i, 'collection_num_t60'] = tmp['collection_num'].sum() / days * 60
                    data.loc[i, 'atc_num_t60'] = tmp['atc_num'].sum() / days * 60
                    data.loc[i, 'transrare_t60'] = tmp['transrare'].sum() / days * 60
                    data.loc[i, 'pay_items_t60'] = tmp['pay_items'].sum() / days * 60
                    data.loc[i, 'inv_num_t60'] = tmp['inv_num'].sum() / days * 60
                    data.loc[i, 'discount_rate_t60'] = tmp['discount_rate'].sum() / days * 60
                    data.loc[i, 'PMV_t60'] = tmp['PMV'].sum() / days * 60
                elif days >= 15:
                    for k in [7, 15]:
                        dstart = data[data.data_date == olddate[dateIndexInOlddate - k]].index[0]
                        tmp = data.loc[dstart:dend]
                        data.loc[i, 'GMV_t' + str(k)] = sum(set(tmp['GMV']))
                        tmp = tmp[tmp.spu == data.loc[i, 'spu']]

                        data.loc[i, 'uv_t' + str(k)] = tmp['uv'].sum()
                        data.loc[i, 'collection_num_t' + str(k)] = tmp['collection_num'].sum()
                        data.loc[i, 'atc_num_t' + str(k)] = tmp['atc_num'].sum()
                        data.loc[i, 'transrare_t' + str(k)] = tmp['transrare'].sum()
                        data.loc[i, 'pay_items_t' + str(k)] = tmp['pay_items'].sum()
                        data.loc[i, 'inv_num_t' + str(k)] = tmp['inv_num'].sum()
                        data.loc[i, 'discount_rate_t' + str(k)] = tmp['discount_rate'].sum()
                        data.loc[i, 'PMV_t' + str(k)] = tmp['PMV'].sum()

                    dstart = data[data.data_date == data.loc[i, 'onshelf_date']].index[0]
                    tmp = data.loc[dstart:dend]
                    for k in [30, 60]:
                        data.loc[i, 'GMV_t' + str(k)] = sum(set(tmp['GMV'])) / days * k
                    tmp = tmp[tmp.spu == data.loc[i, 'spu']]
                    for k in [30, 60]:
                        data.loc[i, 'uv_t' + str(k)] = tmp['uv'].sum() / days * k
                        data.loc[i, 'collection_num_t' + str(k)] = tmp['collection_num'].sum() / days * k
                        data.loc[i, 'atc_num_t' + str(k)] = tmp['atc_num'].sum() / days * k
                        data.loc[i, 'transrare_t' + str(k)] = tmp['transrare'].sum() / days * k
                        data.loc[i, 'pay_items_t' + str(k)] = tmp['pay_items'].sum() / days * k
                        data.loc[i, 'inv_num_t' + str(k)] = tmp['inv_num'].sum() / days * k
                        data.loc[i, 'discount_rate_t' + str(k)] = tmp['discount_rate'].sum() / days * k
                        data.loc[i, 'PMV_t' + str(k)] = tmp['PMV'].sum() / days * k
                elif days >= 7:
                    k = 7
                    dstart = data[data.data_date == olddate[dateIndexInOlddate - k]].index[0]
                    tmp = data.loc[dstart:dend]
                    data.loc[i, 'GMV_t7'] = sum(set(tmp['GMV']))
                    tmp = tmp[tmp.spu == data.loc[i, 'spu']]

                    data.loc[i, 'uv_t' + str(k)] = tmp['uv'].sum()
                    data.loc[i, 'collection_num_t' + str(k)] = tmp['collection_num'].sum()
                    data.loc[i, 'atc_num_t' + str(k)] = tmp['atc_num'].sum()
                    data.loc[i, 'transrare_t' + str(k)] = tmp['transrare'].sum()
                    data.loc[i, 'pay_items_t' + str(k)] = tmp['pay_items'].sum()
                    data.loc[i, 'inv_num_t' + str(k)] = tmp['inv_num'].sum()
                    data.loc[i, 'discount_rate_t' + str(k)] = tmp['discount_rate'].sum()
                    data.loc[i, 'PMV_t' + str(k)] = tmp['PMV'].sum()

                    dstart = data[data.data_date == data.loc[i, 'onshelf_date']].index[0]
                    tmp = data.loc[dstart:dend]
                    for k in [15, 30, 60]:
                        data.loc[i, 'GMV_t' + str(k)] = sum(set(tmp['GMV'])) / days * k
                    tmp = tmp[tmp.spu == data.loc[i, 'spu']]
                    for k in [15, 30, 60]:
                        data.loc[i, 'uv_t' + str(k)] = tmp['uv'].sum() / days * k
                        data.loc[i, 'collection_num_t' + str(k)] = tmp['collection_num'].sum() / days * k
                        data.loc[i, 'atc_num_t' + str(k)] = tmp['atc_num'].sum() / days * k
                        data.loc[i, 'transrare_t' + str(k)] = tmp['transrare'].sum() / days * k
                        data.loc[i, 'pay_items_t' + str(k)] = tmp['pay_items'].sum() / days * k
                        data.loc[i, 'inv_num_t' + str(k)] = tmp['inv_num'].sum() / days * k
                        data.loc[i, 'discount_rate_t' + str(k)] = tmp['discount_rate'].sum() / days * k
                        data.loc[i, 'PMV_t' + str(k)] = tmp['PMV'].sum() / days * k
                elif days > 0:
                    dstart = data[data.data_date == data.loc[i, 'onshelf_date']].index[0]
                    tmp = data.loc[dstart:dend]
                    for k in [7, 15, 30, 60]:
                        data.loc[i, 'GMV_t' + str(k)] = sum(set(tmp['GMV'])) / days * k
                    tmp = tmp[tmp.spu == data.loc[i, 'spu']]
                    for k in [7, 15, 30, 60]:
                        data.loc[i, 'uv_t' + str(k)] = tmp['uv'].sum() / days * k
                        data.loc[i, 'collection_num_t' + str(k)] = tmp['collection_num'].sum() / days * k
                        data.loc[i, 'atc_num_t' + str(k)] = tmp['atc_num'].sum() / days * k
                        data.loc[i, 'transrare_t' + str(k)] = tmp['transrare'].sum() / days * k
                        data.loc[i, 'pay_items_t' + str(k)] = tmp['pay_items'].sum() / days * k
                        data.loc[i, 'inv_num_t' + str(k)] = tmp['inv_num'].sum() / days * k
                        data.loc[i, 'discount_rate_t' + str(k)] = tmp['discount_rate'].sum() / days * k
                        data.loc[i, 'PMV_t' + str(k)] = tmp['PMV'].sum() / days * k
                else:
                    for k in [7, 15, 30, 60]:
                        data.loc[i, 'uv_t' + str(k)] = \
                            data.loc[i, 'collection_num_t' + str(k)] = \
                            data.loc[i, 'atc_num_t' + str(k)] = \
                            data.loc[i, 'transrare_t' + str(k)] = \
                            data.loc[i, 'pay_items_t' + str(k)] = \
                            data.loc[i, 'inv_num_t' + str(k)] = \
                            data.loc[i, 'discount_rate_t' + str(k)] = \
                            data.loc[i, 'GMV_t' + str(k)] = \
                            data.loc[i, 'PMV_t' + str(k)] = 0
            if t_7_30:
                if days >= 30:
                    for k in [7, 30]:
                        dstart = data[data.data_date == olddate[dateIndexInOlddate - k]].index[0]
                        tmp = data.loc[dstart:dend]
                        data.loc[i, 'GMV_t' + str(k)] = sum(set(tmp['GMV']))
                        tmp = tmp[tmp.spu == data.loc[i, 'spu']]

                        data.loc[i, 'uv_t' + str(k)] = tmp['uv'].sum()
                        data.loc[i, 'collection_num_t' + str(k)] = tmp['collection_num'].sum()
                        data.loc[i, 'atc_num_t' + str(k)] = tmp['atc_num'].sum()
                        data.loc[i, 'transrare_t' + str(k)] = tmp['transrare'].sum()
                        data.loc[i, 'pay_items_t' + str(k)] = tmp['pay_items'].sum()
                        data.loc[i, 'inv_num_t' + str(k)] = tmp['inv_num'].sum()
                        data.loc[i, 'discount_rate_t' + str(k)] = tmp['discount_rate'].sum()

                        data.loc[i, 'PMV_t' + str(k)] = tmp['PMV'].sum()
                elif days >= 7:
                    k = 7
                    dstart = data[data.data_date == olddate[dateIndexInOlddate - k]].index[0]
                    tmp = data.loc[dstart:dend]
                    data.loc[i, 'GMV_t' + str(k)] = sum(set(tmp['GMV']))
                    tmp = tmp[tmp.spu == data.loc[i, 'spu']]

                    data.loc[i, 'uv_t' + str(k)] = tmp['uv'].sum()
                    data.loc[i, 'collection_num_t' + str(k)] = tmp['collection_num'].sum()
                    data.loc[i, 'atc_num_t' + str(k)] = tmp['atc_num'].sum()
                    data.loc[i, 'transrare_t' + str(k)] = tmp['transrare'].sum()
                    data.loc[i, 'pay_items_t' + str(k)] = tmp['pay_items'].sum()
                    data.loc[i, 'inv_num_t' + str(k)] = tmp['inv_num'].sum()
                    data.loc[i, 'discount_rate_t' + str(k)] = tmp['discount_rate'].sum()
                    data.loc[i, 'PMV_t' + str(k)] = tmp['PMV'].sum()

                    k = 30
                    dstart = data[data.data_date == data.loc[i, 'onshelf_date']].index[0]
                    tmp = data.loc[dstart:dend]
                    data.loc[i, 'GMV_t' + str(k)] = sum(set(tmp['GMV'])) / days * k
                    tmp = tmp[tmp.spu == data.loc[i, 'spu']]

                    data.loc[i, 'uv_t' + str(k)] = tmp['uv'].sum() / days * k
                    data.loc[i, 'collection_num_t' + str(k)] = tmp['collection_num'].sum() / days * k
                    data.loc[i, 'atc_num_t' + str(k)] = tmp['atc_num'].sum() / days * k
                    data.loc[i, 'transrare_t' + str(k)] = tmp['transrare'].sum() / days * k
                    data.loc[i, 'pay_items_t' + str(k)] = tmp['pay_items'].sum() / days * k
                    data.loc[i, 'inv_num_t' + str(k)] = tmp['inv_num'].sum() / days * k
                    data.loc[i, 'discount_rate_t' + str(k)] = tmp['discount_rate'].sum() / days * k
                    data.loc[i, 'PMV_t' + str(k)] = tmp['PMV'].sum() / days * k

                elif days > 0:
                    dstart = data[data.data_date == data.loc[i, 'onshelf_date']].index[0]
                    tmp = data.loc[dstart:dend]
                    for k in [7, 30]:
                        data.loc[i, 'GMV_t' + str(k)] = sum(set(tmp['GMV'])) / days * k
                    tmp = tmp[tmp.spu == data.loc[i, 'spu']]
                    for k in [7, 30]:
                        data.loc[i, 'uv_t' + str(k)] = tmp['uv'].sum() / days * k
                        data.loc[i, 'collection_num_t' + str(k)] = tmp['collection_num'].sum() / days * k
                        data.loc[i, 'atc_num_t' + str(k)] = tmp['atc_num'].sum() / days * k
                        data.loc[i, 'transrare_t' + str(k)] = tmp['transrare'].sum() / days * k
                        data.loc[i, 'pay_items_t' + str(k)] = tmp['pay_items'].sum() / days * k
                        data.loc[i, 'inv_num_t' + str(k)] = tmp['inv_num'].sum() / days * k
                        data.loc[i, 'discount_rate_t' + str(k)] = tmp['discount_rate'].sum() / days * k
                        data.loc[i, 'PMV_t' + str(k)] = tmp['PMV'].sum() / days * k
                else:
                    for k in [7, 30]:
                        data.loc[i, 'uv_t' + str(k)] = \
                            data.loc[i, 'collection_num_t' + str(k)] = \
                            data.loc[i, 'atc_num_t' + str(k)] = \
                            data.loc[i, 'transrare_t' + str(k)] = \
                            data.loc[i, 'pay_items_t' + str(k)] = \
                            data.loc[i, 'inv_num_t' + str(k)] = \
                            data.loc[i, 'discount_rate_t' + str(k)] = \
                            data.loc[i, 'GMV_t' + str(k)] = \
                            data.loc[i, 'PMV_t' + str(k)] = 0
            if t_7_15_60:
                if days >= 60:
                    for k in [7, 15, 60]:
                        dstart = data[data.data_date == olddate[dateIndexInOlddate - k]].index[0]
                        tmp = data.loc[dstart:dend]
                        data.loc[i, 'GMV_t' + str(k)] = sum(set(tmp['GMV']))
                        tmp = tmp[tmp.spu == data.loc[i, 'spu']]

                        data.loc[i, 'uv_t' + str(k)] = tmp['uv'].sum()
                        data.loc[i, 'collection_num_t' + str(k)] = tmp['collection_num'].sum()
                        data.loc[i, 'atc_num_t' + str(k)] = tmp['atc_num'].sum()
                        data.loc[i, 'transrare_t' + str(k)] = tmp['transrare'].sum()
                        data.loc[i, 'pay_items_t' + str(k)] = tmp['pay_items'].sum()
                        data.loc[i, 'inv_num_t' + str(k)] = tmp['inv_num'].sum()
                        data.loc[i, 'discount_rate_t' + str(k)] = tmp['discount_rate'].sum()

                        data.loc[i, 'PMV_t' + str(k)] = tmp['PMV'].sum()

                elif days >= 15:
                    for k in [7, 15]:
                        dstart = data[data.data_date == olddate[dateIndexInOlddate - k]].index[0]
                        tmp = data.loc[dstart:dend]
                        data.loc[i, 'GMV_t' + str(k)] = sum(set(tmp['GMV']))
                        tmp = tmp[tmp.spu == data.loc[i, 'spu']]

                        data.loc[i, 'uv_t' + str(k)] = tmp['uv'].sum()
                        data.loc[i, 'collection_num_t' + str(k)] = tmp['collection_num'].sum()
                        data.loc[i, 'atc_num_t' + str(k)] = tmp['atc_num'].sum()
                        data.loc[i, 'transrare_t' + str(k)] = tmp['transrare'].sum()
                        data.loc[i, 'pay_items_t' + str(k)] = tmp['pay_items'].sum()
                        data.loc[i, 'inv_num_t' + str(k)] = tmp['inv_num'].sum()
                        data.loc[i, 'discount_rate_t' + str(k)] = tmp['discount_rate'].sum()
                        data.loc[i, 'PMV_t' + str(k)] = tmp['PMV'].sum()

                    dstart = data[data.data_date == data.loc[i, 'onshelf_date']].index[0]
                    tmp = data.loc[dstart:dend]
                    k = 60
                    data.loc[i, 'GMV_t' + str(k)] = sum(set(tmp['GMV'])) / days * k
                    tmp = tmp[tmp.spu == data.loc[i, 'spu']]
                    data.loc[i, 'uv_t' + str(k)] = tmp['uv'].sum() / days * k
                    data.loc[i, 'collection_num_t' + str(k)] = tmp['collection_num'].sum() / days * k
                    data.loc[i, 'atc_num_t' + str(k)] = tmp['atc_num'].sum() / days * k
                    data.loc[i, 'transrare_t' + str(k)] = tmp['transrare'].sum() / days * k
                    data.loc[i, 'pay_items_t' + str(k)] = tmp['pay_items'].sum() / days * k
                    data.loc[i, 'inv_num_t' + str(k)] = tmp['inv_num'].sum() / days * k
                    data.loc[i, 'discount_rate_t' + str(k)] = tmp['discount_rate'].sum() / days * k
                    data.loc[i, 'PMV_t' + str(k)] = tmp['PMV'].sum() / days * k
                elif days >= 7:
                    k = 7
                    dstart = data[data.data_date == olddate[dateIndexInOlddate - k]].index[0]
                    tmp = data.loc[dstart:dend]
                    data.loc[i, 'GMV_t7'] = sum(set(tmp['GMV']))
                    tmp = tmp[tmp.spu == data.loc[i, 'spu']]

                    data.loc[i, 'uv_t' + str(k)] = tmp['uv'].sum()
                    data.loc[i, 'collection_num_t' + str(k)] = tmp['collection_num'].sum()
                    data.loc[i, 'atc_num_t' + str(k)] = tmp['atc_num'].sum()
                    data.loc[i, 'transrare_t' + str(k)] = tmp['transrare'].sum()
                    data.loc[i, 'pay_items_t' + str(k)] = tmp['pay_items'].sum()
                    data.loc[i, 'inv_num_t' + str(k)] = tmp['inv_num'].sum()
                    data.loc[i, 'discount_rate_t' + str(k)] = tmp['discount_rate'].sum()
                    data.loc[i, 'PMV_t' + str(k)] = tmp['PMV'].sum()

                    dstart = data[data.data_date == data.loc[i, 'onshelf_date']].index[0]
                    tmp = data.loc[dstart:dend]
                    for k in [15, 60]:
                        data.loc[i, 'GMV_t' + str(k)] = sum(set(tmp['GMV'])) / days * k
                    tmp = tmp[tmp.spu == data.loc[i, 'spu']]
                    for k in [15, 60]:
                        data.loc[i, 'uv_t' + str(k)] = tmp['uv'].sum() / days * k
                        data.loc[i, 'collection_num_t' + str(k)] = tmp['collection_num'].sum() / days * k
                        data.loc[i, 'atc_num_t' + str(k)] = tmp['atc_num'].sum() / days * k
                        data.loc[i, 'transrare_t' + str(k)] = tmp['transrare'].sum() / days * k
                        data.loc[i, 'pay_items_t' + str(k)] = tmp['pay_items'].sum() / days * k
                        data.loc[i, 'inv_num_t' + str(k)] = tmp['inv_num'].sum() / days * k
                        data.loc[i, 'discount_rate_t' + str(k)] = tmp['discount_rate'].sum() / days * k
                        data.loc[i, 'PMV_t' + str(k)] = tmp['PMV'].sum() / days * k
                elif days > 0:
                    dstart = data[data.data_date == data.loc[i, 'onshelf_date']].index[0]
                    tmp = data.loc[dstart:dend]
                    for k in [7, 15, 60]:
                        data.loc[i, 'GMV_t' + str(k)] = sum(set(tmp['GMV'])) / days * k
                    tmp = tmp[tmp.spu == data.loc[i, 'spu']]
                    for k in [7, 15, 60]:
                        data.loc[i, 'uv_t' + str(k)] = tmp['uv'].sum() / days * k
                        data.loc[i, 'collection_num_t' + str(k)] = tmp['collection_num'].sum() / days * k
                        data.loc[i, 'atc_num_t' + str(k)] = tmp['atc_num'].sum() / days * k
                        data.loc[i, 'transrare_t' + str(k)] = tmp['transrare'].sum() / days * k
                        data.loc[i, 'pay_items_t' + str(k)] = tmp['pay_items'].sum() / days * k
                        data.loc[i, 'inv_num_t' + str(k)] = tmp['inv_num'].sum() / days * k
                        data.loc[i, 'discount_rate_t' + str(k)] = tmp['discount_rate'].sum() / days * k
                        data.loc[i, 'PMV_t' + str(k)] = tmp['PMV'].sum() / days * k
                else:
                    for k in [7, 15, 60]:
                        data.loc[i, 'uv_t' + str(k)] = \
                            data.loc[i, 'collection_num_t' + str(k)] = \
                            data.loc[i, 'atc_num_t' + str(k)] = \
                            data.loc[i, 'transrare_t' + str(k)] = \
                            data.loc[i, 'pay_items_t' + str(k)] = \
                            data.loc[i, 'inv_num_t' + str(k)] = \
                            data.loc[i, 'discount_rate_t' + str(k)] = \
                            data.loc[i, 'GMV_t' + str(k)] = \
                            data.loc[i, 'PMV_t' + str(k)] = 0
            # 生成t特征
            if purpose == 'predict':
                if days < t:
                    data.loc[i, 't'] = days
            # 生成库存周转率
            if days >= 7:
                dstart = data[data.data_date == olddate[dateIndexInOlddate - 7]].index[0]
                tmp = data.loc[dstart:dend]
                tmp = tmp[tmp.spu == data.loc[i, 'spu']]

                sm = tmp['pay_amount'].sum()
                if sm == 0:
                    data.loc[i, 'inv_trans'] = 1000
                else:
                    data.loc[i, 'inv_trans'] = data.loc[i, 'inv_num'] / sm
            else:
                if dend < dstart:
                    data.loc[i, 'inv_trans'] = 1000
                else:
                    dstart = data[data.data_date == data.loc[i, 'onshelf_date']].index[0]
                    tmp = data.loc[dstart:dend]
                    tmp = tmp[tmp.spu == data.loc[i, 'spu']]

                    sm = tmp['pay_amount'].sum()
                    if sm == 0:
                        data.loc[i, 'inv_trans'] = 1000
                    else:
                        data.loc[i, 'inv_trans'] = data.loc[i, 'inv_num'] / (sm / days * 7)
        else:
            if t_all or t_7_30 or t_7_15_60:
                if t_all:
                    k_value = [7, 15, 30, 60]
                if t_7_30:
                    k_value = [7, 30]
                if t_7_15_60:
                    k_value = [7, 15, 60]
                for k in k_value:
                    dstart = data[data.data_date == olddate[dateIndexInOlddate - k]].index[0]
                    tmp = data.loc[dstart:dend]
                    data.loc[i, 'GMV_t' + str(k)] = sum(set(tmp['GMV']))
                    tmp = tmp[tmp.spu == data.loc[i, 'spu']]

                    data.loc[i, 'uv_t' + str(k)] = tmp['uv'].sum()
                    data.loc[i, 'collection_num_t' + str(k)] = tmp['collection_num'].sum()
                    data.loc[i, 'atc_num_t' + str(k)] = tmp['atc_num'].sum()
                    data.loc[i, 'transrare_t' + str(k)] = tmp['transrare'].sum()
                    data.loc[i, 'pay_items_t' + str(k)] = tmp['pay_items'].sum()
                    data.loc[i, 'inv_num_t' + str(k)] = tmp['inv_num'].sum()
                    data.loc[i, 'discount_rate_t' + str(k)] = tmp['discount_rate'].sum()
                    data.loc[i, 'PMV_t' + str(k)] = tmp['PMV'].sum()
            # 生成库存周转率
            dstart = data[data.data_date == olddate[dateIndexInOlddate - 7]].index[0]
            tmp = data.loc[dstart:dend]
            tmp = tmp[tmp.spu == data.loc[i, 'spu']]
            sm = tmp['pay_amount'].sum()
            if sm == 0:
                data.loc[i, 'inv_trans'] = 1000
            else:
                data.loc[i, 'inv_trans'] = data.loc[i, 'inv_num'] / sm
    end = time.time()
    print('耗时：', round(end - start, 3), 's')
    print('-----------------------------------')

    print('生成相对值，百分位数')
    start = time.time()
    for each in ['uv', 'collection_num', 'atc_num', 'transrare', 'pay_items', 'inv_num', 'discount_rate', 'inv_trans']:
        add_feature(data, each, olddate)
    end = time.time()
    print('耗时：', round(end - start, 3), 's')
    print('-----------------------------------')

    columns = ['onshelf_date', 'data_date', 'spu',
               'uv', 'transrare', 'atc_num', 'collection_num',
               'GMV', 'discount_rate', 'pay_items', 'inv_num', 'PMV', 'inv_trans',
               'uv_rate', 'transrare_rate', 'atc_num_rate', 'collection_num_rate',
               'discount_rate_rate', 'pay_items_rate', 'inv_num_rate', 'inv_trans_rate',

               'uv_percentile', 'transrare_percentile', 'atc_num_percentile', 'collection_num_percentile',
               'discount_rate_percentile', 'pay_items_percentile', 'inv_num_percentile', 'inv_trans_percentile'

               ]
    if t_all:
        columns.extend(['uv_t7', 'transrare_t7', 'atc_num_t7', 'collection_num_t7',
                        'GMV_t7', 'discount_rate_t7', 'pay_items_t7', 'inv_num_t7', 'PMV_t7',

                        'uv_t15', 'transrare_t15', 'atc_num_t15', 'collection_num_t15',
                        'GMV_t15', 'discount_rate_t15', 'pay_items_t15', 'inv_num_t15', 'PMV_t15',

                        'uv_t30', 'transrare_t30', 'atc_num_t30', 'collection_num_t30',
                        'GMV_t30', 'discount_rate_t30', 'pay_items_t30', 'inv_num_t30', 'PMV_t30',

                        'uv_t60', 'transrare_t60', 'atc_num_t60', 'collection_num_t60',
                        'GMV_t60', 'discount_rate_t60', 'pay_items_t60', 'inv_num_t60', 'PMV_t60'])
    if t_7_30:
        columns.extend(['uv_t7', 'transrare_t7', 'atc_num_t7', 'collection_num_t7',
                        'GMV_t7', 'discount_rate_t7', 'pay_items_t7', 'inv_num_t7', 'PMV_t7',
                        'uv_t30', 'transrare_t30', 'atc_num_t30', 'collection_num_t30',
                        'GMV_t30', 'discount_rate_t30', 'pay_items_t30', 'inv_num_t30', 'PMV_t30',
                        ])
    if t_7_15_60:
        columns.extend(['uv_t7', 'transrare_t7', 'atc_num_t7', 'collection_num_t7',
                        'GMV_t7', 'discount_rate_t7', 'pay_items_t7', 'inv_num_t7', 'PMV_t7',

                        'uv_t15', 'transrare_t15', 'atc_num_t15', 'collection_num_t15',
                        'GMV_t15', 'discount_rate_t15', 'pay_items_t15', 'inv_num_t15', 'PMV_t15',

                        'uv_t60', 'transrare_t60', 'atc_num_t60', 'collection_num_t60',
                        'GMV_t60', 'discount_rate_t60', 'pay_items_t60', 'inv_num_t60', 'PMV_t60'])

    columns.extend(['t', 'heat', 'cer'])
    new_data = data.loc[start_index:, columns]
    new_data.index = range(len(new_data))
    if outfile:
        print('开始导出新文件...')
        start = time.time()
        new_data.to_csv(outfile, encoding=coding, index=None)
        end = time.time()
        print('结束，用时：', round(end - start, 3))

    te = time.time()
    print('\n总时间：', round(te - ts, 3), 's')
    return new_data


if __name__ == '__main__':
    data = pandas.read_csv('bb.csv', encoding=coding)
    old_date = data['data_date'].tolist()
    old_date_without_repeat = []
    for each in old_date:
        if each not in old_date_without_repeat:
            old_date_without_repeat.append(each)
    new_date = old_date_without_repeat[60:]
    process('bb.csv', old_date_without_repeat, new_date, outfile='middle.csv')
