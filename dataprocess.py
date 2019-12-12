import pandas
import os
import time
coding = 'utf8'

def process(olddatapath, olddate, newdate, outfile=None):
    start = end = 0
    print('开始读取数据...')
    start = time.time()
    data_path = olddatapath

    data = pandas.read_csv(data_path, encoding=coding, engine='python')
    end = time.time()
    print('结束，用时：', round(end - start, 3))



    print('开始增加transrare特征...')
    start = time.time()
    data['transrare'] = 0
    valid = data[(data.uv != 0) & (data.pay_custs != 0)].index.tolist()
    data.loc[valid, 'transrare'] = data.loc[valid, 'pay_custs'] / data.loc[valid, 'uv']
    end = time.time()
    print('结束，用时：', round(end - start, 3))

    print('开始增加GMV特征...')
    start = time.time()

    for each in olddate:
        aday = data[data.data_date == each].index.tolist()
        data.loc[aday, 'GMV'] = data.loc[aday, 'pay_amount'].sum()
    end = time.time()
    print('结束，用时：', round(end - start, 3))

    print('处理售价...')
    start = time.time()
    sale_0 = data[data.sales_amount == 0].index.tolist()
    data.loc[sale_0, 'sales_amount'] = data.loc[sale_0, 'retail_amount']
    end = time.time()
    print('结束，用时：', round(end - start, 3))

    print('开始增加discount_rate和PMV特征...')
    start = time.time()

    data = data[data.retail_amount != 0]
    data.index = range(len(data))
    data['discount_rate'] = data['sales_amount'] / data['retail_amount']
    data['PMV'] = data['pay_amount'] / data['GMV']
    data['PMV'] = data['PMV'].fillna(0)
    end = time.time()
    print('结束，用时：', round(end - start, 3))

    print('开始增加t7,t30特征...')
    start = time.time()
    start_index = data[data.data_date == newdate[0]].index.tolist()[0]
    ########################################
    # data_date_old = data.loc[start_index:start_index+2999, 'data_date'].tolist()
    # for i in range(start_index, start_index+3000):
    age_of_spu = {}
    for i in range(start_index, len(data)):  #len(data)
###################################################
        data_same_item = data[data.spu == data.loc[i, 'spu']]
        inds = data_same_item.index.tolist()
        i_in_inds = inds.index(i)
        pre_day_date = data.loc[i-1, 'data_date']
        this_day_date = data.loc[i, 'data_date']        
        if this_day_date != pre_day_date:
            a_day = data[data.data_date == data.loc[i, 'data_date']].index
            dates = data['data_date'].tolist()
            dates_without_repeat = []
            for each in dates:
                if each not in dates_without_repeat:
                    dates_without_repeat.append(each)
            j = dates_without_repeat.index(this_day_date)-1
            GMVs = []
            while j>=0 and len(GMVs)<=30:
                GMVs.append(data[data.data_date == dates_without_repeat[j]].iloc[0]['GMV'])
                j -= 1
            if len(GMVs)>=30:
                data.loc[a_day, 'GMV_t7'] = sum(GMVs[:7])
                data.loc[a_day, 'GMV_t30'] = sum(GMVs[:30])
            elif len(GMVs)>=7:
                data.loc[a_day, 'GMV_t7'] = sum(GMVs[:7])
                data.loc[a_day, 'GMV_t30'] = sum(GMVs)/len(GMVs)*30
            elif len(GMVs)>0:
                data.loc[a_day, 'GMV_t7'] = sum(GMVs)/len(GMVs)*7
                data.loc[a_day, 'GMV_t30'] = sum(GMVs)/len(GMVs)*30
            
        if i_in_inds >= 30:
            data.loc[i, 'uv_t7'] = data_same_item.loc[inds[i_in_inds-7:i_in_inds], 'uv'].sum()
            data.loc[i, 'discount_rate_t7'] = data_same_item.loc[inds[i_in_inds-7:i_in_inds], 'discount_rate'].sum()
            data.loc[i, 'pay_items_t7'] = data_same_item.loc[inds[i_in_inds-7:i_in_inds], 'pay_items'].sum()
            data.loc[i, 'uv_t30'] = data_same_item.loc[inds[i_in_inds-30:i_in_inds], 'uv'].sum()
            data.loc[i, 'discount_rate_t30'] = data_same_item.loc[inds[i_in_inds-30:i_in_inds], 'discount_rate'].sum()
            data.loc[i, 'pay_items_t30'] = data_same_item.loc[inds[i_in_inds-30:i_in_inds], 'pay_items'].sum()
        elif i_in_inds >= 7:
            data.loc[i, 'uv_t7'] = data_same_item.loc[inds[i_in_inds-7:i_in_inds], 'uv'].sum()
            data.loc[i, 'discount_rate_t7'] = data_same_item.loc[inds[i_in_inds-7:i_in_inds], 'discount_rate'].sum()
            data.loc[i, 'pay_items_t7'] = data_same_item.loc[inds[i_in_inds-7:i_in_inds], 'pay_items'].sum()
            data.loc[i, 'uv_t30'] = (data_same_item.loc[inds[0:i_in_inds], 'uv'].sum() / i_in_inds) * 30
            data.loc[i, 'discount_rate_t30'] = (data_same_item.loc[inds[0:i_in_inds], 'discount_rate'].sum() / i_in_inds) * 30
            data.loc[i, 'pay_items_t30'] = (data_same_item.loc[inds[0:i_in_inds], 'pay_items'].sum() / i_in_inds) * 30
        elif i_in_inds > 0:
            a = data.loc[i, 'uv_t7'] = (data_same_item.loc[inds[0:i_in_inds], 'uv'].sum() / i_in_inds) * 7
            data.loc[i, 'discount_rate_t7'] = (data_same_item.loc[inds[0:i_in_inds], 'discount_rate'].sum() / i_in_inds) * 7
            data.loc[i, 'pay_items_t7'] = (data_same_item.loc[inds[0:i_in_inds], 'pay_items'].sum() / i_in_inds) * 7
            data.loc[i, 'uv_t30'] = (data_same_item.loc[inds[0:i_in_inds], 'uv'].sum() / i_in_inds) * 30
            data.loc[i, 'discount_rate_t30'] = (data_same_item.loc[inds[0:i_in_inds], 'discount_rate'].sum() / i_in_inds) * 30
            data.loc[i, 'pay_items_t30'] = (data_same_item.loc[inds[0:i_in_inds], 'pay_items'].sum() / i_in_inds) * 30
        else:
            data.loc[i, 'uv_t7'] = \
            data.loc[i, 'discount_rate_t7'] = \
            data.loc[i, 'pay_items_t7'] = \
            data.loc[i, 'uv_t30'] = \
            data.loc[i, 'discount_rate_t30'] = \
            data.loc[i, 'pay_items_t30'] = 0

    
    end = time.time()

    print('结束，用时：', round(end - start, 3))
    columns = ['data_date', 'spu', 'uv', 'transrare', 'atc_num', 'collection_num', 'GMV', 'discount_rate',
               'pay_items', 'inv_num', 'PMV', 'division',
               'uv_t7', 'GMV_t7', 'discount_rate_t7', 'pay_items_t7',
               'uv_t30', 'GMV_t30', 'discount_rate_t30', 'pay_items_t30',
               'heat', 'cer']

    #############################################
    new_data = data.loc[start_index:, columns]
    # new_data = data.loc[start_index:start_index+2999, columns]
    ################################################

    new_data.index = range(len(new_data))
    # new_data = pandas.get_dummies(new_data, prefix='division', prefix_sep='=', columns=['division'], drop_first=True)

    cols = ['pay_items', 'inv_num', 'uv', 'collection_num', 'uv_t7',
            'pay_items_t7', 'pay_items_t30', 'uv_t30']
    for each in newdate:
        index_in_same_day = new_data[new_data.data_date == each].index.tolist()
        for each_col in cols:
            s = new_data.loc[index_in_same_day, each_col].sum()
            if s:
                new_data.loc[index_in_same_day, each_col+'_rate'] = \
                                        new_data.loc[index_in_same_day, each_col] / s
            else:
                new_data.loc[index_in_same_day, each_col + '_rate'] = 0

    new_data = pandas.get_dummies(new_data, columns=['division'], prefix_sep='=', drop_first=True)
    if outfile:
        print('开始导出新文件...')
        start = time.time()
        new_data.to_csv(outfile, encoding=coding, index=None)
        end = time.time()
        print('结束，用时：', round(end - start, 3))
    return new_data


if __name__ == '__main__':
    '''
    old_data_path = os.path.join(os.getcwd(), 'data', '20190701-20190914(1)withoutnewitems.csv')
    new_data_path = os.path.join(os.getcwd(), 'data', 'newdata.csv')
    olddate = ['2019/7/' + str(i) for i in range(1, 32)]
    olddate.extend(['2019/8/' + str(i) for i in range(1, 32)])
    olddate.extend(['2019/9/' + str(i) for i in range(1, 15)])

    newdate = (['2019/8/' + str(i) for i in range(4, 32)])
    newdate.extend(['2019/9/' + str(i) for i in range(1, 15)])
    new_data = process(old_data_path, olddate, newdate)
    print('开始导出新文件...')
    start = time.time()
    new_data.to_csv(new_data_path, encoding=coding, index=None)
    end = time.time()
    print('结束，用时：', round(end - start, 3))
    '''
    old_data_path = 'idxwithoutnewitems.csv'
    data = pandas.read_csv(old_data_path)
    old_date = data['data_date'].tolist()
    old_date_without_repeat = []
    for each in old_date:
        if each not in old_date_without_repeat:
            old_date_without_repeat.append(each)
    new_date = [old_date_without_repeat[-1]]
    process(old_data_path, old_date_without_repeat, new_date)

