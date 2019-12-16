import pandas
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz
import pydotplus
import matplotlib.pyplot as plt
import time
import pickle
import os
import dataprocess2
import r2_adjust
coding = 'utf8'


# graph learning curve
def learning_curve(xt, yt, name, depths, outfile=None, show=True):
    print('构建', name, '学习曲线...')
    start = time.time()
    rate = [i * 0.1 for i in range(1, 10)]
    plt.figure(figsize=(13, 8))
    plt.suptitle(name, fontsize='xx-large', fontweight='black', backgroundcolor='y', x=0.05)
    for i in range(4):
        y_te = []
        y_tr = []
        for each in rate:
            xt_sub, xt_drop, yt_sub, yt_drop = train_test_split(
                xt, yt, train_size=each, random_state=0
            )
            rt = DecisionTreeRegressor(max_depth=depths[i])
            rt.fit(xt_sub, yt_sub)
            train_score = rt.score(xt_sub, yt_sub)
            test_score = cross_val_score(rt, xt_sub, yt_sub, cv=10, scoring='r2').mean()
            y_tr.append(train_score)
            y_te.append(test_score)

        plt.subplot(2, 2, i + 1)
        plt.plot(rate, y_tr, 'o-', color='red', label='train')
        plt.plot(rate, y_te, 'o-', color='g', label='verification')
        plt.title("max_depth=" + str(depths[i]))

        plt.xlabel("scale of training set")
        plt.ylabel("r2 score")
        plt.legend(loc='lower right')
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
    if show:
        plt.show()
    end = time.time()
    print('用时：', round(end - start, 3), 's')


# graph complexity curve
def complexity_curve(xt, yt, name, depths, outfile=None, show=True):
    print('构建', name, '复杂度曲线...')
    start = time.time()
    y_te = []
    y_tr = []
    for each in depths:
        rt = DecisionTreeRegressor(max_depth=each)
        rt.fit(xt, yt)
        train_score = rt.score(xt, yt)
        test_score = cross_val_score(rt, xt, yt, cv=10, scoring='r2').mean()
        y_tr.append(train_score)
        y_te.append(test_score)
    plt.suptitle(name, fontsize='xx-large', fontweight='black', backgroundcolor='y', x=0.05)
    plt.plot(depths, y_tr, 'o-', color='red', label='train')
    plt.plot(depths, y_te, 'o-', color='g', label='verification')
    plt.title('Decision Tree Regressor Complexity Performance')
    plt.xlabel("maximum depth")
    plt.ylabel("r2 score")
    plt.legend(loc='lower right')
    if outfile:
        plt.savefig(outfile)
    if show:
        plt.show()
    end = time.time()
    print('用时：', round(end - start, 3), 's')


# by this function, we can get the best parameters from certain range
def parameters_select(x, y, para):
    print('参数搜索...')
    start = time.time()
    rtree = DecisionTreeRegressor()
    gs = GridSearchCV(rtree, para, cv=10, scoring='r2')
    gs.fit(x, y)
    for key, para in gs.best_params_.items():
        print('最优%s：%s' % (key, para))
    print('最好得分：', round(gs.best_score_, 4), '\n',
          '最优模型的叶节点数：', gs.best_estimator_.get_n_leaves(),
          '\n', '最优模型的深度：', gs.best_estimator_.get_depth())
    end = time.time()
    print('用时：', round(end - start, 3), 's')
    return gs


# visualize tree
def tree_graph(tree, outfile, features):
    print('画出决策树...')
    start = time.time()
    dot_data_simple = export_graphviz(tree, filled=True, rounded=True, feature_names=features,
                                      impurity=False, proportion=True)
    dot_data = export_graphviz(tree, filled=True, rounded=True, feature_names=features)
    graph = pydotplus.graph_from_dot_data(dot_data_simple)
    graph.get_node("node")[0].set_fontname(
        "Microsoft YaHei")
    graph.write_png(outfile[0:outfile.rfind('.')] + '_simple.png')
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.get_node("node")[0].set_fontname(
        "Microsoft YaHei")
    graph.write_png(outfile)
    # graph.write_pdf(outfile[0:outfile.index('.')]+'.pdf')
    end = time.time()
    print('用时：', round(end - start, 3), 's')


# save model in path
def save_model(path, model):
    f = open(path, 'wb')
    pickle.dump(model, f)
    f.close()


def train(train_data_path, result_dir, columns_heat, columns_cer, model_save_dir,
          heat_model_max_depth=None,
          heat_model_max_leaf_nodes=None,
          cer_model_max_depth=None,
          cer_model_max_leaf_nodes=None,
          outfilewithnewfeature=None,
          dealt=None,
          learning_depths_heat=None,
          learning_depths_cer=None,
          complexity_depths_heat=None,
          complexity_depths_cer=None,
          grid_search_heat=None,
          grid_search_cer=None,
          t_7_30=False,
          t_7_15_60=False,
          t_all=False
          ):
    '''
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
        热度模型学习曲线深度取值，为一个四元组，如[9, 12, 15, 18]
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
    '''
    if not dealt:
        # 构建特征值
        data = dataprocess2.process(train_data_path, 'train', outfilewithnewfeature,
                                    t_all=t_all, t_7_30=t_7_30, t_7_15_60=t_7_15_60)
    else:
        data = pandas.read_csv(dealt, encoding=coding)

    X_heat = data[columns_heat]
    y_heat = data['heat']
    X_cer = data[columns_cer]
    y_cer = data['cer']
    # X = Imputer().fit_transform(X)

    # 拆分成训练集和测试集
    X_heat_train, X_heat_test, y_heat_train, y_heat_test = train_test_split(
        X_heat, y_heat, test_size=0.3, random_state=0
    )
    X_cer_train, X_cer_test, y_cer_train, y_cer_test = train_test_split(
        X_cer, y_cer, test_size=0.3, random_state=0
    )
    train_sample_count = X_heat_train.shape[0]
    test_sample_count = X_heat_test.shape[0]
    heat_feature_count = X_heat_train.shape[1]
    cer_feature_count = X_cer_train.shape[1]

    # 交叉验证评估模型能力
    # print('评估模型能力..')
    # reg_tree = DecisionTreeRegressor()
    # scores = cross_val_score(reg_tree, X_heat_train, y_heat_train, cv=10, scoring='r2')
    # print(scores)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # select best max_depth
    if grid_search_heat:
        print('热度预测模型参数搜索..........')
        parameters_heat = grid_search_heat
        heat_gs = parameters_select(X_heat_train, y_heat_train, parameters_heat)
    if grid_search_cer:
        print('效率预测模型参数搜索..........')
        parameters_cer = grid_search_cer
        cer_gs = parameters_select(X_cer_train, y_cer_train, parameters_cer)
    if grid_search_heat and grid_search_cer:
        return

    # 学习曲线和复杂度曲线
    if learning_depths_heat:
        learning_curve(X_heat_train, y_heat_train, 'heat', learning_depths_heat,
                       outfile=os.path.join(result_dir, 'l_heat.png'))
    if learning_depths_cer:
        learning_curve(X_cer_train, y_cer_train, 'cer', learning_depths_cer,
                       outfile=os.path.join(result_dir, 'l_cer.png'))
    if complexity_depths_heat:
        complexity_curve(X_heat_train, y_heat_train, 'heat', complexity_depths_heat,
                         outfile=os.path.join(result_dir, 'c_heat.png'))
    if complexity_depths_cer:
        complexity_curve(X_cer_train, y_cer_train, 'cer', complexity_depths_cer,
                         outfile=os.path.join(result_dir, 'c_cer.png'))

    # 训练模型
    print('开始训练热度...')
    start = time.time()
    # reg_tree_heat = sklearn.tree.DecisionTreeRegressor(max_leaf_nodes=heat_gs.best_params_['max_leaf_nodes'],
    #                                                    max_depth=heat_gs.best_params_['max_depth'])
    # reg_tree_heat = sklearn.tree.DecisionTreeRegressor(max_leaf_nodes=160, max_depth=11,
    #                                                    min_samples_split=0.0001)
    reg_tree_heat = sklearn.tree.DecisionTreeRegressor(max_leaf_nodes=heat_model_max_leaf_nodes,
                                                       max_depth=heat_model_max_depth)
    reg_tree_heat.fit(X_heat_train, y_heat_train)
    end = time.time()
    print('用时：', round(end - start, 3), 's')
    print('热度回归树模型········')
    print('深度：', reg_tree_heat.get_depth())
    print('叶子节点数：', reg_tree_heat.get_n_leaves())
    print('训练得分: ', round(r2_adjust.r2_adjust(reg_tree_heat.score(X_heat_train, y_heat_train),
                                              train_sample_count, heat_feature_count), 4))
    print('开始测试...')
    start = time.time()
    print('测试得分: ', round(r2_adjust.r2_adjust(reg_tree_heat.score(X_heat_test, y_heat_test),
                                              test_sample_count, heat_feature_count), 4))
    end = time.time()
    print('用时：', round(end - start, 3), 's')

    print('开始训练效率...')
    start = time.time()
    # reg_tree_cer = sklearn.tree.DecisionTreeRegressor(max_leaf_nodes=cer_gs.best_params_['max_leaf_nodes'],
    #                                                   max_depth=cer_gs.best_params_['max_depth'])
    # reg_tree_cer = sklearn.tree.DecisionTreeRegressor(max_leaf_nodes=240, max_depth=13,
    #                                                   min_samples_split=0.0001)
    reg_tree_cer = sklearn.tree.DecisionTreeRegressor(max_leaf_nodes=cer_model_max_leaf_nodes,
                                                      max_depth=cer_model_max_depth)
    reg_tree_cer.fit(X_cer_train, y_cer_train)
    end = time.time()
    print('用时：', round(end - start, 3), 's')
    print('效率回归树模型········')
    print('深度：', reg_tree_cer.get_depth())
    print('叶子节点数：', reg_tree_cer.get_n_leaves())
    print('训练得分: ', round(r2_adjust.r2_adjust(reg_tree_cer.score(X_cer_train, y_cer_train),
                                              train_sample_count, cer_feature_count), 4))
    print('开始测试...')
    start = time.time()
    print('测试得分: ', round(r2_adjust.r2_adjust(reg_tree_cer.score(X_cer_test, y_cer_test),
                                              test_sample_count, cer_feature_count), 4))
    end = time.time()
    print('用时：', round(end - start, 3), 's')

    # 保存模型
    save_model(os.path.join(model_save_dir, 'reg_tree_heat.pickle'), reg_tree_heat)
    save_model(os.path.join(model_save_dir, 'reg_tree_cer.pickle'), reg_tree_cer)

    # 模型可视化
    tree_graph(reg_tree_heat, os.path.join(result_dir, 'reg_tree_heat.png'), columns_heat)
    tree_graph(reg_tree_cer, os.path.join(result_dir, 'reg_tree_cer.png'), columns_cer)
