# %% [markdown]
# ### 数据描述（**北方华创**）[`002371`]
# Stkcd [证券代码] - 以上交所、深交所公布的证券代码为准\
# Trddt [交易日期] - 以YYYY-MM-DD表示\
# Opnprc [日开盘价] - A股以人民币元计，上海B以美元计，深圳B以港币计\
# Hiprc [日最高价] - A股以人民币元计，上海B以美元计，深圳B以港币计\
# Loprc [日最低价] - A股以人民币元计，上海B以美元计，深圳B以港币计\
# Clsprc [日收盘价] - A股以人民币元计，上海B以美元计，深圳B以港币计\
# Dnshrtrd [日个股交易股数] - 0=没有交易量\
# Dnvaltrd [日个股交易金额] - A股以人民币元计，上海B以美元计，深圳B以港币计，0=没有交易量

# %% [markdown]
# ##### 按照授课教师指定的上海证券交易所和深圳证券交易所的1家上市公司，下载2018年10月1日-2023年9月30日的日度交易行情信息：开盘价、最高价、最低价、收盘价和成交量，进行下列数据分析实验：
# (1)	画出收盘价的时间序列图。（10分）\
# (2)	将收盘价高于开盘价的交易日用1进行标注，其他用0进行标注，给数据增加标签列。（10分）\
# (3)	用收盘价计算日度收益率，增加日度收益率这个字段。对数据进行预处理，检查数据是否存在缺失值、无穷大值，将缺失值和无穷大值用0代替。（10分）\
# (4)	画出股票日度收益的直方图，分析股票收益分布的特征。（10分）\
# (5)	基于7:3的训练集/测试集切分，使用上一个交易日或以前信息，分别用K近邻、决策树、朴素贝叶斯、人工神经网络对下一个交易日股票收盘价是否会高于开盘价进行预测，将收盘价高于开盘价的交易日定义为正类，计算召回率、精度、准确率。要求：除使用下载数据字段作为属性外，自己另外构建2个属性用于预测；说明超参数选择，用测试集展示预测效果。（40分）\
# (6)	基于K近邻、决策树、朴素贝叶斯、人工神经网络构建集成分类器，用硬投票对按时间顺序切分的训练集/测试集进行类别预测，并与随机森林预测效果对比。（20分）

# %%
# 导入所需要的包
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.model_selection import GridSearchCV

# %%
# 读取股票数据（北方华创，股票代码：002371）
data = pd.read_csv('./TRD_Dalyr.csv', encoding='utf-8')
data.head()

# %%
# 其中
# Stkcd：code
# Trddt：date
# Opnprc：open
# Hiprc：high
# Loprc：low
# Clsprc：close
# Dnshrtrd：volume
renames = {
    'Stkcd': 'code',
    'Trddt': 'date',
    'Opnprc': 'open',
    'Hiprc': 'high',
    'Loprc': 'low',
    'Clsprc': 'close',
    'Dnshrtrd': 'volume',
}
data.rename(columns=renames, inplace=True)

# %%
# 重新设置列名并且删除无用的列
data.index = pd.to_datetime(data['date'])
data.drop(['code', 'date'], axis=1, inplace=True)
data.head()

# %%
# 查看数据的基本信息
data.info()

# %%
# (1)画出收盘价的时间序列图。（10分）
data['close'].plot(figsize=(12, 6), title='the time series plot of close price ', fontsize=15)
plt.xlabel('date', fontsize=15)
plt.ylabel('close price', fontsize=15)
plt.show()

# %%
# (2)将收盘价高于开盘价的交易日用1进行标注，其他用0进行标注，给数据增加标签列。（10分）
data['label'] = np.where(data['close'] > data['open'], 1, 0)
data.head(10)

# %%
# (3)用收盘价计算日度收益率，增加日度收益率这个字段。对数据进行预处理，检查数据是否存在缺失值、无穷大值，将缺失值和无穷大值用0代替。（10分）
data['return'] = data['close'].pct_change()
# 查看是否有缺失值和无穷大值
print('无无穷大值' if data['return'].isin([np.inf, -np.inf]).sum() == 0 else '有无穷大值')
print('无缺失值' if data['return'].isnull().sum() == 0 else '有缺失值')

# %%
# 从结果来看，数据中有缺失但是没有无穷大值
# 从info中来看，没有缺失值，但是现在却出现了缺失值，说明缺失值是在计算收益率的时候产生的
# 因此我们采用0来填充缺失值
data['return'].fillna(0, inplace=True)
data.head(10)

# %%
# (4)画出股票日度收益的直方图，分析股票收益分布的特征。（10分）
# 绘制直方图
plt.figure(figsize=(12,6))
ax1 = plt.subplot(121)
bins=50
n,bins,patches = ax1.hist(data['return'],alpha=0.5, bins=bins, density=True,label='return')

# 绘制正态分布曲线
mean = data['return'].mean()
std = data['return'].std()
y = norm.pdf(bins, mean, std)
ax1.plot(bins, y, 'r--', label='normal distribution')
ax1.set_title('Histogram of return', fontsize=15)

# 绘制QQ图
import statsmodels.api as sm
ax2 = plt.subplot(122)
sm.qqplot((data['return']-mean)/std, ax=ax2, line='45',alpha=0.5)
plt.tight_layout()
plt.show()

# %%
# 拟合t分布
params = stats.t.fit(data['return'])

# 生成t分布的概率密度函数 (PDF)
x = np.linspace(min(data['return']), max(data['return']), 100)
fitted_pdf = stats.t.pdf(x, *params)

# 绘制原始数据的直方图和拟合的t分布
plt.hist(data['return'], density=True, alpha=0.5, bins=bins, label='Data')
plt.plot(x, fitted_pdf, 'r-', label='t distribution')
plt.legend()

plt.show()

# %% [markdown]
# #### 从结果来看，收益分布更加倾向于t分布

# %%
'''
(5)	基于7:3的训练集/测试集切分，使用上一个交易日或以前信息，
分别用K近邻、决策树、朴素贝叶斯、人工神经网络对下一个交易日股票收盘价是否会高于开盘价进行预测，
将收盘价高于开盘价的交易日定义为正类，计算召回率、精度、准确率。
要求：除使用下载数据字段作为属性外，自己另外构建2个属性用于预测；说明超参数选择，用测试集展示预测效果。（40分）
'''
# 构建属性'放量情况'，'日内涨幅'，并处理缺失值和无穷值
data['volume_change'] = data['volume'].pct_change()
data['volume_change'].fillna(0, inplace=True)
data['volume_change'].replace([np.inf, -np.inf], 0, inplace=True)
data['close_open_change'] = (data['close'] - data['open']) / data['open']
data['close_open_change'].fillna(0, inplace=True)
data['close_open_change'].replace([np.inf, -np.inf], 0, inplace=True)
data.head()

# %%
# 构建y
y = data['label'][1:]
# 构建X
X = data.drop(['label'], axis=1)[:-1]

# %%
# 划分训练集和测试集
# 时间序列数据不能随机划分，只能按照时间顺序划分
train_x = X[:int(len(X) * 0.7)]
train_y = y[:int(len(y) * 0.7)]
test_x = X[int(len(X) * 0.7):]
test_y = y[int(len(y) * 0.7):]

# %%
# 储存结果
compare = pd.DataFrame(columns=['recall', 'precision', 'accuracy_on_test', 'accuracy_on_train'])

# %%
# KNN
from sklearn.neighbors import KNeighborsClassifier

# 训练模型
n = 15
knn = KNeighborsClassifier(n_neighbors=n)
knn.fit(train_x, train_y)
y_pred = knn.predict(test_x)

# 计算召回率、精度、准确率
compare.loc['KNN'] = [recall_score(test_y, y_pred), precision_score(test_y, y_pred),
                      accuracy_score(test_y, y_pred), accuracy_score(train_y, knn.predict(train_x))]
print('召回率：', recall_score(test_y, y_pred))
print('精度：', precision_score(test_y, y_pred))
print('准确率：', accuracy_score(test_y, y_pred))

# %%
# bestKNN
# 寻找最优的K
# 设置需要搜索的K值
k_range = [i for i in range(1, 31)]
# 设置网格参数
param_grid = dict(n_neighbors=k_range)
# 设置网格搜索
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
# 训练模型
grid.fit(train_x, train_y)
# 输出最优的K值
print('最优的K值为：', grid.best_params_['n_neighbors'])
# 保存最优参数
best_knn_params = grid.best_params_

# 查看最优模型的召回率、精度、准确率
y_pred = grid.predict(test_x)
compare.loc['bestKNN'] = [recall_score(test_y, y_pred), precision_score(test_y, y_pred),
                          accuracy_score(test_y, y_pred), accuracy_score(train_y, grid.predict(train_x))]
print('召回率：', recall_score(test_y, y_pred))
print('精度：', precision_score(test_y, y_pred))
print('准确率：', accuracy_score(test_y, y_pred))

# %%
# 决策树
from sklearn.tree import DecisionTreeClassifier

# 训练模型
dtc = DecisionTreeClassifier()
dtc.fit(train_x, train_y)
y_pred = dtc.predict(test_x)

# 计算召回率、精度、准确率
compare.loc['DecisionTree'] = [recall_score(test_y, y_pred), precision_score(test_y, y_pred),
                               accuracy_score(test_y, y_pred), accuracy_score(train_y, dtc.predict(train_x))]
print('召回率：', recall_score(test_y, y_pred))
print('精度：', precision_score(test_y, y_pred))
print('准确率：', accuracy_score(test_y, y_pred))

# %%
# bestDTC
# 寻找最优的max_depth, min_samples_split, min_samples_leaf
# 设置需要搜索的参数
param_grid = {'max_depth': range(1, 10), 'min_samples_split': range(2, 10), 'min_samples_leaf': range(1, 10)}
# 设置网格搜索
grid = GridSearchCV(dtc, param_grid, cv=10, scoring='accuracy')
# 训练模型
grid.fit(train_x, train_y)
# 输出最优的参数
print('最优的max_depth为：', grid.best_params_['max_depth'])
print('最优的min_samples_split为：', grid.best_params_['min_samples_split'])
print('最优的min_samples_leaf为：', grid.best_params_['min_samples_leaf'])
# 保存最优的参数
best_dtc_params = grid.best_params_

# 查看最优模型的召回率、精度、准确率
y_pred = grid.predict(test_x)
compare.loc['bestDecisionTree'] = [recall_score(test_y, y_pred), precision_score(test_y, y_pred),
                                   accuracy_score(test_y, y_pred), accuracy_score(train_y, grid.predict(train_x))]
print('召回率：', recall_score(test_y, y_pred))
print('精度：', precision_score(test_y, y_pred))
print('准确率：', accuracy_score(test_y, y_pred))

# %%
# 朴素贝叶斯
from sklearn.naive_bayes import GaussianNB

# 训练模型
gnb = GaussianNB()
gnb.fit(train_x, train_y)
y_pred = gnb.predict(test_x)

# 计算召回率、精度、准确率
compare.loc['GaussianNB'] = [recall_score(test_y, y_pred), precision_score(test_y, y_pred),
                             accuracy_score(test_y, y_pred), accuracy_score(train_y, gnb.predict(train_x))]
print('召回率：', recall_score(test_y, y_pred))
print('精度：', precision_score(test_y, y_pred))
print('准确率：', accuracy_score(test_y, y_pred))

# %%
# bestGNB
# 朴素贝叶斯没有超参数需要调整，因此不需要进行网格搜索

# %%
# 人工神经网络
from sklearn.neural_network import MLPClassifier

# 训练模型
mlp = MLPClassifier(activation='relu',learning_rate='adaptive',max_iter=1000,hidden_layer_sizes=(100,10))
mlp.fit(train_x, train_y)
y_pred = mlp.predict(test_x)

# 计算召回率、精度、准确率
compare.loc['MLP'] = [recall_score(test_y, y_pred), precision_score(test_y, y_pred),
                      accuracy_score(test_y, y_pred), accuracy_score(train_y, mlp.predict(train_x))]
print('召回率：', recall_score(test_y, y_pred))
print('精度：', precision_score(test_y, y_pred))
print('准确率：', accuracy_score(test_y, y_pred))

# %%
# bestMLP
# 寻找最优的activation, learning_rate, hidden_layer_sizes
# 设置需要搜索的参数
params_grid = {'activation': ['identity', 'logistic', 'tanh', 'relu'],
                'learning_rate': ['constant', 'invscaling', 'adaptive'],
                'hidden_layer_sizes': [(100, 10), (100, 20), (100, 30), (100, 40), (100, 50)]}
# 设置网格搜索
grid = GridSearchCV(mlp, params_grid, cv=10, scoring='accuracy')
# 训练模型
grid.fit(train_x, train_y)
# 输出最优的参数
print('最优的activation为：', grid.best_params_['activation'])
print('最优的learning_rate为：', grid.best_params_['learning_rate'])
print('最优的hidden_layer_sizes为：', grid.best_params_['hidden_layer_sizes'])
# 保存最优的参数
best_mlp_params = grid.best_params_

# 查看最优模型的召回率、精度、准确率
y_pred = grid.predict(test_x)
compare.loc['bestMLP'] = [recall_score(test_y, y_pred), precision_score(test_y, y_pred),
                          accuracy_score(test_y, y_pred), accuracy_score(train_y, grid.predict(train_x))]
print('召回率：', recall_score(test_y, y_pred))
print('精度：', precision_score(test_y, y_pred))
print('准确率：', accuracy_score(test_y, y_pred))

# %%
compare

# %%
# (6)	基于K近邻、决策树、朴素贝叶斯、人工神经网络构建集成分类器，\
# 用硬投票对按时间顺序切分的训练集/测试集进行类别预测，并与随机森林预测效果对比。（20分）

# 集成分类器
from sklearn.ensemble import VotingClassifier

# 构建集成分类器
bestKNN = KNeighborsClassifier(n_neighbors=best_knn_params['n_neighbors'])
bestDTC = DecisionTreeClassifier(max_depth=best_dtc_params['max_depth'], min_samples_split=best_dtc_params['min_samples_split'], min_samples_leaf=best_dtc_params['min_samples_leaf'])
bestMLP = MLPClassifier(activation=best_mlp_params['activation'], learning_rate=best_mlp_params['learning_rate'], hidden_layer_sizes=best_mlp_params['hidden_layer_sizes'])
bestGNB = GaussianNB()
estimators = [('bestKNN', bestKNN), ('bestDTC', bestDTC), ('bestMLP', bestMLP), ('bestGNB', bestGNB)]
vc = VotingClassifier(estimators=estimators, voting='hard')
# 训练模型
vc.fit(train_x, train_y)
y_pred = vc.predict(test_x)

# 计算召回率、精度、准确率
compare.loc['VotingClassifier'] = [recall_score(test_y, y_pred), precision_score(test_y, y_pred),
                                   accuracy_score(test_y, y_pred), accuracy_score(train_y, vc.predict(train_x))]
print('召回率：', recall_score(test_y, y_pred))
print('精度：', precision_score(test_y, y_pred))
print('准确率：', accuracy_score(test_y, y_pred))

# %%
# 随机森林
from sklearn.ensemble import RandomForestClassifier

# 利用网格搜索寻找最优的参数
# 设置需要搜索的参数
params_grid = {'n_estimators': range(1, 100)}
# 设置网格搜索
grid = GridSearchCV(RandomForestClassifier(), params_grid, cv=10, scoring='accuracy')
# 训练模型
grid.fit(train_x, train_y)
# 输出最优的参数
print('最优的n_estimators为：', grid.best_params_['n_estimators'])
# 预测结果
y_pred = grid.predict(test_x)

# 计算召回率、精度、准确率
compare.loc['RandomForestClassifier'] = [recall_score(test_y, y_pred), precision_score(test_y, y_pred),
                                         accuracy_score(test_y, y_pred), accuracy_score(train_y, grid.predict(train_x))]
print('召回率：', recall_score(test_y, y_pred))
print('精度：', precision_score(test_y, y_pred))
print('准确率：', accuracy_score(test_y, y_pred))

# %%
compare


