# 姓名：王宇彬 学号：3220190887
import pandas as pd
import numpy as np
# 处理电子游戏销售数据集
io = "D:/dataMiner/vgsales.csv"
data = pd.read_csv(io)
# 展示表格信息
data.info()
# 查看上面的表格信息，发现只有年份以及发行商这两列属性有缺失值，考虑到缺失的行不多，所以直接删除缺失的行
data.dropna(inplace = True)
# 展示删除缺失值后的表格信息
data.info()
# 展示每个年份的数据分布
data.groupby('Year').count()
# 由上面的数据分布发现2017年以及2020年的数据很少，因此直接删除这两年的数据。
data=data[-data.Year.isin([2017])]
data=data[-data.Year.isin([2020])]
# 展示删除后的表格信息
data.info()
# 查看表格中数值属性列值的分布
data.describe().T
# 查看表格中标称属性列值的分布
data.describe(include='object').T
# 由于可视化应用中需要用到电子游戏视场分析，因此把这两部分放在一起
# 计算销售额最高的五款游戏
topGlobalSales = data.groupby('Name').sum().sort_values('Global_Sales',ascending=False).head()
globalSales_y = topGlobalSales['Global_Sales'].values[::-1]
globalSales_x = topGlobalSales.index.values[::-1]
# 由下图可知，Wii Sports，Grand Theft Auto V等5款游戏的全球销售额很高，建议游戏开发商出新作。
import matplotlib.pyplot as plt
plt.figure(figsize=(18,3))
# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("全球销售额",fontdict={'weight':'normal','size': 20})
plt.ylabel("游戏名",fontdict={'weight':'normal','size': 20})
plt.tick_params(labelsize=13) 
plt.barh(globalSales_x, globalSales_y)  # 横放条形图函数 barh
plt.title('游戏销售额排行',fontdict={'weight':'normal','size': 20})
plt.show()

# 由下图可知，PS2，X360，PS3，Wii以及DS是玩家们常用的游戏平台，游戏开发者应该着重考虑在这几个平台上开发自己的游戏。
# 计算销售额最高的五个平台
topPlatform= data.groupby('Platform').sum().sort_values('Global_Sales',ascending=False).head()
platform_y = topPlatform['Global_Sales'].values[::-1]
platform_x = topPlatform.index.values[::-1]
plt.figure(figsize=(18,3))
plt.xlabel("全球销售额",fontdict={'weight':'normal','size': 20})
plt.ylabel("平台名",fontdict={'weight':'normal','size': 20})
plt.tick_params(labelsize=13) 
plt.barh(platform_x, platform_y)  # 横放条形图函数 barh
plt.title('平台销售额排行',fontdict={'weight':'normal','size': 20})
plt.show()

# 由下图可知，Action，Sports，Shooter，Role-Playing以及Platfrom是热门的游戏类型，游戏开发商应着重于这几种游戏类型的开发。
# 计算销售额最高的五种游戏类型
topGenre= data.groupby('Genre').sum().sort_values('Global_Sales',ascending=False).head()
genre_y = topGenre['Global_Sales'].values[::-1]
genre_x = topGenre.index.values[::-1]
plt.figure(figsize=(18,3))
plt.xlabel("全球销售额",fontdict={'weight':'normal','size': 20})
plt.ylabel("游戏类型",fontdict={'weight':'normal','size': 20})
plt.tick_params(labelsize=13) 
plt.barh(genre_x, genre_y)  # 横放条形图函数 barh
plt.title('游戏类型销售额排行',fontdict={'weight':'normal','size': 20})
plt.show()

# 计算销售额最高的五家发行商
# 由下图可知，Ninetendo，Electronic Arts等5家发行商发行的游戏最受玩家所喜爱，所以当玩家想尝试接触新的游戏时，
# 这5个发行商出品的游戏不失为一种好的选择。
topPublisher= data.groupby('Publisher').sum().sort_values('Global_Sales',ascending=False).head()
publisher_y = topPublisher['Global_Sales'].values[::-1]
publisher_x = topPublisher.index.values[::-1]
plt.figure(figsize=(18,3))
plt.xlabel("全球销售额",fontdict={'weight':'normal','size': 20})
plt.ylabel("发行商",fontdict={'weight':'normal','size': 20})
plt.tick_params(labelsize=13) 
plt.barh(publisher_x, publisher_y)  # 横放条形图函数 barh
plt.title('发行商销售额排行',fontdict={'weight':'normal','size': 20})
plt.show()

# 计算销售额最高的五家发行商发行的游戏数量
publisher_y2 = data.groupby('Publisher').count().loc[publisher_x,:]['Global_Sales']
plt.figure(figsize=(18,3))
plt.xlabel("游戏数量",fontdict={'weight':'normal','size': 20})
plt.ylabel("发行商",fontdict={'weight':'normal','size': 20})
plt.tick_params(labelsize=13) 
plt.barh(publisher_x, publisher_y2)  # 横放条形图函数 barh
plt.title('发行商销售游戏数量',fontdict={'weight':'normal','size': 20})
plt.show()

y = data.groupby('Year').sum()['Global_Sales'].values.reshape([37,1])
x = np.array(list(data.groupby('Year').groups.keys())).reshape([37,1])
from sklearn import linear_model
# 建立线性模型预测全球销售额
model = linear_model.LinearRegression() 
model.fit(x,y)
test = np.array([2017.0,2018.0,2019.0,2020.0,2021.0,2022.0,2023.0,2024.0,2025.0,2026.0]).reshape([10,1])
pre_y = model.predict(test)

from matplotlib.pyplot import MultipleLocator
# 对预测结果进行展示
# 根据图中的曲线可以看出，电子游戏的发展从1980年到2020年，全球销售额缓缓上升。从2001年到2008年，增长迅速。
# 但是从2009开始到2016年，呈现逐年下滑的趋势。到2016年时，销售额已经低于1996年了。
y = np.concatenate((y,pre_y))
x = np.concatenate((x,test))
plt.figure(figsize=(18,3))
plt.xlabel("年份",fontdict={'weight':'normal','size': 20})
plt.ylabel("全球销售额",fontdict={'weight':'normal','size': 20})
plt.tick_params(labelsize=13)
ax=plt.gca()
plt.xticks(rotation=60)
x_major_locator=MultipleLocator(1)
ax.xaxis.set_major_locator(x_major_locator)
plt.plot(x, y)
plt.title('全球销售额预测',fontdict={'weight':'normal','size': 20})
plt.show()

# 本文档针对近40年的电子游戏销售数据进行分析并预测之后的销售额
# 对于数据的处理，主要是删除了存在缺失值的数据以及数据较少的年份——2017年以及2020年。
# 在分析数据的过程中，用到了pandas工具，分析得到最受欢迎的游戏，平台以及发行商等，并根据这些数据给出自己对于游戏发行商的建议。
# 最后根据现有的每年的销售额数据，利用sklearn中的线性模型预测之后几年的销售情况。