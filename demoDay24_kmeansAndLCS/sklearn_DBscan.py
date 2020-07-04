import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans


def getClusterData(flag="c", ns=1000, nf=2, centers=[[-1, -1], [1, 1], [2, 2]], cluster_std=[0.4, 0.5, 0.2]):
    '''
    得到回归数据
    centers(簇中心的个数或者自定义的簇中心)
    cluster_std(簇数据方差代表簇的聚合程度)
    '''
    if flag == 'c':
        cluster_X, cluster_y = datasets.make_circles(n_samples=ns, factor=.6, noise=.05)
    elif flag == 'b':
        cluster_X, cluster_y = datasets.make_blobs(n_samples=ns, n_features=nf, centers=centers,
                                                   cluster_std=cluster_std, random_state=9)
    else:
        cluster_X, cluster_y = datasets.make_moons(n_samples=ns, noise=0.1, random_state=1)
    return cluster_X, cluster_y


def dataSplit(dataset, label, ratio=0.3):
    '''
    数据集分割-----训练集、测试集合
    '''
    try:
        X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=ratio)
    except:
        dataset, label = np.array(dataset), np.array(label)
        X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=ratio)
    print('--------------------------------split_data shape-----------------------------------')
    print(len(X_train), len(y_train))
    print(len(X_test), len(y_test))
    return X_train, X_test, y_train, y_test


def saveModel(model, save_path="model.pkl"):
    '''
    模型持久化存储
    '''
    joblib.dump(model, save_path)
    print(u"持久化存储完成!")


def loadModel(model_path="model.pkl"):
    '''
    加载保存本地的模型
    '''
    model = joblib.load(model_path)
    return model


def clusterModel(flag='c'):
    '''
    Kmeans算法关键参数：
    n_clusters：数据集中类别数目
    DBSCAN算法关键参数：
    eps： DBSCAN算法参数，即我们的ϵ-邻域的距离阈值，和样本距离超过ϵ的样本点不在ϵ-邻域内
    min_samples： DBSCAN算法参数，即样本点要成为核心对象所需要的ϵ-邻域的样本数阈值
    '''
    X, y = getClusterData(flag=flag, ns=3000, nf=5, centers=[[-1, -1], [1, 1], [2, 2]],
                          cluster_std=[0.4, 0.5, 0.2])
    X_train, X_test, y_train, y_test = dataSplit(X, y, ratio=0.3)
    # 绘图
    plt.figure(figsize=(16, 8))
    # Kmeans模型
    model = KMeans(n_clusters=3, random_state=9)
    model.fit(X_train)
    y_pred = model.predict(X_test)
    plt.subplot(121)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
    plt.title('KMeans Cluster Result')
    # DESCAN模型
    # 下面的程序报错：AttributeError: 'DBSCAN' object has no attribute 'predict'
    # model=DBSCAN(eps=0.1,min_samples=10)
    # model.fit(X_train)
    # y_pred=model.predict(X_test)
    # 改为这样形式的可以了 moons 0.2,20 circle:0.1,7 blob:0.2,13
    y_pred = DBSCAN(eps=0.1, min_samples=4).fit_predict(X_test)
    plt.subplot(122)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
    plt.title('DBSCAN Cluster Result')
    if flag == 'c':
        plt.savefig('circleData.png')
    elif flag == 'b':
        plt.savefig('blobData.png')
    else:
        plt.savefig('moonsData.png')


if __name__ == '__main__':
    clusterModel("c")
