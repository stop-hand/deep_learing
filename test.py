 #实现K-means算法对以下模式分布进行聚类分析:{x1(0,0),x2(3,8),x3(2,2),x4(1,1),x5(5,3),x6(4,8),x7(6,3),x8(5,4),x9(6,4),x10(7,5)}
import numpy as np  
import matplotlib.pyplot as plt  
  
#初始化质心  
def init_centroids(data,k):  
  # 获取数据的特征数量  
  n_features = data.shape[1]  
  # 初始化质心为0矩阵，形状为(k,n_features)  
  centroids = np.zeros((k,n_features))  
  for i in range(k):  
    # 随机选取一个数据的索引  
    index = int(np.random.uniform(0,data.shape[0]))  
    # 将选取的数据设为当前质心  
    centroids[i] = data[index]  
  # 返回质心  
  return centroids  
  
# K-means聚类算法  
def k_means(data,k,max_iter):  
  # 获取数据总数量和特征数量  
  m = data.shape[0]  
  n_features = data.shape[1]  
  # 初始化质心  
  centroids = init_centroids(data,k)  
  
  for i in range(max_iter):  
    # 初始化一个长度为数据总数量的数组，用于存储每个数据点的聚类结果  
    cluster = np.zeros((m,))  

    for j in range(m):  
      # 初始化最小距离为非常大的数  
      min_dist = 100000000  
      # 对每一个质心，计算其与当前数据点的距离pip install d2l
      for l in range(k):  
        # 计算距离，使用欧几里得距离的公式：sqrt((x1-x2)^2 + (y1-y2)^2)  
        dist = np.sum(np.power(data[j]-centroids[l],2))  
        # 如果距离小于当前最小距离，更新最小距离和对应的聚类结果  
        if dist < min_dist:  
          min_dist = dist  
          cluster[j] = l  
    # 对每一个质心，更新其位置为对应聚类中心的位置（所有属于该聚类的数据点的位置的平均值）  
    for l in range(k):  
      centroids[l] = np.mean(data[cluster==l],axis=0)  
  # 返回质心和聚类结果  
  return centroids,cluster  
  
#可视化聚类结果（点的分布和质心的位置）  

def plot_cluster(data,centroids,cluster):   
  # 使用scatter函数绘制数据点，点的颜色由聚类结果决定（红色为错误的数据点）  
  plt.scatter(data[:,0],data[:,1],c=cluster)  
  # 使用scatter函数绘制质心，颜色为红色  
  plt.scatter(centroids[:,0],centroids[:,1],c='r',s=100)  
  plt.show()  
  
if __name__ == '__main__':  
  # 定义数据 
  data = np.array([[0,0],[3,8],[2,2],[1,1],[5,3],[4,8],[6,3],[5,4],[6,4],[7,5]])  
  k = 3  
  max_iter = 100    
  centroids,cluster = k_means(data,k,max_iter)  
  # 调用plot_cluster函数进行可视化展示
  plot_cluster(data,centroids,cluster)