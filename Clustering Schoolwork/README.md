### 作业完成情况简介

##### 1.将鸢尾花数据集画成图的形式

```python
#将鸢尾花数据集可视化（聚类之前）
plt.figure(figsize=(8,8))
G = nx.Graph()
#添加点
G.add_nodes_from([i for i in range(150)])
#添加边
for i in range(len(sim)):
    for j in range(len(sim)):
        if(i in n[j] and j in n[i]):
            G.add_edge(i,j,weight=sim[i,j])
        else:
            sim[i,j] = 0
#画出结点
nx.draw_networkx_nodes(G,layout,node_color='r',node_size=30,node_shape='o')
#将图G中的边按照权重分组
edges_list1=[]
edges_list2=[]
edges_list3=[]
for(u,v,d) in G.edges(data='weight'):
    if d>0.9998:
        edges_list1.append((u,v))
    elif d>0.9996:
        edges_list2.append((u,v))
    else:
        edges_list3.append((u,v))
#以不同样式画出三组的边
nx.draw_networkx_edges(G,layout,edgelist=edges_list1,width=1,alpha=1.0,edge_color='k',style='solid')
nx.draw_networkx_edges(G,layout,edgelist=edges_list2,width=1,alpha=0.3,edge_color='k',style='solid')
nx.draw_networkx_edges(G,layout,edgelist=edges_list3,width=1,alpha=0.5,edge_color='k',style='dashed')
plt.savefig('iris_graph.png')
plt.show()
```

![](F:\Iris\第一个实训作业\iris_graph.png)

##### 2.确定一个合适的**阈值**，只有两个样本之间的相似度大于该阈值时，这两个样本之间才有一条边。

```python
#计算近邻矩阵（取每个点的最相似的16个点的索引）n为近邻矩阵
n =[]
for i in range(len(sim)):
    inds = np.argsort(sim[i])
    inds = inds[-16:-1]
    n.append(inds)
```

##### 3.求取带权**邻接矩阵**。

```python
#计算相似度矩阵sim
from sklearn.metrics.pairwise import cosine_similarity
sim=cosine_similarity(data)
```

##### 4.根据邻接矩阵进行聚类

```python
#运用谱聚类的归一割，取归一化对称拉普拉斯矩阵的前3小的特征值对应的特征向量Vectors
A = sim  #相似度矩阵
D = np.diag(A.sum(axis=0))  #度数矩阵
L = D - A #拉普拉斯矩阵
D = np.linalg.inv(np.sqrt(D))
L = D.dot(L).dot(D)  #对称拉普拉斯矩阵
w,v = np.linalg.eig(L)  #求出L的特征值和特征向量
inds = np.argsort(w)[:3] #求出前三小的特征值对应的下标
Vectors = v[:,inds]   #前三小的特征向量
normalizer = np.linalg.norm(Vectors,axis=1)
normalizer = np.repeat(np.transpose([normalizer]),3,axis=1)
Vectors = Vectors / normalizer #归一化
```

```python
#用k-means方法对归一化堆成拉普拉斯矩阵聚类
#随机选择3个点作为质心点
centroids_idx = np.random.choice(Vectors.shape[0],size=3)
centroids = Vectors[centroids_idx]
centroids = np.array(centroids)
print(centroids)
```

```python
#不断以新的质心聚类，并计算新的质心，直到迭代次数到达上限或者质心变化小于某阈值
max_iter = 300 #最大迭代次数
epsilon = 0.001  #阈值
a = 0
#根据质心聚类的函数
def split_cluster(Vectors,centroids):
    clusters = [[] for i in range(centroids.shape[0])]
    for i in range(Vectors.shape[0]):
        dist = np.square(Vectors[i]-centroids).sum(axis=1)
        idx = np.argmin(dist)
        clusters[idx].append(i)
    return(np.array(clusters))
#更新质心的函数
def update_centroids(clusters,Vectors):
    n_features = Vectors.shape[1]
    k = clusters.shape[0]
    centroids = np.zeros((k,n_features))
    for i,cluster in enumerate(clusters):
        centroid = np.mean(Vectors[cluster],axis=0)
        centroids[i] = centroid
    return(centroids)
#求出聚类好的分布矩阵clusters,其中每行代表一个类，每行的数据是点的索引值
for _ in range(max_iter):
    clusters = split_cluster(Vectors,centroids)
    former_centroids = centroids
    centroids = update_centroids(clusters,Vectors)
    diff = centroids - former_centroids
    if diff.any() < epsilon:
        break

```

```python
#根据聚类好的分布矩阵clusters预测出Vectors每行数据对应的标签
y_pred = np.zeros(Vectors.shape[0],dtype=np.int64)
for cluster_i,cluster in enumerate(clusters):
    for sample_i in cluster:
        y_pred[sample_i] = cluster_i
y_pred
```

##### 5.将聚类结果可视化，重新转换成图的形式，其中每一个簇应该用一种形状表示，比如分别用圆圈、三角和矩阵表示各个簇。

```python
#画出聚类后的可视化图，结点形状表示类别
#创建画布，创建图
plt.figure(figsize=(8,8))
N = nx.Graph()
#添加结点和边
N.add_nodes_from([i for i in range(150)])
for i in range(len(sim)):
    for j in range(len(sim)):
        if(i in n[j] and j in n[i]):
            N.add_edge(i,j,weight = sim[i,j])
#根据边的权重将边分为三类
edges_list1 = []
edges_list2 = []
edges_list3 = []
for (u,v,d) in N.edges(data='weight'):
    if d>0.9998:
        edges_list1.append((u,v))
    elif d>0.9996:
        edges_list2.append((u,v))
    else:
        edges_list3.append((u,v))
#画出结点        
nx.draw_networkx_nodes(N, layout, node_size=30, nodelist=clusters[0], node_shape='o')
nx.draw_networkx_nodes(N, layout, node_size=30, nodelist=clusters[1], node_shape='^')
nx.draw_networkx_nodes(N, layout, node_size=30, nodelist=clusters[2], node_shape='s')
#画出边
nx.draw_networkx_edges(N,layout,edgelist=edges_list1,width=1,alpha=1.0,edge_color='k',style='solid')
nx.draw_networkx_edges(N,layout,edgelist=edges_list2,width=1,alpha=0.3,edge_color='k',style='solid')
nx.draw_networkx_edges(N,layout,edgelist=edges_list3,width=1,alpha=0.5,edge_color='k',style='dashed')
#保存图片
plt.savefig('cl_iris_graph.png')
plt.show()
```

![](F:\Iris\第一个实训作业\cl_iris_graph.png)

##### 6.求得分簇正确率

划分的正确率为0.9333

```python
#计算聚类的正确率acc
cnt = 0
for i in range(150):
    if label[i] == y_pred[i]:
        cnt += 1
acc = cnt / 150
acc
```

##### 7.完成代码的描述文档

描述文档为每点要求对应的代码块和图片