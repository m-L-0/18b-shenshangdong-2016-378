### TensorFlow Schoolwork完成情况简介

##### 1.将鸢尾花数据集安装8 : 2的比例划分成训练集与验证集

```python
#切分数据，测试集百分之二十，训练集百分之八十
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(iris_data,iris_label,test_size=0.2,shuffle=True,random_state = 5)
```

##### 2.设计模型：

```python
#训练集和测试集的占位符
train = tf.placeholder(tf.float32,shape=[None,4])
test = tf.placeholder(tf.float32,shape=[4])
#计算训练集与单个测试集的距离向量
distance = tf.reduce_sum(tf.abs(tf.add(train,tf.negative(test))),reduction_indices=1)
#预测所有测试集对应的类别的函数
def train_knn(K):
    with tf.Session() as sess:
        #定义一个空的预测数组
        pred = []
        #遍历所有测试集
        for i in range(len(X_test)):
            dist = sess.run(distance,feed_dict={train:X_train,test:X_test[i]})
            knn_idx = np.argsort(dist)[:K]
            #找出k个近邻中最多的那个类别，视为本测试样本的类别
            classes = [0,0,0]
            for i in knn_idx:
                if(Y_train[i]==0):
                    classes[0]+=1
                elif(Y_train[i]==1):
                    classes[1] += 1
                else:
                    classes[2]+=1
            y_pred = np.argmax(classes)
            pred.append(y_pred)
        return pred
```

##### 3.训练模型：

```python
def acc(k):
    y_pred = train_knn(k)
    y_true = Y_test
    acc = np.sum(np.equal(y_pred,y_true))/len(y_true)
    return acc
```

##### 4.验证模型,调整参数

K值最后确定为16

```python
#让k值遍历1到30，显示出对应的正确率
for i in range(1,31):
    print(i)
    print(acc(i))
```

```python
#使正确率最大的k值中最小的是16，所以当划分数据的random_state=5时，k值可以取16最合适
k = 1
for i in range(1,31):
    if (acc(i) > acc(k)):
        k = i
print(k)
```

##### 5.提交模型：

本文档按照作业要求给出对应的实现代码块