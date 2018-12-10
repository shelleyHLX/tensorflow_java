import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import graph_util

## -1到1之间随机数 100个
train_X = np.linspace(-1, 1, 100)
train_Y = 2*train_X + np.random.randn(*train_X.shape)*0.1

# 显示模拟数据点

plt.plot(train_X, train_Y, 'ro', label='test')
plt.legend()
plt.show()


# 创建模型
# 占位符
X = tf.placeholder("float",name='X')
Y = tf.placeholder("float",name='Y')

# 模型参数
# W初始化为-1到1之间的一个数字
W = tf.Variable(tf.random_normal([1]), name="weight")
# b初始化为0 也是一维  定义变量
b = tf.Variable(tf.zeros([1]), name="bias")

# 前向结构   mulpiply两个数 相乘
z = tf.multiply(X, W) + b
op = tf.add(tf.multiply(X, W),b,name='results')
# 反向优化
cost = tf.reduce_mean(tf.square(Y - z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 初始化所有变量
init = tf.global_variables_initializer()

# 定义参数
training_epochs = 20
display_step = 2

def moving_avage(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx<w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

saver = tf.train.Saver()

# 启动session
with tf.Session() as sess:
    sess.run(init)
    # 存放批次值和损失值
    plotdata = {"batchsize": [], "loss": []}

    # 向量模型输入数据
    for epoch in range(training_epochs):
        for(x, y) in zip(train_X, train_Y):
            sess.run(optimizer, {X:x, Y:y})

        # 显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost, {X:train_X, Y:train_Y})
            print("Epoch:", epoch+1, "cost=", loss, "W=", sess.run(W), "b=",sess.run(b))
            if not (loss == "NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)

    print("Finished!")

    #保存模型
    saver.save(sess, "model/first")

    print("cost =", sess.run(cost,  feed_dict={X:train_X, Y:train_Y}), "W=", sess.run(W), "b=", sess.run(b))

    const_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,["results"])

    with tf.gfile.FastGFile("model/first.pb",mode='wb') as f:
        f.write(const_graph.SerializeToString())

    # 图形显示
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W)*train_X+sess.run(b),label='Filttedline')
    plt.legend()
    plt.show()

    plotdata["avgloss"] = moving_avage(plotdata["loss"])
    # plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["batchsize"],plotdata["avgloss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs, Trainging loss')
    plt.show()

    print("x=0.2, z=", sess.run(z, {X:0.2}))


