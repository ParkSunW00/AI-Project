
# coding: utf-8

# In[10]:


import tensorflow as tf
import numpy as np
from pandas.io.parsers import read_csv


# In[25]:


data = read_csv('traffic_clean_utf.csv', sep=',')
xy = np.array(data, dtype=np.float32)

# 4개의 변인을 입력을 받습니다.
x_data = xy[:, 1:-1]

# 가격 값을 입력 받습니다.
y_data = xy[:, [-1]]


# In[22]:


# 입력 데이터 변수를 지정한다. 
X = tf.placeholder(tf.float32, shape=[None, 7])
Y = tf.placeholder(tf.float32, shape=[None, 1])

#변수를 초기화 한다.
W = tf.Variable(tf.random_normal([7, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# 가설을 설정합니다.
hypothesis = tf.matmul(X, W) + b

# 비용 함수를 설정합니다.
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 최적화 함수를 설정합니다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)
train = optimizer.minimize(cost)

# 세션을 생성합니다.
sess = tf.Session()

# 글로벌 변수를 초기화합니다.
model = tf.global_variables_initializer()
sess.run(model)

# 학습을 수행합니다.
for step in range(100000):

    cost_, hypo_, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})

    if step % 500 == 0:
        print("#", step, " 손실 비용: ", cost_)
        print("사상자: ", hypo_[0])

# 학습된 모델을 저장합니다.
saver = tf.train.Saver()
save_path = saver.save(sess, "./saved.cpkt")
print('학습된 모델을 저장했습니다.')


# In[ ]:




