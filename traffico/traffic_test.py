
# coding: utf-8

# In[19]:


import tensorflow as tf
import numpy as np

# 플레이스 홀더를 설정합니다.
X = tf.placeholder(tf.float32, shape=[None, 7])
Y = tf.placeholder(tf.float32, shape=[None, 1])


W = tf.Variable(tf.random_normal([7, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")



# 가설을 설정합니다.
hypothesis = tf.matmul(X, W) + b


# 저장된 모델을 불러오는 객체를 선언합니다.

saver = tf.train.Saver()
model = tf.global_variables_initializer()


# 7가지 변수를 입력 받습니다. 기온,강수량,풍속,습도,일조,적설,지면온도

avg_temp = float(input('온  도: '))
avg_rain = float(input('강수량: '))
wind_vel = float(input('풍  속: '))
humidity = float(input('습  도: '))
sunshine = float(input('일조량: '))
snowfall = float(input('적설량: '))
suf_temp = float(input('표면온도: '))


with tf.Session() as sess:
    sess.run(model)


    # 저장된 학습 모델을 파일로부터 불러옵니다.
    save_path = "saved.cpkt"
    saver.restore(sess, save_path)

    # 사용자의 입력 값을 이용해 배열을 만듭니다.
    data = ((avg_temp, avg_rain, wind_vel, humidity, sunshine, snowfall, suf_temp), )
    arr = np.array(data, dtype=np.float32)



    # 예측을 수행한 뒤에 그 결과를 출력합니다.

    x_data = arr[:]
    dict = sess.run(hypothesis, feed_dict={X: x_data})

    print(dict[0])


# # 
