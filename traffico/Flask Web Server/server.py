# -*- coding: utf-8 -*-
from flask import Flask, render_template, request

import datetime
import tensorflow as tf
import numpy as np

app = Flask(__name__)

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

# 세션 객체를 생성합니다.
sess = tf.Session()
sess.run(model)

# 저장된 모델을 세션에 적용합니다.
save_path = "./model/saved.cpkt"
saver.restore(sess, save_path)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        # 파라미터를 전달 받습니다.
        avg_temp = float(request.form['avg_temp'])
        avg_rain = float(request.form['avg_rain'])
        wind_vel = float(request.form['wind_vel'])
        humidity = float(request.form['humidity'])
        sunshine = float(request.form['sunshine'])
        snowfall = float(request.form['snowfall'])
        suf_temp = float(request.form['suf_temp'])

        # 결과 변수 초기화
        casualty = 0

        # 입력된 파라미터를 배열 형태로 준비합니다.
        data = ((avg_temp, avg_rain, wind_vel, humidity, sunshine, snowfall, suf_temp), (0,0,0,0,0,0,0))
        arr = np.array(data, dtype=np.float32)

        # 입력 값을 토대로 예측 값을 찾아냅니다.
        x_data = arr[:]
        dict = sess.run(hypothesis, feed_dict={X: x_data})
            
        # 결과 배추 가격을 저장합니다.
        casualty = dict[0]

        return render_template('index.html', casualty=casualty)

if __name__ == '__main__':
   app.run(debug = False)
