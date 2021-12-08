import tensorflow as tf
import pandas as pd
import numpy as np



data = pd.read_csv(r'C:\Users\kimtu\OneDrive\바탕 화면\머닝러신\github\Tensorflow_basic\success_rate\gpascore.csv',encoding='utf-8')

#data.isnull().sum() 빈칸이 몇개있는지 찾아줌
#data.fillna(100) 빈칸을 100으로 채워줌
#data['gre'].min() gre라는 속성에서 최저값을 찾아줌 최댓값은 max 데이터의 개수는 count 등 많음

data = data.dropna()
#.dropna() 이건 빈칸의 행을 지워줌


y데이터 = data['admit'].values 
#.values 는 리스트에 담아줌

x데이터 = []
for i, rows in data.iterrows():
    #iterrows를 한 행씩 출력할수있음
    x데이터.append([rows['gre'], rows['gpa'], rows['rank']])


    
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')    
])

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

model.fit(np.array(x데이터), np.array(y데이터) ,epochs=10000)

#예측
예측값 = model.predict( [ [750, 3.40, 3], [400, 2.2, 1] ] )
#아까 만든 model값으로 y값 예측
print(예측값)
