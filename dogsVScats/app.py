import tensorflow as tf
#import os
#import shutil

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dogsVScats\dataset',
    image_size=(64,64),#이미지 사이즈 설정
    batch_size=64,#2만장을 한번에 넣지 않고 32개씩 집어 넣는다.
    subset='training',#학습에 쓰일것
    validation_split=0.2,#얼만큼 20/80으로 사용
    seed=1234
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dogsVScats\dataset',
    image_size=(64,64),#이미지 사이즈 설정
    batch_size=64,#2만장을 한번에 넣지 않고 32개씩 집어 넣는다.
    subset='validation',#확인에 쓰일것
    validation_split=0.2,#얼만큼
    seed=1234
)

#print(train_ds)#train_ds에는 아마 ((이미지2만개).(정답2만개)가 들어있을것)

#전처리함수
def 전처리함수(i,정답):#0고 1사이로 압축하는거
    i = tf.cast(i/255.0,tf.float32) 
    return i, 정답


train_ds = train_ds.map(전처리함수)#전처리함수 넣어줘야함
val_ds = val_ds.map(전처리함수)
#for i in os.listdir(r'dogsVScats\train\train'):

#    if 'cat' in i:
#        shutil.move('dogsVScats\\train\\train\\'+i, r'dogsVScats\dataset\cat')
#    if 'dog' in i:
#        shutil.move('dogsVScats\\train\\train\\'+i, r'dogsVScats\dataset\dog') 


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),padding="same",activation='relu', input_shape=(64,64,3)),#칼라사진이면 마지막에 3(rgb가 들어감)
    tf.keras.layers.MaxPooling2D( (2,2) ),
    tf.keras.layers.Conv2D(64,(3,3),padding="same",activation='relu'),
    tf.keras.layers.MaxPooling2D( (2,2) ),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(128,(3,3),padding="same",activation='relu'),
    tf.keras.layers.MaxPooling2D( (2,2) ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),#오버피팅 해결법 레이어 줄이기
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.summary()

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(train_ds,validation_data=val_ds, epochs=5)