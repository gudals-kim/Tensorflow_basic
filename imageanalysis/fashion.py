import tensorflow as tf
import matplotlib.pyplot as plt
#숫자의 이미지를 이미지로 치환시켜서 볼수있음

(trainX, trainY),(testX, testY) = tf.keras.datasets.fashion_mnist.load_data()
#( (trainX, trainY),(testX, testY) )

#print(trainY)

#plt.imshow(trainX[92])#trainX 의 92번째 이미지를 이미지로 볼수있게 치환
#plt.gray()#색상을 흑백으로 
#plt.colorbar()#옆에 컬러바옵션추가
#plt.show()#출력옵션

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot']

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(28,28), activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation="softmax")
])


model.summary()

exit()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(trainX, trainY, epochs=5)