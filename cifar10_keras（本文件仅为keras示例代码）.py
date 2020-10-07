import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()

print(len(x_train))
print(len(x_test))
print(x_train.shape)

#第一个[0]代表取50000张图片中的第一张，第二个[0]代表取第一行，第三个[0]代表取第一列
x_train[0][0][0] 

label_dict={0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}

def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig=plt.gcf()
    fig.set_size_inches(12,14)
    if num>25: num=25
    for i in range(0,num):
        ax = plt.subplot(5,5,i+1)
        ax.imshow(images[idx],cmap='binary') 
        title= str(i)+' '+label_dict[labels[i][0]]   #显示数字对应的类别
        if len(prediction)>0:
            title+= '=>'+label_dict[prediction[i]]   #显示数字对应的类别
        ax.set_title(title,fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx+=1
    plt.show()
 
plot_images_labels_prediction(x_train,y_train,[],0)

x_train_normalize = x_train.astype('float32')/255
x_test_normalize = x_test.astype('float32')/255

print(x_train_normalize[0][0][0])

y_train_OneHot = tf.keras.utils.to_categorical(y_train)
y_test_OneHot = tf.keras.utils.to_categorical(y_test)

model = tf.keras.models.Sequential()
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),padding='same',input_shape=(32,32,3),activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'))
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(2500,activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(1500,activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10,activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

train_history=model.fit(x_train_normalize,y_train_OneHot,validation_split=0.2,epochs=5,batch_size=128,verbose=1)

def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.xlabel('epoch')
    plt.ylabel(train)
    plt.legend(['train','validation'],loc='upper left')
 
show_train_history(train_history,'accuracy','val_accuracy')
show_train_history(train_history,'loss','val_loss')

scores = model.evaluate(x_test_normalize,y_test_OneHot,verbose=2)
print(scores[1])

prediction = model.predict_classes(x_test_normalize)
print(prediction)

plot_images_labels_prediction(x_test,y_test,prediction,0,10)

prediction_probability = model.predict(x_test_normalize)
print(prediction_probability[0])

def show_predicted_probability(x,y,prediction,prediction_probability,i):
    print("label:",label_dict[y[i][0]],'predict,',label_dict[prediction[i]])
    plt.figure(figsize=(2,2))
    plt.imshow(x[i])
    plt.show()
    for j in range(10):          %输出10个类别概率
        print(label_dict[j]+'Probability:%1.9f'%(prediction_probability[i][j]))
        
print(prediction.shape)
print(y_test.shape)
pd.crosstab(y_test.reshape(-1),prediction,rownames=['label'],colnames=['prediction'])

model.save_weights("cifar.h5")

model.load_weights("cifar.h5")
