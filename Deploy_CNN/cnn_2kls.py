# Masukkan library program
import os,cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras import backend as K
K.set_image_dim_ordering('th')

from keras.utils import np_utils
from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D,ZeroPadding2D
from keras.optimizers import Adam

PATH = os.getcwd()
# Mendefinisikan Path directory
data_path = PATH + '/data_cnn_modify3'
data_dir_list = os.listdir(data_path)

img_rows=32
img_cols=32
num_channel=1
num_epoch=200

# Mendefinisikan Kelas Output
num_classes = 2

img_data_list=[]
kernel = np.ones((16,16), np.uint8) #kernel untuk morphology opening
for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Memuat gambar untuk dataset - '+'{}\n'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
#        input_img=cv2.flip(input_img,1) #Melakukan flip vertikal
#        input_img=cv2.flip(input_img,0) #Melakukan flip horizontal
        input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img=cv2.GaussianBlur(input_img,(5,5),0)
        ret3,th3 = cv2.threshold(input_img,0,255,cv2.THRESH_OTSU)
        open_img = cv2.morphologyEx(th3, cv2.MORPH_OPEN, kernel)
        input_img_resize=cv2.resize(open_img,(img_rows,img_cols))
        img_data_list.append(input_img_resize)

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print (img_data.shape)

if num_channel==1:
	if K.image_dim_ordering()=='th':
		img_data= np.expand_dims(img_data, axis=1) 
		print (img_data.shape)
	else:
		img_data= np.expand_dims(img_data, axis=4) 
		print (img_data.shape)
else:
	if K.image_dim_ordering()=='th':
		img_data=np.rollaxis(img_data,3,1)
		print (img_data.shape)
		
# Membuat Label Data
# Definisi kembali Kelas Output
num_classes = 2

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:383]=0
labels[384:]=1
	  
names = ['layak','tidak layak']
	  
# Konversi label setiap kelas dengan meng-encoding data
Y = np_utils.to_categorical(labels, num_classes)

# Memecahkan data untuk training dan testing
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)


# Setup Model CNN
input_shape=img_data[0].shape
					
model = Sequential()

model.add(Convolution2D(16, 3,3,border_mode='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

adam = Adam(lr=0.001) #uji dengan 3 learning rate 0,0001 hingga 0,01
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# Hasil konfigurasi model
model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape			
model.layers[0].output_shape			
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable

#%%
# Training data

## Penggunaan Tensorboard untuk menampilkan graph hasil peltihan
#callbacks = TensorBoard(log_dir='./Graph')
#
## Proses Training data
#model.fit(X_train, y_train, batch_size=128, epochs=num_epoch, verbose=1, 
#          validation_data=(X_test, y_test), callbacks=[callbacks])

# Training with callbacks
from keras import callbacks
hist = model.fit(X_train, y_train, batch_size=128, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test))
filename='model_train_new.csv'

csv_log=callbacks.CSVLogger(filename, separator=',', append=False)

early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')

filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [csv_log,early_stopping,checkpoint]

# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(num_epoch)

plt.figure(1,figsize=(10,6))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('Jumlah Epoch')
plt.ylabel('Nilai loss (dalam 100%)')
plt.title('')
plt.grid(True)
plt.legend(['data latih','data tes'])

plt.figure(2,figsize=(10,6))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('Jumlah Epoch')
plt.ylabel('Nilai Akurasi (dalam 100%)')
plt.title('')
plt.grid(True)
plt.legend(['data latih','data tes'],loc=4)

# Evaluasi Model
score = model.evaluate(X_test, y_test, verbose=0)
print('Test Loss:', score[0])
print("Accuracy: %.2f%%" % (score[1]*100))

#%%

#test_image = X_test[0:1]
#print(y_test[0:1])

# Testing dengan gambar baru
test_image = cv2.imread('data_cnn_modify3/layak/IMG_7425.jpg')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
#test_image = cv2.flip(input_img,1) #Melakukan flip vertikal
#test_image = cv2.flip(input_img,0) #Melakukan flip horizontal
ret,test_image = cv2.threshold(test_image,0,255,cv2.THRESH_OTSU)
test_image = cv2.morphologyEx(test_image, cv2.MORPH_OPEN, kernel)
test_image = cv2.resize(test_image,(img_rows,img_cols))
test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
#print (test_image.shape)
   
if num_channel==1:
	if K.image_dim_ordering()=='th':
		test_image= np.expand_dims(test_image, axis=0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=4) 
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
else:
	if K.image_dim_ordering()=='th':
		test_image=np.rollaxis(test_image,2,0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)

# Predicting the test image
print("Hasil prediksi = "+ str(model.predict_classes(test_image)))

#%%

# Visualisasi layer
def get_featuremaps(model, layer_idx, X_batch):
	get_activations = K.function([model.layers[0].input, K.learning_phase()],
                               [model.layers[layer_idx].output,])
	activations = get_activations([X_batch,0])
	return activations

layer_num=3
filter_num=0

activations = get_featuremaps(model, int(layer_num),test_image)

print (np.shape(activations))
feature_maps = activations[0][0]      
print (np.shape(feature_maps))

if K.image_dim_ordering()=='th':
	feature_maps=np.rollaxis((np.rollaxis(feature_maps,2,0)),2,0)
print (feature_maps.shape)

fig=plt.figure(figsize=(16,16))
plt.imshow(feature_maps[:,:,filter_num],cmap='gray')
plt.savefig("featuremaps-layer-{}".format(layer_num) + 
            "-filternum-{}".format(filter_num)+'.jpg')

num_of_featuremaps=feature_maps.shape[2]
fig=plt.figure(figsize=(16,16))	
plt.title("featuremaps-layer-{}".format(layer_num))
subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
for i in range(int(num_of_featuremaps)):
	ax = fig.add_subplot(subplot_num, subplot_num, i+1)
	ax.imshow(feature_maps[:,:,i],cmap='gray')
	plt.xticks([])
	plt.yticks([])
	plt.tight_layout()
plt.show()
fig.savefig("featuremaps-layer-{}".format(layer_num) + '.jpg')

#%%
# Printing the confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
import itertools

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)
target_names = ['layak', 'tidak layak']
					
print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))
#print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))


# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Confusion matrix")
    else:
        print('Confusion matrix, tanpa normalisasi')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix, tanpa normalisasi')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
                      title='Confusion matrix')

plt.show()

#%%
# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

model.save('model.hdf5')
loaded_model=load_model('model.hdf5')

