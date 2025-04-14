import tensorflow
from tensorflow.keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

train = ImageDataGenerator(rescale=1./255,zoom_range=0.2, horizontal_flip=True, shear_range=0.2)
test = ImageDataGenerator(rescale=1./255)

train_data = train.flow_from_directory('path',batch_size=32, class_mode='binary',target_size=(128,128))
test_data = test.flow_from_directory('path', batch_size=32, class_mode='binary',target_size=(128,128))


cnn = models.Sequential([
                        # cnn
                        layers.Conv2D(filters=32, kernel_size = (3,3), activation='relu', input_shape=(128,128,3)),
                        layers.MaxPooling2D((2,2)),

                        layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'), 
                        layers.MaxPooling2D((2,2)),

                        layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'), 
                        layers.MaxPooling2D((2,2)),
                        # dense
                        layers.Flatten(),
                        layers.Dense(100, activation='relu'),
                        layers.Dense(1, activation='sigmoid')
])
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model1 = cnn.fit(train_data,epochs=10,validation_data=test_data)
cnn.save("model_1.h5",model1)