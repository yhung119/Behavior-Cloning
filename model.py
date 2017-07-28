import csv 
import numpy as np
import cv2
from sklearn.utils import shuffle


lines = []
with open("./data/driving_log.csv") as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
print("{} snapshots".format(len(lines)))

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

print("Training : {} samples".format(len(train_samples)*2))
print("Validation set: {} samples".format(len(validation_samples)*2))
def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1:
		shuffle(samples)
		for offset in range(0, len(samples), int(batch_size)):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			measurements = []
			for batch_sample in batch_samples:
				for i in range(3):
					source_path = batch_sample[i]
					filename = source_path.split('/')[-1]
					current_path = "./data/IMG/" + filename
					image = cv2.imread(current_path)
					image = image[50:140,:,:]
					image = cv2.resize(image,(200, 66), interpolation = cv2.INTER_AREA)
					images.append(image)
					measurement = float(line[3])
					# adding correction factor
					if i == 1:
						measurement += 0.2
					elif i == 2:
						measurement -=0.2
					measurements.append(measurement)
			aug_images, aug_measurements = ([], [])
			for image, measurement in zip(images, measurements):
				aug_images.append(image)
				aug_measurements.append(measurement)
				image_flipped = np.fliplr(image)
				aug_images.append(image_flipped)
				aug_measurements.append(-measurement)

			X_train = np.array(aug_images)
			y_train = np.array(aug_measurements)
			yield (shuffle(X_train, y_train))
	
# training the model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.regularizers import l2, activity_l2
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
import matplotlib.pyplot as plt

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
#Preprocessing
# / 255.0 to make range 0 to 1
# minus 0.5 to shift mean
model.add(Lambda(lambda x : x/255.0 - 0.5, input_shape=(66,200,3), output_shape=(66,200,3)))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="elu"))

model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="elu"))

model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="elu"))

# Add two 3x3 convolution layers (output depth 64, and 64)
model.add(Convolution2D(64, 3, 3, activation="elu"))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, activation="elu"))
# Add a flatten layer
model.add(Flatten())
model.add(Dense(100, activation="elu"))
model.add(Dense(50, activation="elu"))
model.add(Dense(10, activation="elu"))
model.add(Dense(1))

# Compile and train the model, 
#model.compile('adam', 'mean_squared_error')
model.compile(optimizer=Adam(lr=1e-4), loss='mse')
checkpoint = ModelCheckpoint('model{epoch:02d}.h5')
history_object = model.fit_generator(train_generator, validation_data=validation_generator, nb_val_samples=len(validation_samples)*6, samples_per_epoch=len(train_samples)*6, nb_epoch=5, verbose=2, callbacks=[checkpoint])

#save model
model.save('model.h5')

print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()