import csv 
import numpy as np
import cv2
from sklearn.utils import shuffle

# load the file source path
lines = []
with open("../../../../Desktop/data/driving_log.csv") as csvfile:
	reader = csv.reader(csvfile)
	next(reader,None)
	for line in reader:
		lines.append(line)
print("{} snapshots".format(len(lines)))

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

print("Training : {} samples".format(len(train_samples)*2))
print("Validation set: {} samples".format(len(validation_samples)*2))
# generator function to push data to network as defined
# to speed up the training and efficiently use memory
def generator(samples, batch_size=16):
	num_samples = len(samples)
	while 1:
		shuffle(samples)
		for offset in range(0, len(samples), batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			measurements = []
			for batch_sample in batch_samples:
				# get images from three angles
				for i in range(3):
					source_path = batch_sample[i]
					filename = source_path.split('/')[-1]
					current_path = "../../../../Desktop/data/IMG/" + filename
					image = cv2.imread(current_path)
					images.append(image)
					measurement = float(line[3])
					# adding correction factor
					if i == 1:
						measurement += 0.2
					elif i == 2:
						measurement -=0.2
					measurements.append(measurement)
			aug_images, aug_measurements = [], []
			# augmenting images by flipping the image
			for image, measurement in zip(images, measurements):
				aug_images.append(image)
				aug_measurements.append(measurement)
				image_flipped = cv2.flip(image,1)
				aug_images.append(image_flipped)
				aug_measurements.append(measurement*-1.0)

			X_train = np.array(aug_images)
			y_train = np.array(aug_measurements)
			yield shuffle(X_train, y_train)

# training the model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D, Dropout
from keras.callbacks import ModelCheckpoint, Callback
import matplotlib.pyplot as plt

train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)

model = Sequential()
#Preprocessing
# / 255.0 to make range 0 to 1
# minus 0.5 to shift mean
model.add(Lambda(lambda x : x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Convolution2D(64, 3, 3, subsample=(2,2), activation="relu"))
# Add a flatten layer
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))#, activation="relu"))
model.add(Dense(10))#, activation="relu"))
model.add(Dense(1))#, activation="relu"))

# Compile and train the model
model.compile(optimizer="adam", loss='mse')
history_object = model.fit_generator(train_generator, validation_data=validation_generator, nb_val_samples=len(validation_samples)*6, samples_per_epoch=len(train_samples)*6, nb_epoch=5, verbose=1)

#save model
model.save('model.h5')

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig("history.png")
plt.show()