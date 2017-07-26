import csv 
import numpy as np
import cv2

lines = []
with open("./data/driving_log.csv") as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
print("{} snapshots".format(len(lines)))

def Generator():
	images = []
	measurements = []
	for line in lines:
		for i in range(3):
			source_path = line[i]
			filename = source_path.split('/')[-1]
			current_path = "./data/IMG/" + filename
			image = cv2.imread(current_path)
			images.append(image)
			measurement = float(line[3])
			# adding correction factor
			# if i == 1:
			# 	measurement += 0.2
			# elif i == 2:
			# 	measurement -=0.2
			measurements.append(measurement)

	print("Augmenting Images...")
	aug_images, aug_measurements = [], []
	for image, measurement in zip(images, measurements):
		aug_images.append(image)
		aug_measurements.append(measurement)
		image_flipped = np.fliplr(image)
		aug_images.append(image_flipped)
		aug_measurements.append(-measurement)
	print(len(aug_images))
	while 1:
		for i in range(int(len(aug_images)/6./11)):
			j = 281
			yield np.array(aug_images[i*j: (i+1)*j]), np.array(aug_measurements[i*j: (i+1)*j])


# training the model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D, MaxPooling2D
model = Sequential()
#Preprocessing
# / 255.0 to make range 0 to 1
# minus 0.5 to shift mean
model.add(Lambda(lambda x : x/255.0 - 0.5, input_shape=(160,320,3)))
#model.add(Cropping2D(cropping=((75,25),(0,0))))
model.add(Convolution2D(24,5,5,border_mode='valid', activation='elu', subsample=(2,2)))
model.add(Convolution2D(36,5,5,border_mode='valid', activation='elu', subsample=(2,2)))
model.add(Convolution2D(48,5,5,border_mode='valid', activation='elu', subsample=(2,2)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='elu', subsample=(1,1)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='elu', subsample=(1,1)))
model.add(Flatten())
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dense(10, activation='elu'))
model.add(Dense(1, activation='tanh'))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(Generator(), samples_per_epoch=18546, nb_epoch=3, verbose=1)

#save model
model.save('model.h5')