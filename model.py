import cv2
import csv
import numpy as np
import os
import sklearn

def get_driving_logs(data_path, skipHeader=True):
    #Returns the lines from a driving log with base directory
    lines = []
    with open(data_path + '/driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        if skipHeader:
            next(reader, None)
        for line in reader:
            lines.append(line)
    return lines


def find_images_old(data_path):
    #Finds all the images needed for training 
    centers = []
    measurements = []
    lines = get_driving_logs(data_path)
    for line in lines:
        source_path = line[0].strip()
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/' + filename
        image = cv2.imread(current_path)
        centers.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
    X_train = np.array(centers)
    y_train = np.array(measurements)
    return (X_train, y_train)


def find_images(data_path):
    #Finds all the images needed for training 
    directories = [x[0] for x in os.walk(data_path)]
    data_directories = list(filter(lambda directory: os.path.isfile(directory + '/driving_log.csv'), directories))
    centers = []
    lefts = []
    rights = []
    measurements = []
    for directory in data_directories:
        lines = get_driving_logs(directory)
        center = []
        left = []
        right = []
        measurements = []
        for line in lines:
            measurements.append(float(line[3]))
            center.append(directory + '/' + line[0].strip())
            left.append(directory + '/' + line[1].strip())
            right.append(directory + '/' + line[2].strip())
        centers.extend(center)
        lefts.extend(left)
        rights.extend(right)
        measurements.extend(measurements)

    return (centers, lefts, rights, measurements)


def combine_images(center, left, right, measurement, correction):
    #Combine the image paths from `center`, `left` and `right` with correction factor
    
    image_paths = []
    image_paths.extend(center)
    image_paths.extend(left)
    image_paths.extend(right)
    measurements = []
    measurements.extend(measurement)
    measurements.extend([x + correction for x in measurement])
    measurements.extend([x - correction for x in measurement])
    return (image_paths, measurements)


def generator(samples, batch_size=32):
    #Generate the required images and measurments for training/
    
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for image_path, measurement in batch_samples:
                original_image = cv2.imread(image_path)
                image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                                
                #if measurement < 0.001: # to ignore zero steering angle data
                #    if np.random.random() < 0.5:# to delete 50% straight line image data.
                #        continue
                
                images.append(image)
                angles.append(measurement)
                
                # Flipping
                images.append(cv2.flip(image,1))
                angles.append(measurement*-1.0)

            # trim image to only see section with road
            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)
            
   
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


# Resize images as required for network.
def resize_im(x):
    from keras.backend import tf
    return tf.image.resize_images(x, (66, 160))


# Alternative models, flowing through basic to complex one
def Simple():
    # Simple model to state to output as a benchmark
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Flatten())
    model.add(Dense(1))
    return model
    
def LeNet():
    # Creates a LeNet model
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def nvidia_model():
    # Creates nvidia model
    model = Sequential()
    
    model.add(Lambda(lambda x: (x / 255.0 - 0.5), input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,20), (0,0))))
    #model.add(Lambda(resize_im)) # resizing images.
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


#### Alternative models training ####

##1## First model - Very simple network
# Reading images locations
'''
X_train, y_train = find_images_old('data')
print('Total Images: {}'.format( len(X_train)))
'''

# Training model
'''
model = Simple()
model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2,shuffle=True, nb_epoch=2)
'''


##2## Second model - LeNet network
# Reading images locations
'''
X_train, y_train = find_images('data')
print('Total Images: {}'.format( len(X_train)))
'''

# Training model
'''
model = LeNet()
model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2,shuffle=True, nb_epoch=5)
'''


##3## Third model - Nvidia Autonomus Car network
# Reading images locations.
center_paths, left_paths, right_paths, measurements = find_images('data')
image_paths, measurements = combine_images(center_paths, left_paths, right_paths, measurements, 0.2)
print('Total Images: {}'.format( len(image_paths)))

# Splitting samples and creating generators.
from sklearn.model_selection import train_test_split
samples = list(zip(image_paths, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Model creation
model = nvidia_model()

# Complie model with mean squared error losss function and ada optimizer. & # Training model
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples),\
        validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

model.save('model.h5')
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])