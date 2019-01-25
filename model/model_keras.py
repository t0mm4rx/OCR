from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation
import numpy as np
from PIL import Image

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def get_image_pixels(cl, nb):
    im = Image.open('data/' + str(cl) + "/" + str(nb) + ".png").convert('L')
    im.thumbnail((32, 32), Image.ANTIALIAS)
    pixels = im.getdata()
    return np.array(pixels) / 127.5 - 1.0

def get_batch(size):
    features = []
    targets = []
    for i in range(size):
        cl = random.choice(classes)
        im_nb = random.randint(0, 29)
        targets.append(class_to_output(cl))
        features.append(get_image_pixels(cl, im_nb))
    return (features, targets)

def get_all_data():
    x = []
    y = []
    for cl in classes:
        for i in range(30):
            y.append(class_to_output(cl))
            x.append(get_image_pixels(cl, i))
    return (np.array(x), np.array(y))

def class_to_output(cl):
    a = [0] * len(classes)
    a[int(cl)] = 1
    return np.array(a)

def output_to_class(output):
    return list(output).index(max(output))

"""
    Learning
"""

(x, y) = get_all_data()

model = Sequential()
model.add(Dense(500, input_dim=1024))
model.add(Activation('sigmoid'))
model.add(Dense(300, input_dim=500))
model.add(Activation('sigmoid'))
model.add(Dense(len(classes), input_dim=300))
model.add(Activation('softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x, y, epochs=30, batch_size=10)

model.save('model.h5')

#model = load_model('model.h5')

print(output_to_class(model.predict(np.array([get_image_pixels('guess', '0')]))[0]))
print(output_to_class(model.predict(np.array([get_image_pixels('guess', '1')]))[0]))
print(output_to_class(model.predict(np.array([get_image_pixels('guess', '2')]))[0]))
print(output_to_class(model.predict(np.array([get_image_pixels('guess', '3')]))[0]))
print(output_to_class(model.predict(np.array([get_image_pixels('guess', '4')]))[0]))
print(output_to_class(model.predict(np.array([get_image_pixels('guess', '5')]))[0]))
print(output_to_class(model.predict(np.array([get_image_pixels('guess', '6')]))[0]))
print(output_to_class(model.predict(np.array([get_image_pixels('guess', '7')]))[0]))
print(output_to_class(model.predict(np.array([get_image_pixels('guess', '8')]))[0]))
print(output_to_class(model.predict(np.array([get_image_pixels('guess', '9')]))[0]))
