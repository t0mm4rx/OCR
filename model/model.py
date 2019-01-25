from PIL import Image
import numpy as np
import random
from neural_network_np import NeuralNetwork

classes = ['0', '1', '2', '3', '4']

def get_image_pixels(cl, nb):
    im = Image.open('data/' + str(cl) + "/" + str(nb) + ".png").convert('L')
    im.thumbnail((32, 32), Image.ANTIALIAS)
    pixels = im.getdata()
    return np.array(pixels)

def get_batch(size):
    features = []
    targets = []
    for i in range(size):
        cl = random.choice(classes)
        im_nb = random.randint(0, 29)
        targets.append(class_to_output(cl))
        features.append(get_image_pixels(cl, im_nb) / 127.5 - 1.0)
    return (features, targets)

def get_all_data():
    x = []
    y = []
    for cl in classes:
        for i in range(30):
            targets.append(class_to_output(cl))
            features.append(get_image_pixels(cl, im_nb) / 127.5 - 1.0)
    return (x, y)

def class_to_output(cl):
    a = [0] * len(classes)
    a[int(cl)] = 1
    return np.array(a)

"""
    Learning
"""
batch_size = 50
iters = 10
epochs = 20

val_x, val_y = get_batch(50)

nn = NeuralNetwork(1024, [200], len(classes), learning_rate=0.07)

for i in range(epochs):
    (x, y) = get_batch(batch_size)
    for iter in range(iters):
        for a in range(len(x)):
            nn.train(x[a], y[a])
    #print("Epoch", i+1, "of", epochs, "cost:", round(nn.cost(x, y), 3), "accuracy:", round(nn.accuracy(x, y), 3), "%")
    print(round(nn.cost(val_x, val_y), 3))
