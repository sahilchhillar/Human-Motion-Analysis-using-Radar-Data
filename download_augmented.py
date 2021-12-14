from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def download_files(x_train, y_train, x_test, y_test, shape):
    labels = []
    for i in range(len(x_train)):
        img = x_train[i].reshape(shape)
        img = Image.fromarray(np.uint8(img)).convert('RGB')
        img = plt.imshow(img)
        plt.savefig(fname=f'Compressed Images/{i}.png')
        labels.append(y_train[i])

    for i in range(len(x_test)):
        img = x_test[i].reshape(shape)
        img = Image.fromarray(np.uint8(img)).convert('RGB')
        img = plt.imshow(img)
        plt.savefig(fname=f'Compressed Images/{i}.png')
        labels.append(y_test[i])

    return np.array(labels)