import numpy as np

def train_images(train):
    imgs = []
    lbls = []

    for i in range(len(train)):
        lbls.append(train[i][1])
        img = train[i][0]
        img = img[-1, :, :].cpu().detach().numpy()
        img = np.ravel(img.reshape(1, img.shape[0]*img.shape[1])).tolist()
        imgs.append(img)
    
    return imgs, lbls

def valid_images(valid, imgs, lbls):
    for i in range(len(valid)):
        lbls.append(valid[i][1])
        img = valid[i][0]
        img = img[-1, :, :].cpu().detach().numpy()
        img = np.ravel(img.reshape(1, img.shape[0]*img.shape[1])).tolist()
        imgs.append(img)

    return np.array(imgs), np.array(lbls)


def test_images(test):
    imgs_test = []
    lbls_test = []

    for i in range(len(test)):
        lbls_test.append(test[i][1])
        img = test[i][0]
        img = img[-1, :, :].cpu().detach().numpy()
        img = np.ravel(img.reshape(1, img.shape[0]*img.shape[1])).tolist()
        imgs_test.append(img)

    return np.array(imgs_test), np.array(lbls_test)