from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
num_classes = 10
seed = 1
# featurewise需要数据集的统计信息，因此需要先读入一个x_train，用于对增强图像的均值和方差处理。
img = load_img("./data/image/a/8.jpg")
image = img_to_array(img)
image = image.reshape((1,)+image.shape)

imagegen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=44,
)

maskgen = ImageDataGenerator(

)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
imagegen.fit(image)
image_iter = imagegen.flow_from_directory('./data/image', class_mode=None, batch_size=8, seed=seed)
mask_iter = maskgen.flow(imagegen.standardize(image), batch_size=8, seed=seed)
data_iter = zip(image_iter, mask_iter)

for i in range(4):
    x_batch,y_batch = data_iter.__next__()
    plt.subplot(2,4,i+1)
    plt.imshow(x_batch[0])
    plt.subplot(2,4,4+i+1)
    plt.imshow(y_batch[0])
plt.show()