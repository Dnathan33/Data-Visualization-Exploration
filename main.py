from __future__ import print_function
from os import listdir
from os.path import join
import binascii
from PIL import Image, ImageStat
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import matplotlib.pyplot as plt
import colour
import cv2




IMG_PATH = "InstaNY100K/img_resized/newyork"
CAPTION_PATH = "InstaNY100K/captions/newyork"

def get_captions():
    max = 3500
    captions = []
    for caption in listdir(CAPTION_PATH):
        caption_path = join(CAPTION_PATH, caption)
        captions.append(caption_path)
        max -= 1
        if (max) == 0: break
    return captions

def get_images():
    max = 1000
    images = []
    for image in listdir(IMG_PATH):
        img_path = join(IMG_PATH, image)
        images.append(img_path)
        max -= 1
        if (max) == 0: break
    return images

def get_dominant_color_from_image(img):
    NUM_CLUSTERS = 5
    img = img.resize((150, 150))  # optional, to reduce time
    ar = np.asarray(img)
    shape = ar.shape
    ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)
    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
    counts, bins = np.histogram(vecs, len(codes))  # count occurrences

    index_max = np.argmax(counts)  # find most frequent
    peak = codes[index_max]
    colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
    # print('most frequent is %s (#%s)' % (peak, colour))
    return (peak, colour)

def get_brightness_of_image(img):
    # im = Image.open(original_image).convert('L')
    stat = ImageStat.Stat(img)
    return stat.rms[0]

def get_contrast_of_image(img):
    img = cv2.imread(img)
    Y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:, :, 0]

    # compute min and max of Y
    min = np.min(Y)
    max = np.max(Y)

    # compute contrast
    if (max + min) == 0: contrast = 0
    else: contrast = (max - min) / (max + min)

    return contrast


def get_temprature_from_color(rgb_values):
    # Assuming sRGB encoded colour values.
    RGB = np.array(rgb_values)
    # Conversion to tristimulus values.
    XYZ = colour.sRGB_to_XYZ(RGB / 255)
    # Conversion to chromaticity coordinates.
    xy = colour.XYZ_to_xy(XYZ)
    # Conversion to correlated colour temperature in K.
    CCT = colour.xy_to_CCT(xy, 'hernandez1999')
    return CCT


def scatterplot(data):
    Y = [x[2] for x in data]
    X = [x[0] for x in data]
    colors = [x[1] for x in data]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.patch.set_facecolor('black')
    fig.patch.set_facecolor('black')
    plt.rcParams['axes.facecolor'] = 'black'

    ax.set_xlabel('Average color (rgb)')
    ax.xaxis.label.set_color('white')

    ax.set_ylabel('Brightness')
    ax.yaxis.label.set_color('white')

    # ax.spines['bottom'].set_color('white')
    # ax.spines['left'].set_color('white')

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    for i in range(len(X)):
        ax.scatter(X[i], Y[i], s=0.7)
    plt.show()
    plt.savefig('fig1.png')

def show(data):
    rgb_array = [x[1] for x in data]
    # rgb_array += [[255 - 255 * i / 600, 255 - 255 * i / 600, 255] for i in range(600)]
    img = np.array(rgb_array, dtype=int).reshape((1, len(rgb_array), 3))
    plt.imshow(img, extent=[0, 16000, 0, 1], aspect='auto')
    plt.show()
    plt.savefig('fig2.png')


def writeToFile(data):
    Y = [x[2] for x in data]
    X = [x[0] for x in data]

    f = open("result.csv", "w")
    for i in range(len(X)):
        f.write(str(X[i]) + "," + str(Y[i]) + "\n")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # get_captions()
    images = get_images()

    data = []
    for image in images:
        img = Image.open(image)
        color = get_dominant_color_from_image(img)[0]
        contrast = get_contrast_of_image(image)
        data.append((sum(color) // 3, color, get_brightness_of_image(img.convert('L')), get_temprature_from_color(color), contrast))

    scatterplot(data)
    show(data)
    writeToFile(data)
    print("DONE")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
