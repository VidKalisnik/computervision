# %%
# 1 a
import math
from UZ_utils import *
print("1 a")
I = imread('images/umbrellas.jpg')
imshow(I)

# %%
# 1 b
print("1 b")


def rgb2gray(I):
    I_gray = np.copy(I)

    I_gray[:, :, 0] = (I[:, :, 0] + I[:, :, 1] + I[:, :, 2]) / 3
    I_gray[:, :, 1] = (I[:, :, 0] + I[:, :, 1] + I[:, :, 2]) / 3
    I_gray[:, :, 2] = (I[:, :, 0] + I[:, :, 1] + I[:, :, 2]) / 3

    return I_gray


# %%
I_gray = rgb2gray(I)
imshow(I_gray)

# %%
# 1 c
print("1 c")


def cutout(I, x, y, w, h, c):
    cutout = np.copy(I)
    cutout = cutout[x:x+w, y:y+h, c]

    return cutout


# %%
I_cutout = cutout(I, 130, 130, 240, 210, 0)
imshow(I_cutout)

# %%
# 1 d
print("1 d")


def invert(I, x, w, y, h):
    I_invert = np.copy(I)

    I_invert[x:x+w, y:y+h] = 1 - I_invert[x:x+w, y:y+h]

    return I_invert


# %%
I_invert = invert(I, 130, 130, 240, 210)
imshow(I_invert)

# %%
# 1 e
print("1 e")


def reduce_gray_levels(I):
    I_gray = rgb2gray(I)
    I_gray = I_gray * 0.3

    return I_gray


# %%
I_gray2 = reduce_gray_levels(I)
plt.subplot(1, 2, 1)
plt.imshow(I_gray)
plt.subplot(1, 2, 2)
plt.imshow(I_gray2, vmin=0, vmax=1)

plt.show()


# %%
# 2 a
print("2 a")
I = imread('images/bird.jpg')
I_gray = imread_gray('images/bird.jpg')


# %%
def binaryMask(I, threshold):
    Image = np.copy(I)
    Image[Image < threshold] = 0
    Image[Image >= threshold] = 1

    return Image


def binaryMaskALT(I, threshold):
    Image = np.copy(I)
    binary_mask = np.where(Image < threshold, 0, 1).astype(np.float64)

    return binary_mask


# %%
threshold = 0.2

I_binaryMask = binaryMask(I_gray, threshold)
plt.subplot(1, 3, 1)
plt.imshow(I)
plt.subplot(1, 3, 2)
plt.imshow(I_gray)
plt.subplot(1, 3, 3)
plt.imshow(I_binaryMask)


plt.show()


# %%
# 2 b
print("2 b")


def myhist(I, bins):
    I = I.reshape(-1)
    H = np.zeros(bins)
    for i in range(len(I)):

        j = math.floor(I[i] * bins)
        j = min(max(j, 0), bins - 1)
        # print(I[i], j)
        H[j] += 1

    H = H / np.sum(H)
    return H


# %%
bin20 = myhist(I_gray, 20)
bin100 = myhist(I_gray, 100)

plt.figure(figsize=(20, 5))

plt.subplot(1, 3, 1)
plt.imshow(I_gray)
plt.subplot(1, 3, 2)
plt.bar(np.arange(100), bin100)
plt.subplot(1, 3, 3)
plt.bar(np.arange(20), bin20)

plt.show()

# %%
# 2 c
print("2 c")


def myhist2(I, bins):
    I = I.reshape(-1)
    H = np.zeros(bins)
    max = np.max(I)
    min = np.min(I)
    # print(max)
    # print(min)
    for i in range(len(I)):
        j = math.floor(((I[i]-min)/(max-min))*(bins-1))
        H[j] += 1

    H = H / np.sum(H)
    return H


# %%
I_gray2 = I_gray * 0.2
bin20 = myhist2(I_gray2, 20)
bin100 = myhist2(I_gray2, 100)

plt.figure(figsize=(20, 5))

plt.subplot(1, 3, 1)
plt.imshow(I_gray)
plt.subplot(1, 3, 2)
plt.bar(np.arange(100), bin100)
plt.subplot(1, 3, 3)
plt.bar(np.arange(20), bin20)

plt.show()

# %%
# 2 d
print("2 d")

im1 = imread_gray('images/IMG_9211.jpg')
im2 = imread_gray('images/IMG_9212.jpg')
im3 = imread_gray('images/IMG_9213.jpg')

plt.subplot(1, 2, 1)
plt.imshow(im1)
plt.subplot(1, 2, 2)
plt.bar(np.arange(100), myhist(im1, 100))
plt.show()

plt.subplot(1, 2, 1)
plt.imshow(im2)
plt.subplot(1, 2, 2)
plt.bar(np.arange(100), myhist(im2, 100))
plt.show()

plt.subplot(1, 2, 1)
plt.imshow(im3)
plt.subplot(1, 2, 2)
plt.bar(np.arange(100), myhist(im3, 100))
plt.show()

plt.show()

# %%
# 2 e
print("2 e")


def otsu_th(I, th):

    # spremenim v uin8 od 0 do 255
    I = (I * 255).astype(np.uint8)

    thresholded_im = np.zeros(I.shape)
    thresholded_im[I >= th] = 1

    nb_pixels = I.size
    nb_pixels1 = np.count_nonzero(thresholded_im)

    w1 = nb_pixels1 / nb_pixels  # delez belih
    w0 = 1 - w1  # delez crnih

    if w1 == 0 or w0 == 0:
        return np.inf

    # vsi pixli ki so beli
    val1 = I[thresholded_im == 1]
    # print(val1)
    # vsi pixli ki so crni
    val0 = I[thresholded_im == 0]
    # print(val0)

    o1 = np.var(val1) if len(val1) > 0 else 0
    o0 = np.var(val0) if len(val0) > 0 else 0

    return w0 * o0 + w1 * o1


def otsu(I):
    threshold_range = range(np.max((I.reshape(-1) * 255).astype(np.uint8))+1)
    criterias = [otsu_th(I, th) for th in threshold_range]

    best_threshold = threshold_range[np.argmin(criterias)]
    return best_threshold / 256


# %%
I_otsu = binaryMask(I_gray, otsu(I_gray))
imshow(I_otsu)


# %%
im_earth = imread_gray('images/earth.jpeg')
im_earth_otsu = binaryMask(im_earth, otsu(im_earth))
imshow(im_earth_otsu)

im_candy = imread_gray('images/candy.jpg')
im_candy_otsu = binaryMask(im_candy, otsu(im_candy))
imshow(im_candy_otsu)

im_flowers = imread_gray('images/flowers.jpeg')
im_flowers_otsu = binaryMask(im_flowers, otsu(im_flowers))
imshow(im_flowers_otsu)

# %%
# 3 a

I = imread_gray('images/mask.png')

n = 5
SE = np.ones((n, n))  # create a square structuring element
I_eroded = cv2.erode(I, SE)
I_dilated = cv2.dilate(I, SE)
I_both1 = cv2.dilate(I_eroded, SE)
I_both2 = cv2.erode(I_dilated, SE)

plt.subplot(1, 5, 1)
plt.imshow(I)
plt.subplot(1, 5, 2)
plt.imshow(I_eroded)
plt.subplot(1, 5, 3)
plt.imshow(I_dilated)
plt.subplot(1, 5, 4)
plt.imshow(I_both1)
plt.subplot(1, 5, 5)
plt.imshow(I_both2)

plt.show()


# %%
# 3 b

bird_gray = imread_gray('images/bird.jpg')
bird_binary = binaryMask(bird_gray, otsu(bird_gray))

SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))

plt.subplot(1, 2, 1)
plt.imshow(bird_binary)
plt.subplot(1, 2, 2)

# closing
bird = cv2.erode(cv2.dilate(bird_binary, SE), SE)
plt.imshow(bird)

plt.show()


# %%
# 3 c
def immask(I, mask):

    if I.shape != mask.shape:
        mask = np.expand_dims(mask, axis=2)

    I = I * mask

    return I


# %%
birdRGB = imread('images/bird.jpg')

onlyBird = immask(birdRGB, bird)
imshow(onlyBird)

# %%
# 3 d

im_eagle = imread('images/eagle.jpg')
im_eagle_binary = binaryMask(rgb2gray(im_eagle), otsu(im_eagle))

plt.figure(figsize=(20, 5))

plt.subplot(1, 3, 1)
plt.imshow(im_eagle)
plt.subplot(1, 3, 2)

eagle = cv2.dilate(cv2.erode(im_eagle_binary, SE), SE)

plt.imshow(eagle)

plt.subplot(1, 3, 3)
onlyEagle = immask(im_eagle, eagle)
plt.imshow(onlyEagle)

plt.show()

# %%
# 3 e
print("3 e")

im_coins = imread('images/coins.jpg')
im_coins_gray = imread_gray('images/coins.jpg')
im_coins_binary = binaryMask(im_coins_gray, otsu(im_coins_gray))

plt.figure(figsize=(20, 5))

# plt.imshow(im_coins_binary)
SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
coins = cv2.erode(cv2.erode(im_coins_binary, SE), SE)
coins = cv2.erode(cv2.erode(cv2.erode(coins, SE), SE), SE)
coins = cv2.dilate(cv2.dilate(cv2.dilate(
    cv2.dilate(cv2.dilate(coins, SE), SE), SE), SE), SE)

# invert
coins = 1 - coins

imshow(coins)


num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
    coins.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)

print(num_labels)

for label in range(1, num_labels):
    if stats[label, cv2.CC_STAT_AREA] > 700:
        coins[labels == label] = 0  # Set the region to background

imshow(coins)
