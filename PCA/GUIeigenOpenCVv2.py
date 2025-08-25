import cv2
import numpy as np

image_scale = 5

def scale(np1):
    np2 = (np1 - np.min(np1)) / np.ptp(np1)
    return np2

def scale_and_reshape(np1, mf, old_shape):
    if mf is None:
        np2 = np1.reshape(old_shape, order='F')
    else:
        np2 = (np1 + mf).reshape(old_shape, order='F')
    np2 = scale(np2)
    return np2

def nothing(x):
    pass


T = np.load("pca.res_FACES/T_st500.npy")
v_correct = np.load("pca.res_FACES/v_st500.npy")
w = np.load("pca.res_FACES/w_st500.npy")
mean_face = np.load("pca.res_FACES/mean_face_st500.npy")
norms = np.load("pca.res_FACES/norms_st500.npy")
old_shape = np.load("pca.res_FACES/old_shape_st500.npy")
how_many_images = T.shape[1]

eigen_offset = 0;
number_of_sliders = 10;
divider = 0
if eigen_offset + number_of_sliders > how_many_images:
    eigen_offset = how_many_images - number_of_sliders

window_size = ((int)(image_scale * old_shape[1]), int(image_scale * old_shape[0]))
print(window_size)
start = 0
stop = v_correct.shape[1]
v_correct_use = v_correct[:,start:stop]

# Create a black image, a window
#img = np.zeros((300,512,3), np.uint8)
img = mean_face
cv2.namedWindow('image',1)
cv2.namedWindow('image2',1)

img = scale_and_reshape(mean_face, None, old_shape)
img = cv2.resize(img, window_size)

cv2.imshow('image', img)

cv2.createTrackbar('Eigen offset', 'image2', 0, how_many_images - number_of_sliders, nothing)
cv2.setTrackbarPos('Eigen offset', 'image2', eigen_offset)

cv2.createTrackbar('Divider', 'image2', 0, 64, nothing)
cv2.setTrackbarPos('Divider', 'image2', divider)

cv2.resizeWindow('image2', 768, 768)
#cv2.resizeWindow('image2', 512, 512)

trackbar_names = []
for a in range(number_of_sliders):
    trackbar_names.append("f" + str(a))
    cv2.createTrackbar(trackbar_names[a], 'image2', 0, 512, nothing)
    cv2.setTrackbarPos(trackbar_names[a], 'image2', 256)

ff = np.zeros(stop)


def reset_trackbar(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global ff
        print('Features:')
        print(ff)
        ff = np.zeros(stop)
        for a in range(number_of_sliders):
            cv2.setTrackbarPos(trackbar_names[a], 'image2', 256)

cv2.setMouseCallback('image',reset_trackbar)


while(1):
    old_eigen_offset = eigen_offset
    eigen_offset = cv2.getTrackbarPos('Eigen offset', 'image2')
    if old_eigen_offset != eigen_offset:
        for a in range(number_of_sliders):
            new_value = ff[a + eigen_offset] * (divider + 1) + 256
            new_value = int(new_value)
            if  new_value < 0:
                new_value = 0
            cv2.setTrackbarPos(trackbar_names[a], 'image2', new_value)


    divider = cv2.getTrackbarPos('Divider', 'image2')
    #ff = np.zeros(stop)
    for a in range(number_of_sliders):
        ff[a + eigen_offset] = (cv2.getTrackbarPos(trackbar_names[a], 'image2') - 256) / (divider + 1)

    reconstruct = np.matmul(v_correct_use, ff)
    img = scale_and_reshape(reconstruct, mean_face, old_shape)
    img = cv2.resize(img, window_size)

    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()