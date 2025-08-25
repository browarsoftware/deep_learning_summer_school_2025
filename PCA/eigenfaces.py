import cv2
import numpy as np
import os
# http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
#path = 'e:\\Projects\\python\\img_align_celeba'

#path = 'e:\\Projects\\python\\same_twarze'
#path = 'd:/Projects/Python/PycharmProjects/twarze_align'
#path = 'd:/Projects/Python/PycharmProjects/same_twarze'
path = 'd://dane//CASIA-WebFace_align-same_twarze'


files = []
#how_many_images = 100
#how_many_images = 10000
how_many_images = 500
variance_explained = 0.99

# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
      files.append(os.path.join(r, file))

print(len(files))
#how_many_images = len(files)
a = 0
offset = 0
img = cv2.imread(files[0 + offset])
old_shape = img.shape
img_flat = img.flatten('F')

T = np.zeros((img_flat.shape[0], how_many_images))
for i in range(how_many_images):
    img_help = cv2.imread(files[i + offset])
    T[:,i] = img_help.flatten('F') / 255

mean_face = T.mean(axis = 1)

for i in range(how_many_images):
    T[:,i] -= mean_face

C = np.matmul(T.transpose(), T)
C = C / how_many_images

from scipy.linalg import eigh
w, v = eigh(C)
v_correct = np.matmul(T, v)

print(T.shape)
print(v.shape)
print(v_correct.shape)

image_to_code = T[:,0]
result = np.matmul(v_correct.transpose(), image_to_code)
reconstruct = np.matmul(v_correct, result)

if True:

    sort_indices = w.argsort()[::-1]
    w = w[sort_indices]  # puttin the evalues in that order
    v_correct = v_correct[:, sort_indices]

    w_percen = w / sum(w)
    variance = 0
    cooef_number = 0
    while variance < variance_explained:
        variance += w_percen[cooef_number]
        cooef_number = cooef_number + 1

    how_many_eigen = cooef_number
    print("requires ", how_many_eigen, " components to get ", variance, " of variance.")



    norms = np.linalg.norm(v_correct, axis=0)# find the norm of each eigenvector
    v_correct = v_correct / norms

    #save results
    np.save("pca.res//T_st" + str(how_many_images), T)
    np.save("pca.res//v_st" + str(how_many_images), v_correct)
    np.save("pca.res//w_st" + str(how_many_images), w)
    np.save("pca.res//mean_face_st" + str(how_many_images), mean_face)
    np.save("pca.res//norms_st" + str(how_many_images), norms)
    np.save("pca.res//old_shape_st" + str(how_many_images), np.asarray(old_shape))
    '''
    '''
    start = 0
    stop = how_many_eigen
    v_correct_use = v_correct[:,start:stop]

    #change all eigenvectors to have first coordinate positive - optional
    for i in range(how_many_eigen):
        if v_correct_use[0,i] < 0:
            v_correct_use[:, i] = -1 * v_correct_use[:, i]

    w_correct_use = w[start:stop]

    #image_to_code = T[:,50]
    image_to_code = T[:,0]

    result = np.matmul(v_correct_use.transpose(), image_to_code)
    reconstruct = np.matmul(v_correct_use, result)

    #result_features = (1 / np.sqrt(w_correct_use)) * result

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

    print(min(reconstruct + mean_face))
    v1 = np.array([1, 1, 1])
    v2 = np.array([2, 4, 8])
    print((1.0 / v2) * v1)
    reconstruct2 = scale_and_reshape(reconstruct, mean_face, old_shape)
    cv2.imshow('reconstructed',reconstruct2)

    image_to_code2 = scale_and_reshape(image_to_code, mean_face, old_shape)
    cv2.imshow('original',image_to_code2)

    mean_face2 = scale_and_reshape(mean_face, None,old_shape)
    cv2.imshow('mean_face',mean_face2)

    fe1 = scale_and_reshape(v_correct_use[:,0] * norms[0], None,old_shape)
    cv2.imshow('First eigenface',fe1)

    fe2 = scale_and_reshape(v_correct_use[:,1] * norms[1], None,old_shape)
    cv2.imshow('Second eigenface',fe2)


    fe2 = scale_and_reshape(v_correct[:,how_many_images-1] * norms[how_many_images-1], None,old_shape)
    cv2.imshow('Last eigenface',fe2)

    print(T.shape)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

