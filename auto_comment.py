import cv2
import numpy as np
import matplotlib.pyplot as plt
import face_recognition
from PIL import Image
import os
import argparse

def preprocess(path, imsuffix='.png', visualize=False):
    img = cv2.imread(path+imsuffix)
    try:
        landmarks = face_recognition.face_landmarks(img, model="large")[0]
    except:
        return
    left_eye = np.mean(landmarks['left_eye'], axis=0).astype(np.int32)
    right_eye = np.mean(landmarks['right_eye'], axis=0).astype(np.int32)
    with open(path+'.txt', 'w') as f:
        line = '{} {} {} {} '.format(left_eye[0], left_eye[1], right_eye[0], right_eye[1])
        f.write(line)

    if visualize == True:
        plt.imshow(Image.fromarray(img))
        plt.plot(left_eye[0], left_eye[1], 'o')
        plt.plot(right_eye[0], right_eye[1], 'o')
        plt.show()

def main(args):
    for i in range(1, 41):
        for j in range(1, 11):
            path = os.path.join(args.path, '{}'.format(i))
            preprocess(path+'/{:02d}'.format(j), '.png', args.visualize)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='main')
    parser.add_argument('--path', type=str, default='/home/haotongl/datasets/orl')
    parser.add_argument('--visualize', type=int, default=False)
    args = parser.parse_args()
    globals()[args.type](args)

