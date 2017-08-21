
import cv2
import os
from tqdm import tqdm
from random import shuffle
import numpy as np
import glob
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt


tf.reset_default_graph()

TRAIN_DIR = '/media/dhanush/DATA/Linux Chrome Downloads/req_extr_images'
TEST_DIR = '/home/dhanush/Desktop/TextReco/test_data'



# NEURAL NETWORK MODEL


IMG_SIZE = 45
LR = 1e-3

MODEL_NAME = 'digit_recog-{}-{}.model'.format(LR, '8conv-basic-3epoch-img_th_1')


convnet = input_data(shape=[None,IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)


convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 13, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')





def label_image(img): # Returns the training label
    word_label = img.split('_')[0]
    options = {'=': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               '-': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               '+': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               '0': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               '1': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               '2': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               '3': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
               '4': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
               '5': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               '6': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
               '7': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
               '8': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
               '9': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], }


    return options[word_label]


def create_train_data():  # Creates the training data as numpy array
    training_data=[]
    for m_label in tqdm(glob.glob(TRAIN_DIR+'/*')):
        for img_p in tqdm(glob.glob(m_label+'/*')):
            label = label_image(m_label[-1])
            img = cv2.imread(img_p,cv2.IMREAD_GRAYSCALE)
            _, img_th = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
            training_data.append([np.array(img_th), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy',training_data)
    return training_data



def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_label_act = img.split('_')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        _,img=cv2.threshold(img,50,255,cv2.THRESH_BINARY)
        img=cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img),img_label_act])

    np.save('test_data.npy',testing_data)
    return testing_data

def initialize_train_data():
    train_data=create_train_data()
    return



def train_the_network():

    # First run initialize_train_data and generate the training data file train_data.npy

    if glob.glob('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('The saved model '+str(MODEL_NAME) +' has been successfully loaded')


    if glob.glob('train_data.npy'):
        print('Saved training data found and loaded to memory.')
        train = np.load('train_data.npy')
    else:
        print("ERROR : No saved training data. Please run the function initialize_train_data")
        exit()


    X = np.array([i[0] for i in train[:-50]]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    Y = [i[1] for i in train[:-50]]

    test_x = np.array([i[0] for i in train[-50:]]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    test_y = [i[1] for i in train[-50:]]

    model.fit({'input': X}, {'targets': Y}, n_epoch=4, validation_set=({'input': test_x}, {'targets': test_y}),
              snapshot_step=500, show_metric=True, run_id='MODEL_NAME')

    model.save(MODEL_NAME)

    return

def accuracy_on_same_dataset():

    if glob.glob('train_data.npy'):
        print('Saved training data found and loaded to memory.')
        train = np.load('train_data.npy')
    else:
        print("ERROR : No saved training data. Please run the function initialize_train_data")
        exit()




    model.load(MODEL_NAME)

    test_x = np.array([i[0] for i in train[-12:]]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    test_y = [i[1] for i in train[-12:]]


    #fig = plt.figure()

    data = test_x[0]
    img_data = data[0]
    #y = fig.add_subplot(3, 4, 1)
    orig = img_data
    model_out = model.predict([data])

    cv2.imshow('img',cv2.resize(orig,(200,200),interpolation= cv2.INTER_AREA))
    print(model_out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


def test_model(path):



    options_num = { 0 : '=' ,
                    1 : '-',
                    2 : '+',
                    3 : '0',
                    4 : '1',
                    5 : '2',
                    6 : '3',
                    7 : '4',
                    8 : '5',
                    9 : '6',
                    10: '7',
                    11: '8',
                    12: '9',
                    }




    model.load(MODEL_NAME)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_r = cv2.resize(img, (45, 45), interpolation=cv2.INTER_AREA)

    _, img_r = cv2.threshold(img_r, 100, 255, cv2.THRESH_BINARY_INV)
    a = img_r.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    pred = model.predict(a)
    res=options_num[np.argmax(pred)]
    return res

def test_on_raw(path):

    options_num = { 0 : '=' ,
                    1 : '-',
                    2 : '+',
                    3 : '0',
                    4 : '1',
                    5 : '2',
                    6 : '3',
                    7 : '4',
                    8 : '5',
                    9 : '6',
                    10: '7',
                    11: '8',
                    12: '9',
                    }




    # Load the classifier
    model.load(MODEL_NAME)

    # Read the input image
    im = cv2.imread(path)

    # Convert to grayscale
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Threshold the image and apply Gaussian filtering
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
    im_th = cv2.GaussianBlur(im_th, (5, 5), 0)
    # im_gray = cv2.dilate(im_gray, (5, 5))

    # Find contours in the image
    _, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    #print(len(rects))

    # For each rectangular region, calculate HOG features and predict
    # the digit using Linear SVM.
    for i, rect in enumerate(rects):
        # Draw the rectangles
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        if pt1 < 0:
            pt1 = 0
        if pt2 < 0:
            pt2 = 0
        roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]
        cv2.imwrite('data_lib/'+str(i)+'.jpg',roi)

        # Resize the image
        roi = cv2.resize(roi, (45, 45), interpolation=cv2.INTER_AREA)
        roi = cv2.erode(roi, (3, 3))
        _,roi = cv2.threshold(roi,100,255,cv2.THRESH_BINARY_INV)
        # Calculate the HOG features
        # roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)

        data_tp = roi.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        pred = model.predict(data_tp)
        cv2.putText(im, str(options_num[np.argmax(pred)]), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        print(options_num[np.argmax(pred)])
    cv2.imshow("Resulting Image with Rectangular ROIs", im)
    cv2.waitKey()
    return





test_on_raw("/home/dhanush/Desktop/Internship Projects/TextReco/raw5.jpg")





























































