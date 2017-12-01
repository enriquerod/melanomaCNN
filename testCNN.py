
import argparse
import cv2
import numpy as np
#from the python script models.py
from models import tiny_XCEPTION



def get_args():
    parser = argparse.ArgumentParser(description="This script runs a cnn classfication "
                                                 "for product recognition.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--im_path", "-i_p", type=str, required=True,
                        help="path to output database mat file")

    args = parser.parse_args()
    return args

#   480 x 640 pixeles
#   

#
#

def main():

    args = get_args()
    image_path = args.im_path

    img_w = 64
    img_h = 64
    img_d = 3

    n_Class = 25
 
    model = tiny_XCEPTION((img_h, img_w, img_d), n_Class, l2_regularization=0.01)
    # load weights into the model
    model.load_weights("weights-19.hdf5")
    #compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print("Weights model loaded from disk")

    # image_test = cv2.imread('2.jpg')
    image_test = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)

    rgb_image = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)

    image = cv2.resize(rgb_image, (img_w, img_h))
    data_x = image.reshape(1, img_w, img_h, img_d)
    data_x = data_x.astype('float32')
    data_x = data_x/255
    preds = model.predict(data_x)
    # print(preds)
    print('Clase: ', np.argmax(preds,axis=1)+1)
    
if __name__ == '__main__':
    main()