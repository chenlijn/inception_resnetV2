
import numpy as np
import os
import shutil
import cv2
import keras
from keras.models import load_model


def get_result(model, imgfile_list):
    results = []
    for imgfile in imgfile_list:
        try:
            image = cv2.imread(imgfile)
            resized_img = cv2.resize(image, (299, 299))
            cv2.imwrite('resized_img.png', resized_img)
            rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            prediction = model.predict(np.expand_dims(rgb/255.0, axis=0), batch_size=1)
            #print(prediction[0])
            results.append(prediction[0].argmax())
        except:
            print(imgfile)
            print('get result failed !')
    return results


def get_conf_mat(predictions, gts, class_num=2):
    conf_mat = np.zeros((class_num, class_num), np.int)
    for idx, predict in enumerate(predictions):
        conf_mat[predict, gts[idx]] += 1
    return conf_mat


def store_data(imgfile_list, predictions, dst_path):
    for idx, predict in enumerate(predictions):
        folder = dst_path+str(predict)
        if not os.path.exists(folder):
            os.mkdir(folder)

        shutil.copy(imgfile_list[idx], folder)

def write_result_to_txt(imgfile_list, predictions, txt_file): 
    with open(txt_file, 'w+') as tf:
        for idx, imgfile in enumerate(imgfile_list):
            tf.write(imgfile.split('/')[-1] + ' ' + str(predictions[idx]) + '\n') 


def main(model_path, txt_file, store_path):
    with open(txt_file, 'r') as tf:
        lines = tf.readlines()

    imgfile_list = []
    gt_list = []
    for line in lines:
        imgfile, gt = line.strip('\n').split()
        imgfile_list.append(imgfile)
        gt_list.append(int(gt))
    
    model = load_model(model_path)
    predictions = get_result(model, imgfile_list)
    write_result_to_txt(imgfile_list, predictions, txt_file.split('.')[0] + '_eye_side.txt')
    store_data(imgfile_list, predictions, store_path)
    conf_mat = get_conf_mat(predictions, gt_list)
    print('confusion matrix: \n{}'.format(conf_mat))


# without ground truth
def main2(model_path, txt_file, store_path):
    with open(txt_file, 'r') as tf:
        lines = tf.readlines()

    imgfile_list = []
    for line in lines:
        imgfile_list.append(line.strip('\n'))
    
    model = load_model(model_path)
    predictions = get_result(model, imgfile_list)
    write_result_to_txt(imgfile_list, predictions, txt_file.split('.')[0] + '_eye_side.txt')
    store_data(imgfile_list, predictions, store_path)


if __name__ == '__main__':
    model_path = 'model/weights-0065-0.005.hdf5'
    ##txt_file = '/root/mount_out/data/left_right_disc_data/test/right_label_docker.txt'
    ##txt_file = '/root/mount_out/data/left_right_disc_data/train_label_docker.txt'
    ##txt_file = 'test/disc_val_docker.txt'
    #img_type = 'online_abnormal_test'
    #txt_file = '/root/mount_out/data/original_glaucoma_data/all_data_sorted/glaucoma_cls_test_set/splits/det/{}_with_path.txt'.format(img_type)
    ##store_path = '/root/mount_out/work2018/test/eye_cls/'
    #store_path = '/root/mount_out/data/original_glaucoma_data/all_data_sorted/glaucoma_cls_test_set/splits/det/eye_cls/{}/'.format(img_type)
    #main2(model_path, txt_file, store_path)


    # Process REFUGE data
    img_type = 'non_glaucoma_with_path'
    txt_file = '/root/mount_out/data/public_datasets/disc_cup_segmentation_dataset/det/validation/{}.txt'.format(img_type)
    #store_path = '/root/mount_out/work2018/test/eye_cls/'
    #store_path = '/root/mount_out/data/original_glaucoma_data/all_data_sorted/glaucoma_cls_test_set/splits/det/eye_cls/{}/'.format(img_type)
    store_path = '/root/mount_out/data/public_datasets/disc_cup_segmentation_dataset/det/validation/eye_cls/'
    main2(model_path, txt_file, store_path)



