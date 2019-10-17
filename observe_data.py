import os, glob

#basedir
BASE_TRAIN_DIR = os.path.abspath("../C-NMC_training_data")
#no. of folds according to data description
folds = list(filter( lambda x:x.startswith('f'), os.listdir(BASE_TRAIN_DIR)))
#ALL(Acoustic Lymphocytic Leukemia) and health img dirs for each fold collected separately 
train_dir = [os.path.join(BASE_TRAIN_DIR, path) for path in folds]
all_imgs = [glob.glob( path + "\\all\\*.bmp") for path in train_dir]
hem_imgs = [glob.glob( path + "\\hem\\*.bmp") for path in train_dir]
#aggregate ALL images

all_imgs_dict_folded = {}   #with folds
properties = ['PAT_ID','NUM_IMG', 'NUM_CELL']

for c, fold in enumerate(folds):
    in_fold_dict = {}    
    for sample_id, img in enumerate(all_imgs[c]):
        sample_dic = {}
        info = os.path.split(img)[-1].split('.')[0].split('_') 
        sample_dic['PATH'] = img
        for i, prop in enumerate(properties):
            sample_dic[prop] = info[i+1]
        sample_dic['CONDITION'] = 1
        in_fold_dict[os.path.split(img)[-1].split('.')[0]] = sample_dic
    all_imgs_dict_folded[fold] = in_fold_dict
    
'''import this variable for future use -- all_imgs_dict'''
#all imgs without folds
all_imgs_dict = {**all_imgs_dict_folded['fold_0'],
                 **all_imgs_dict_folded['fold_1'],
                 **all_imgs_dict_folded['fold_2']}
 
#aggregate hem(healthyman) images

hem_imgs_dict_folded = {}   #with folds
properties = ['PAT_ID','NUM_IMG', 'NUM_CELL']

for c, fold in enumerate(folds):
    in_fold_dict = {}    
    for sample_id, img in enumerate(hem_imgs[c]):
        sample_dic = {}
        info = os.path.split(img)[-1].split('.')[0].split('_') 
        sample_dic['PATH'] = img
        for i, prop in enumerate(properties):
            sample_dic[prop] = info[i+1]
        sample_dic['CONDITION'] = 0
        in_fold_dict[os.path.split(img)[-1].split('.')[0]] = sample_dic
    hem_imgs_dict_folded[fold] = in_fold_dict
    
'''import this variable for future use -- hem_imgs_dict'''
#hem imgs without folds
hem_imgs_dict = {**hem_imgs_dict_folded['fold_0'],
                 **hem_imgs_dict_folded['fold_1'],
                 **hem_imgs_dict_folded['fold_2']}