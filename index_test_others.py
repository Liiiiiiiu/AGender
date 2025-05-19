#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:28:43 2021

@author: ubuntu
"""
from __future__ import division
import os
import argparse
import scipy.io
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from datafolder.folder_other import Test_Dataset
from net import get_model
import matplotlib.pyplot as plt
from pycm import *
######################################################################
# Settings
# ---------
use_gpu = True
dataset_dict = {
    'market'  :  'Market-1501',
    'duke'  :  'DukeMTMC-reID',
    'data_sets' : 'data_sets' ,
    'attri_bot_cox' : 'attri_bot_cox'
}
num_cls_dict = { 'market':30, 'duke':23 , 'attri_box_cox':28 ,'data_sets':36}
num_ids_dict = { 'market':751, 'duke':702 ,'attri_bot_cox':10000,'data_sets':10000}
classes = ['Beard','no beard','beard others','Glasses','no Glasses','unsure Glasses','upper-short','upper-long','lower-short','lower-long',
                 'baby','child','Teenager','Youth','Middle-age','old','female','male','upper-black','upper-blue','upper-grey','upper-red',
                 'upper-white','upper-yellow','upper-others','lower-black','lower-blue','lower-grey','lower-red','lower-white','lower-yellow',
                 'lower-others','hari-short','hari-long','hari-hat','hari-others']

######################################################################
# Argument
# ---------
parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--data-path', default='/media/ubuntu/7934b444-78b3-4e80-b232-9aa3383920e4/dataset/', type=str, help='path to the dataset')
parser.add_argument('--dataset', default='attri_bot_cox', type=str, help='dataset')
parser.add_argument('--backbone', default='resnet50', type=str, help='model')
parser.add_argument('--batch-size', default=50, type=int, help='batch size')
parser.add_argument('--num-epoch', default=60, type=int, help='num of epoch')
parser.add_argument('--num-workers', default=2, type=int, help='num_workers')
parser.add_argument('--which-epoch',default='last2', type=str, help='0,1,2,3...or last')
parser.add_argument('--print-table',action='store_true', help='print results with table format')
parser.add_argument('--use-id', action='store_true', help='use identity loss')
args = parser.parse_args()

assert args.dataset in ['market', 'duke','attri_bot_cox','data_sets']
assert args.backbone in ['resnet50', 'resnet34', 'resnet18', 'densenet121']

dataset_name = dataset_dict[args.dataset]
#id model test_id
#model_name = '{}_nfc_id'.format(args.backbone)
model_name = '{}_nfc_id'.format(args.backbone) if args.use_id else '{}_nfc'.format(args.backbone)
data_dir = args.data_path
model_dir = os.path.join('./checkpoints/data_sets', model_name)
result_dir = os.path.join('./result', args.dataset, model_name)

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)


######################################################################
# Function
# ---------
def load_network(network):
    save_path = os.path.join(model_dir,'net_%s.pth'%args.which_epoch)
    network.load_state_dict(torch.load(save_path))
    print('Resume model from {}'.format(save_path))
    return network


def get_dataloader():
    image_datasets = {}
    image_datasets['gallery'] = Test_Dataset(data_dir, dataset_name=dataset_name, query_gallery='gallery')
    image_datasets['query'] = Test_Dataset(data_dir, dataset_name=dataset_name, query_gallery='query')
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.num_workers)
                  for x in ['gallery', 'query']}
    return dataloaders


def check_metric_vaild(y_pred, y_true):
    if y_true.min() == y_true.max() == 0:   # precision
        return False
    if y_pred.min() == y_pred.max() == 0:   # recall
        return False
    return True


######################################################################
# Load Data
# ---------
# Note that we only perform evaluation on gallery set.
test_loader = get_dataloader()['query']

attribute_list = test_loader.dataset.labels()
num_label = len(attribute_list)
num_sample = len(test_loader.dataset)
num_id = num_ids_dict[args.dataset]

######################################################################
# Model
# ---------
model = get_model(model_name, num_label, use_id=args.use_id, num_id=num_id)
model = load_network(model)
if use_gpu:
    model = model.cuda()
model.train(False)  # Set model to evaluate mode


######################################################################
# Testing
# ---------
preds_tensor = np.empty(shape=[0, num_label], dtype=np.byte)   # shape = (num_sample, num_label)
labels_tensor = np.empty(shape=[0, num_label], dtype=np.byte)   # shape = (num_sample, num_label)

# Iterate over data.
with torch.no_grad():
    for count, (images, labels, ids, file_name) in enumerate(test_loader):
        # move input to GPU
        if use_gpu:
            images = images.cuda()
        # forward
        if not args.use_id:
            pred_label = model(images)
        else:
            pred_label, _ = model(images)

        preds = torch.gt(pred_label, torch.ones_like(pred_label)/2)
        # transform to numpy format
        labels = labels.cpu().numpy()
        preds = preds.cpu().numpy()
        # append
        preds_tensor = np.append(preds_tensor, preds, axis=0)
        labels_tensor = np.append(labels_tensor, labels, axis=0)
        # print info
        if count*args.batch_size % 5000 == 0:
            print('Step: {}/{}'.format(count*args.batch_size, num_sample))
# Evaluation.
'''
def np_transform(array):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i][j]== 1:
                array[i][j] = j
            elif array[i][j] == 0:
                
                array[i][j] = array.shape[1]
    array_new = array.flatten()
    return array_new

def delete_ele(array,n):
    
    array_new = np.delete(array,np.where(array == n))
    return array_new
def nomalization(array):
    temp = np.zeros((array.shape[0],array.shape[1]))
    for i in range(array.shape[0]):
        sum_arr = np.sum(array[i])
        for j in range(array.shape[1]):
             temp[i][j] ='%.2f'%(array[i][j]/sum_arr)
    return temp
'''            
preds_tensor_test = preds_tensor
def num_list(array,index1,index2):
    num = 0
    for j in range(index1,index2):
        if array[j] == 1:
            num += 1
    return num
array_list = []
for i in range(preds_tensor_test.shape[0]):        
    temp =[]
    temp.append(num_list(preds_tensor_test[i],0,3)) #beard
    temp.append(num_list(preds_tensor_test[i],3,6)) #glasses
    temp.append(num_list(preds_tensor_test[i],6,8)) #up
    temp.append(num_list(preds_tensor_test[i],8,10)) #lower
    temp.append(num_list(preds_tensor_test[i],10,16)) # age
    temp.append(num_list(preds_tensor_test[i],16,18)) #gender
    temp.append(num_list(preds_tensor_test[i],18,25)) #upper-color
    temp.append(num_list(preds_tensor_test[i],25,32)) #lower-color
    temp.append(num_list(preds_tensor_test[i],32,36)) #hair style
    array_list.append(temp)
age_err =[]
downcolor_err = []
upcolor_err = []
up_err = []
down_err = []
hair_err = []
glass_err = []
gender_err = []
beard_err = []
for i in range(len(array_list)):
    if array_list[i][0] !=1:
        beard_err.append(i)
    if array_list[i][1] != 1:
        glass_err.append(i)
    if array_list[i][2] != 1:
        up_err.append(i)
    if array_list[i][3] != 1:
        down_err.append(i)
    if array_list[i][4] != 1:
        age_err.append(i)
    if array_list[i][5] != 1:
        gender_err.append(i)
    if array_list[i][6] != 1:
        upcolor_err.append(i)
    if array_list[i][7] != 1:
        downcolor_err.append(i)
    if array_list[i][8] != 1:
        hair_err.append(i)
print('#age%:',len(age_err))
print('#downcolor%:',len(downcolor_err))
print('#upcolor%:',len(upcolor_err))
print('@uperror*:',len(up_err))
print('@down_err*:',len(down_err))
print('@hair#:',len(hair_err))
print('@galss#:',len(glass_err))
print('!gender*:',len(gender_err))
print('beard',len(beard_err))
#plot confusion_matrix
print('@@@')
preds_tensor_beard = preds_tensor[:,0:3]
labels_tensor_beard = labels_tensor[:,0:3]
preds_tensor_true = np.delete(preds_tensor_beard,beard_err,axis=0)
labels_tensor_true = np.delete(labels_tensor_beard,beard_err,axis=0)
preds_tensor_glass = preds_tensor[:,3:6]
labels_tensor_glass = labels_tensor[:,3:6]
preds_glass_true = np.delete(preds_tensor_glass,glass_err,axis=0)
labels_glass_true = np.delete(labels_tensor_glass,glass_err,axis=0)
preds_tensor_up = preds_tensor[:,6:8]
labels_tensor_up = labels_tensor[:,6:8]
preds_up_true = np.delete(preds_tensor_up,up_err,axis=0)
labels_up_true = np.delete(labels_tensor_up,up_err,axis=0)
preds_tensor_down = preds_tensor[:,8:10]
labels_tensor_down = labels_tensor[:,8:10]
preds_down_true = np.delete(preds_tensor_down,down_err,axis=0)
labels_down_true = np.delete(labels_tensor_down,down_err,axis=0)
preds_tensor_age = preds_tensor[:,10:16]
labels_tensor_age = labels_tensor[:,10:16]
preds_age_true = np.delete(preds_tensor_age,age_err,axis=0)
labels_age_true = np.delete(labels_tensor_age,age_err,axis=0)
preds_tensor_gender = preds_tensor[:,16:18]
labels_tensor_gender = labels_tensor[:,16:18]
preds_gender_true = np.delete(preds_tensor_gender,gender_err,axis=0)
labels_gender_true = np.delete(labels_tensor_gender,gender_err,axis=0)
preds_tensor_upcolor = preds_tensor[:,18:25]
labels_tensor_upcolor = labels_tensor[:,18:25]
preds_upcolor_true = np.delete(preds_tensor_upcolor,upcolor_err,axis=0)
labels_upcolor_true = np.delete(labels_tensor_upcolor,upcolor_err,axis=0)
preds_tensor_downcolor = preds_tensor[:,25:32]
labels_tensor_downcolor = labels_tensor[:,25:32]
preds_downcolor_true = np.delete(preds_tensor_downcolor,downcolor_err,axis=0)
labels_downcolor_true = np.delete(labels_tensor_downcolor,downcolor_err,axis=0)
preds_tensor_hair = preds_tensor[:,32:36]
labels_tensor_hair = labels_tensor[:,32:36]
preds_hair_true = np.delete(preds_tensor_hair,hair_err,axis=0)
labels_hair_true = np.delete(labels_tensor_hair,hair_err,axis=0)
def plot_confusion_matrix(cm,savename,classes,title='Confusion Matrix'):
    plt.figure(figsize=(12,8),dpi = 100)
    np.set_printoptions(precision=2)
    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()
def np_transform(array):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i][j]== 1:
                array[i][j] = j
            elif array[i][j] == 0:
                
                array[i][j] = array.shape[1]
    array_new = array.flatten()
    return array_new
def delete_ele(array,n):
    
    array_new = np.delete(array,np.where(array == n))
    return array_new
def np_transform_age(array,arr_err):
    for i in range(array.shape[0]):
        if i not in arr_err:
            for j in range(array.shape[1]):
                if array[i][j] == 1:
                    array[i][j] = j
                elif array[i][j] == 0:
                    array[i][j] = array.shape[1]
        else:
            for n in range(array.shape[1]):
                if n ==0:
                    array[i][0] = array.shape[1]+1
                else:
                    array[i][n] = array.shape[1]
    array_new = array.flatten()
    return array_new
def nomalization(array):
    temp = np.zeros((array.shape[0],array.shape[1]))
    for i in range(array.shape[0]):
        sum_arr = np.sum(array[i])
        for j in range(array.shape[1]):
             temp[i][j] ='%.2f'%(array[i][j]/sum_arr)
    return temp

def check_zeor_one(array):
    temp = 0
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i][j] == 1:
                temp += 1
    return temp

#去掉异常值
labels_beard_delete = delete_ele(np_transform(labels_tensor_true),3)
preds_beard_delete =  delete_ele(np_transform(preds_tensor_true),3)
classes_beard = classes[0:3]

#cm = confusion_matrix(labels_beard_delete,preds_beard_delete)
#plot_confusion_matrix(nomalization(cm),'beard_version_120epoch.png',classes_beard)
labels_glasses_delete = delete_ele(np_transform(labels_glass_true),3)
preds_glasses_delete =  delete_ele(np_transform(preds_glass_true),3)
classes_glass = classes[3:6]
cm_glass = confusion_matrix(labels_glasses_delete,preds_glasses_delete)
plot_confusion_matrix(nomalization(cm_glass),'glasses_version_120epoch.png',classes_glass)

labels_up_delete = delete_ele(np_transform(labels_up_true),2)
preds_up_delete =  delete_ele(np_transform(preds_up_true),2)
classes_up = classes[6:8]
cm_up = confusion_matrix(labels_up_delete,preds_up_delete)
plot_confusion_matrix(nomalization(cm_up),'up_version_120epoch.png',classes_up)

labels_down_delete = delete_ele(np_transform(labels_down_true),2)
preds_down_delete =  delete_ele(np_transform(preds_down_true),2)
classes_down = classes[8:10]
cm_down = confusion_matrix(labels_down_delete,preds_down_delete)
plot_confusion_matrix(nomalization(cm_down),'down_version_120epoch.png',classes_down)

labels_age_delete = delete_ele(np_transform(labels_age_true),6)
preds_age_delete =  delete_ele(np_transform(preds_age_true),6)
classes_age = classes[10:16]
cm_age = confusion_matrix(labels_age_delete,preds_age_delete)
plot_confusion_matrix(nomalization(cm_age),'age_version_120epoch.png',classes_age)

labels_gender_delete = delete_ele(np_transform(labels_gender_true),2)
preds_gender_delete =  delete_ele(np_transform(preds_gender_true),2)
classes_gender = classes[16:18]
cm_gender = confusion_matrix(labels_gender_delete,preds_gender_delete)
plot_confusion_matrix(nomalization(cm_gender),'gender_version_120epoch.png',classes_gender)

labels_upcolor_delete = delete_ele(np_transform(labels_upcolor_true),7)
preds_upcolor_delete =  delete_ele(np_transform(preds_upcolor_true),7)
classes_upcolor = classes[18:25]
cm_upcolor = confusion_matrix(labels_upcolor_delete,preds_upcolor_delete)
plot_confusion_matrix(nomalization(cm_upcolor),'upcolor_version_120epoch.png',classes_upcolor)

labels_hair_delete = delete_ele(np_transform(labels_hair_true),4)
preds_hair_delete =  delete_ele(np_transform(preds_hair_true),4)
classes_hair = classes[32:36]
cm_hair = confusion_matrix(labels_hair_delete,preds_hair_delete)
plot_confusion_matrix(nomalization(cm_hair),'hair_version_120epoch.png',classes_hair)

print('%%^^&&**',labels_downcolor_true.shape)
print('**&&^^%%',preds_downcolor_true.shape)
print('1 ::',check_zeor_one(labels_downcolor_true))
labels_downcolor_delete = delete_ele(np_transform(labels_downcolor_true),7)
preds_downcolor_delete =  delete_ele(np_transform(preds_downcolor_true),7)
classes_downcolor = classes[25:32]
cm_downcolor = confusion_matrix(labels_downcolor_delete,preds_downcolor_delete)
plot_confusion_matrix(nomalization(cm_downcolor),'downcolor_version_120epoch.png',classes_downcolor)
