#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 14:30:30 2021

@author: ubuntu
"""
import os
data_dir = '/media/ubuntu/7934b444-78b3-4e80-b232-9aa3383920e4/dataset/linkdome_data/body_agender_datasets/'

path = os.listdir(data_dir)

print(os.path.join(data_dir,path[1]))
print(path)

def import_Session_nodistractors(data_dir, dataset_name):
    dataset_dir = os.path.join(data_dir,dataset_name)
    
    if not os.path.exists(dataset_dir):
        print('Please Download '+dataset_name+ ' Dataset')
        
    dataset_dir = os.path.join(data_dir,dataset_name)
    data_group = ['train','query','gallery']
    for group in data_group:
        if group == 'train':
            name_dir = os.path.join(dataset_dir , 'train')
        elif group == 'query':
            name_dir = os.path.join(dataset_dir,'query')
        else:
            name_dir = os.path.join(dataset_dir, 'gallery')
        dir_list=os.listdir(name_dir)
        
        globals()[group]={}
        globals()[group]['data']=[]
        globals()[group]['ids'] = []
        for i in range (len(dir_list)):
            print('list:',dir_list[i])
            dir_path = os.path.join(name_dir,dir_list[i])
            if os.path.isdir(dir_path):
                dir_path_list = os.listdir(dir_path)
                for j in range(len(dir_path_list)):
                    print('j:',j)
                    end_path = os.path.join(dir_path,dir_path_list[j])
                    if os.path.isdir(end_path):
                        file_list = os.listdir(end_path)
                        for name in file_list:
                            if name[-3:]=='jpg':
                                id = name.split('_')[0]
                                cam = name.split('_')[1] + '.' + name.split('_')[2] 
                                images = os.path.join(end_path,name)
                                globals()[group]['ids'].append(id)
                                globals()[group]['data'].append([images,globals()[group]['ids'].index(id),id,cam,name.split('.')[0]])
    return train, query, gallery
def import_SessionAttribute(data_dir,dataset_name):
    train,query,test = import_Session_nodistractors(data_dir,dataset_name)
    if not os.path.exists(os.path.join(data_dir,dataset_name)):
        print('Please Download the Market1501Attribute Dataset')
    train_label=['age',
           'gender']
    
    test_label=['age',
           'gender'
           ]
    '''
    train_person_id = []
    for personid in train:
        train_person_id.append(personid)
    print('train_person_id',train_person_id)
    train_person_id.sort(key=int)

    test_person_id = []
    for personid in test:
        test_person_id.append(personid)
    test_person_id.sort(key=int)
    test_person_id.remove('-1')
    test_person_id.remove('0000')
    '''
    #f = scipy.io.loadmat(os.path.join(dataset_dir,dataset_name,'market_attribute.mat'))
    query_attribute = import_Session_binary(query)
    train_attribute = import_Session_binary(train)
    test_attribute = import_Session_binary(test)
    label = train_label
    label.pop(0)
    label.insert(0,'baby')
    label.insert(1,'child')
    label.insert(2,'adolescent')
    label.insert(3,'young')
    label.insert(4,'YM')
    label.insert(5,'middle-age')
    label.insert(6,'old')
    return train_attribute,query_attribute,test_attribute,label
    
def import_Session_binary(train):
    train_attribute = {}
    for i in range(len(train['data'])):
        list_attribute = []
        age = train['data'][i][3].split('.')[1]
        if age == 'baby':
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        elif age == 'child':
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        elif age == 'adolescent':
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
         
        elif age == 'young':
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        elif age == 'YM':
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
        elif age =='middle-age':
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
        else:
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
        gender = train['data'][i][3].split('.')[0]
        if gender == 'woman':
            list_attribute.append(1)
        else: 
            list_attribute.append(0)
        train_attribute[train['ids'][i]] = list_attribute         
    return train_attribute

    

