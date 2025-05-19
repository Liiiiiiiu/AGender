#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 15:44:13 2021

@author: ubuntu
"""
import os
import time
data_dir = '/media/ubuntu/7934b444-78b3-4e80-b232-9aa3383920e4/test/test_qt/person_attribute_index_version4/data_sets/'

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
            name_dir = os.path.join(dataset_dir , 'test')
        elif group == 'query':
            name_dir = os.path.join(dataset_dir,'test')
        else:
            name_dir = os.path.join(dataset_dir, 'test')
        file_list=os.listdir(name_dir)
        
        globals()[group]={}
        globals()[group]['data']=[]
        globals()[group]['ids'] = []
        for name in file_list:
            print('**',name)
            if name[-3:]=='jpg':
                id = name.split('.')[0]
                cam = name.split('.')[1] 
                images = os.path.join(name_dir,name)
                globals()[group]['ids'].append(id)
                globals()[group]['data'].append([images,globals()[group]['ids'].index(id),id,cam,name.split('.')[0]])
    return train, query, gallery
def import_indexAttribute(data_dir,dataset_name):
    train,query,test = import_Session_nodistractors(data_dir,dataset_name)
    if not os.path.exists(os.path.join(data_dir,dataset_name)):
        print('Please Download the Market1501Attribute Dataset')
    train_label=['baby','child','Teenager','Youth','Middle-age','old','gender']
    
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
    return train_attribute,query_attribute,test_attribute,label

def extract_txt(file):
    dic = {}
    print('@@@:','/media/ubuntu/7934b444-78b3-4e80-b232-9aa3383920e4/test/test_qt/person_attribute_index_version4/data_sets/txt/' + file)
    with open('/media/ubuntu/7934b444-78b3-4e80-b232-9aa3383920e4/test/test_qt/person_attribute_index_version4/data_sets/txt/' + file,'r') as file_read:
        lines = file_read.readlines()
        for j in lines:
            if len(j.strip().split(':')[0].split(' ')) == 1:
                continue
            key = j.strip().split(':')[0].split(' ')[0]
            values = j.strip().split(':')[1]
            dic[key] = int(values)
    return dic
def import_Session_binary(train):
    a = 0
    b =0
    train_attribute = {}
    for i in range(len(train['data'])):
        list_attribute = []
        txt_name = train['data'][i][2] +'.txt'
        print('@@!!##',txt_name)
        dic = extract_txt(txt_name)

        if dic['Age'] == 0:
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        elif dic['Age'] ==1:
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        elif dic['Age'] ==2:
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        elif dic['Age'] ==3:
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
        elif dic['Age'] ==4:
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
        elif dic['Age'] ==5 or dic['Age'] == 6:
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            
        if dic['Gender'] == 0:
            a += 1
            list_attribute.append(0)
        else: 
            b += 1
            list_attribute.append(1)
        
        train_attribute[train['ids'][i]] = list_attribute
    print('a,b value:',a,b)
    time.sleep(10)          
    return train_attribute
