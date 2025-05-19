#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 12:06:42 2021

@author: ubuntu
"""
import os
from .import_Session import *
from .reiddataset_downloader import *
import scipy.io
import json

def import_Session_Attribute(data_dir, dataset_name):
    dataset_dir = os.path.join(data_dir,dataset_name)
    
    if not os.path.exists(dataset_dir):
        print('Please Download '+dataset_name+ ' Dataset')
        
    dataset_dir = os.path.join(data_dir,dataset_name)
    data_group = ['train','query','gallery']
    for group in data_group:
        if group == 'train':
            name_dir = os.path.join(dataset_dir , 'session_train')
            
        elif group == 'query':
            name_dir = os.path.join(dataset_dir, 'session_query')
            
        else:
            name_dir = os.path.join(dataset_dir,'session_gary')
            
        file_list=sorted(os.listdir(name_dir))
        globals()[group]={}
        globals()[group]['data']=[]
        globals()[group]['ids'] = []
        for name in file_list:
            if name[-4:]=='json':
                id = name.split('_')[3]+name.split('_')[4].split('.')[0]
                cam = int(name.split('_')[1][1])
                jsons = os.path.join(name_dir,name)
                if (id!='0000' and id !='-1'):
                    if id not in globals()[group]['ids']:
                        globals()[group]['ids'].append(id)
                    globals()[group]['data'].append([jsons,globals()[group]['ids'].index(id),id,cam,name.split('.')[0]])
    return train, query, gallery

def import_SessionAttribute(data_dir,dataset_name):
    train,query,test = import_Session_Attribute(data_dir,dataset_name)
    if not os.path.exists(os.path.join(data_dir,dataset_name)):
        print('Please Download the Market1501Attribute Dataset')
    train_label=['age',
           'downblack',
           'downblue',
           'downgray',
           'downred',
           'downwhite',
           'downyellow',
           'others',
           'upblack',
           'upblue',
           'upred',
           'upwhite',
           'upyellow',
           'upgray',
           'others',
           'top',
           'down',
           'hair',
           'glasses',
           'gender']
    
    test_label=['age',
           'down',
           'up',
           'hair',
           'glasses',
           'gender',
           'upblack',
           'upwhite',
           'upred',
           'uppurple',
           'upyellow',
           'upgray',
           'upblue',
           'upgreen',
           'downblack',
           'downwhite',
           'downpink',
           'downpurple',
           'downyellow',
           'downgray',
           'downblue',
           'downgreen',
           'downbrown'
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
    label.insert(0,'child')
    label.insert(1,'young')
    label.insert(2,'mid-age')
    label.insert(3,'old')
    label.insert(21,'long hair')
    label.insert(22,'wearing hat')
    label.insert(23,'others')
    return train_attribute,query_attribute,test_attribute,label
    
def import_Session_binary(train):
    train_attribute = {}
    for i in range(len(train['data'])):
        list_attribute = []
        with open(train['data'][i][0],'r') as f:
            json_list = json.load(f)
        age = json_list['age']
        if age == 'children' or age == 'students' or age == 'infants':
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        elif age == 'the youth':
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
        elif age =='middle age':
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
        else:
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
        
        down_color = json_list['bottom color']
        if down_color == 'black':
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        elif down_color == 'blue' or down_color == 'green' or down_color == 'cyan':
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        elif down_color == 'grey' or down_color == 'brown':
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        
        elif down_color == 'red' or down_color == 'purple' or down_color =='magenta':
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        elif down_color == 'white':
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
        elif down_color == 'yellow' or down_color == 'orange':
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
        top_color = json_list['top color']
        if top_color == 'black':
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        elif top_color == 'blue' or top_color == 'green' or top_color == 'cyan':
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        elif top_color == 'red'or down_color == 'purple' or down_color =='magenta':
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        elif top_color == 'white':
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        elif top_color == 'yellow' or top_color == 'orange':
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
        elif top_color == 'grey' or top_color == 'brown':
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
        top = json_list['top']
        if top == 'T shirt' or top == 'sleeveless':
            list_attribute.append(1)
        else:
            list_attribute.append(0)
        down = json_list['bottom']
        if down == 'shorts' or down =='short skirts':
            list_attribute.append(1)
        else:
            list_attribute.append(0)
        hair = json_list['hair style']
        if hair == 'short hair':
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        elif hair == 'long hair':
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
        elif hair == 'wearing hat':
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
        else:
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
        glasses = json_list['glasses']
        if glasses == 'glasses' or glasses == 'transparent glasses':
            list_attribute.append(1)
        else:
            list_attribute.append(0)
        gender = json_list['gender']
        if gender == 'female':
            list_attribute.append(1)
        else:
            list_attribute.append(0)
        train_attribute[train['ids'][i]] = list_attribute         
    return train_attribute