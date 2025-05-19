#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 10:45:23 2021

@author: ubuntu
"""
"""
Created on Thu Sep 16 14:30:30 2021

@author: ubuntu
"""
import os
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
            name_dir = os.path.join(dataset_dir , 'train')
        elif group == 'query':
            name_dir = os.path.join(dataset_dir,'valid')
        else:
            name_dir = os.path.join(dataset_dir, 'test')
        file_list=os.listdir(name_dir)
        
        globals()[group]={}
        globals()[group]['data']=[]
        globals()[group]['ids'] = []
        for name in file_list:
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
    train_label=['Beard','no beard','beard others','Glasses','no Glasses','unsure Glasses','upper-short','upper-long','lower-short','lower-long',
                 'baby','child','Teenager','Youth','Middle-age','old','female','male','upper-black','upper-blue','upper-grey','upper-red',
                 'upper-white','upper-yellow','upper-others','lower-black','lower-blue','lower-grey','lower-red','lower-white','lower-yellow',
                 'lower-others','hari-short','hari-long','hari-hat','hari-others']
    
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
            key = j.strip().split(':')[0].split(' ')[0]
            values = j.strip().split(':')[1]
            dic[key] = int(values)
    return dic
def import_Session_binary(train):
    train_attribute = {}
    for i in range(len(train['data'])):
        list_attribute = []
        txt_name = train['data'][i][2] +'.txt'
        print('@@!!##',txt_name)
        dic = extract_txt(txt_name)

        if dic['Beard'] == 0:
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
        elif dic['Beard'] == 1:
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
        elif dic['Beard'] == 2:
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
         
        if dic['Glasses'] == 0:
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
        elif dic['Glasses'] == 1:
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
        elif dic['Glasses'] == 2:
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
        if dic['Upper-clothes'] == 0 :
            list_attribute.append(1)
            list_attribute.append(0)
        elif dic['Upper-clothes'] == 1:
            list_attribute.append(0)
            list_attribute.append(1)
        if dic['Lower-clothes'] == 0 :
            list_attribute.append(1)
            list_attribute.append(0)
        elif dic['Lower-clothes'] == 1:
            list_attribute.append(0)
            list_attribute.append(1)
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
        elif dic['Age'] ==5:
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
        if dic['Gender'] == 0 :
            list_attribute.append(1)
            list_attribute.append(0)
        elif dic['Gender'] == 1 :
            list_attribute.append(0)
            list_attribute.append(1)
        if dic['Upper-clothes-color'] == 0:
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        elif dic['Upper-clothes-color'] == 1 or dic['Upper-clothes-color'] == 7:
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        elif dic['Upper-clothes-color'] == 2:
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        elif dic['Upper-clothes-color'] == 3 or dic['Upper-clothes-color'] == 4:
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        elif dic['Upper-clothes-color'] == 5:
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
        elif dic['Upper-clothes-color'] == 6:
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
        elif dic['Upper-clothes-color'] == 8:
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
        
        if dic['Lower-clothes-color'] == 0:
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        elif dic['Lower-clothes-color'] == 1 or dic['Lower-clothes-color'] == 7:
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        elif dic['Lower-clothes-color'] == 2:
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        elif dic['Lower-clothes-color'] == 3 or dic['Lower-clothes-color'] == 4:
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        elif dic['Lower-clothes-color'] == 5:
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
        elif dic['Lower-clothes-color'] == 6:
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
        elif dic['Lower-clothes-color'] == 8:
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
        if dic['Hair'] == 0:
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
        elif dic['Hair'] == 1:
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
            list_attribute.append(0)
        elif dic['Hair'] == 2:
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
            list_attribute.append(0)
        elif dic['Hair'] == 3:
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(0)
            list_attribute.append(1)
        train_attribute[train['ids'][i]] = list_attribute         
    return train_attribute
