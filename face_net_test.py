#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import facenet
import tensorflow as tf
import utils
import random


def evaluate(embeddings,actual_issame,nrof_folds=10,distance_metric=0,subtract_mean=False):
    thresholds=np.arange(0,4,0.01)
    embeddings1=embeddings[0::2]
    embeddings2=embeddings[1::2]

    tpr,fpr,accuracy=facenet.calculate_roc(thresholds,embeddings1,embeddings2,
                                           np.asarray(actual_issame),nrof_folds=nrof_folds,distance_metric=distance_metric,
                                           subtract_mean=subtract_mean)
    thresholds=np.arange(0,4,0.001)
    val,val_std,far=facenet.calculate_val(thresholds,embeddings1,embeddings2,
                                          np.asarray(actual_issame),1e-3,nrof_folds=nrof_folds,distance_metric=distance_metric,
                                          subtract_mean=subtract_mean)
    return tpr,fpr,accuracy,val,val_std,far

def read_pairs(txt_path):
    images=[]
    labels=[]
    with open(txt_path)as f:
        lines=f.readlines()
        for line in lines:
            list=line.strip().split(' ')
            list[-1]=list[-1].split('\n')[0]
            list[2]=int(list[2])
            images.append([list[0],list[1]])
            if list[2]:
                labels.append(True)
    return images,labels


def make_pairs(file_path,per_tf,out_put_file_path):
    dataset=utils.get_dataset_predit(file_path)
    output_data=[]
    with open(out_put_file_path,'w')as f:
        for i in range(10):
            temp_output_data=[]
            for _class in dataset:
                for j in range(per_tf):
                    id1,id2=get_two_differ_item(len(_class))
                    temp_output_data.append([_class[id1],_class[id2],'1'])
            for i,_class in enumerate(dataset):
                for j in range(per_tf):
                    other=get_other(i,len(dataset))
                    item1,item2=get_two_random_item(_class,dataset[other])
                    temp_output_data.append([item1,item2,'0'])
            random.shuffle(temp_output_data)
            output_data=output_data+temp_output_data
        for item in output_data:
            for i in range(3):
                f.write(item[i])
                f.write(' ')
            f.write('\n')

def get_other(index,len):
    id=index
    while(id==index):
        id=int(random.random()*len)
    return id

def get_two_random_item(class1,class2):
    return class1[int(random.random()*len(class1))],class2[int(random.random()*len(class2))]

def get_two_differ_item(len):
    id1=int(random.random()*len)
    id2=id1
    count=0
    while(id1==id2):
        id2=int(random.random()*len)
        count+=1
        if count>10:
            print('dead loop')
            break

    return id1,id2

file_path='/home/shikigan/out_res_5/'
output_file_path='/home/shikigan/pairs.txt'

if __name__=='__main__':
    make_pairs(file_path,10,output_file_path)