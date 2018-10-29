#coding=utf-8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import os
import sys
import facenet
from tensorflow.python.ops import data_flow_ops
from scipy.optimize import brentq
from scipy import interpolate

path_pre='/home/shikigan/res/'



#获得csv对应的文件夹的路径
def csv_solver(path):
    with open(path)as c:
        path_dict={}
        lines=c.readlines()
        for line in lines:
            line_list=line.split(',')
            if int(line_list[-1])==-1:
                continue
            if path_dict.has_key(int(line_list[-1])):
                path_dict[int(line_list[-1])].append(path_pre+line_list[0].split('\'')[-1]+'/'+line_list[2]+'.jpg')
            else:
                path_dict[int(line_list[-1])]=[path_pre+line_list[0].split('\'')[-1]+'/'+line_list[2]+'.jpg']
    return path_dict

#对于每一个视频文件夹中的图片进行配对（A,P,N)
'''
    如果在dict中共有m个label,总共有n个,选中的label中有n_i个，sum_i(n_i(n_i-1)*n)个匹配
'''
def make_pairs(dict):
    pairs=[]
    for key in dict.keys():
        for item in dict[key]:
            for _item in dict[key]:
                if _item!=item:
                    for _key in dict.keys():
                        if _key!=key:
                            for __item in dict[_key]:
                                pairs.append((item,_item,__item))
    return pairs

def main(args):
    #创建图
    #获得path下面的所有的csv文件路径
    #pairs [list] list是每一个视频下面的三元组list
    pairs=[]
    for paths,dir,root in os.walk(args.csv_path):
        for path in paths:
            list=make_pairs(csv_solver(path))
            pairs.append(list)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            #placeholder setting

            image_paths_placeholder=tf.placeholder(tf.string,shape=(None,1),name='image_paths')
            labels_placeholder=tf.placeholder(tf.int32,shape=(None,1),name='labels')
            batch_size_placeholder=tf.placeholder(tf.int32,name='batch_size')
            control_placeholder=tf.placeholder(tf.int32,shape=(None,1),name='control')
            phase_train_placeholder=tf.placeholder(tf.bool,name='phase_train')

            #something relative to image preprocess
            nrof_preprocess_threads=4
            image_size=(args.image_size,args.image_size)
            eval_input_queue=data_flow_ops.FIFOQueue(capacity=10000000,
                                                     dtype=[tf.string])
