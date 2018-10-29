#coding=utf-8
import tensorflow as tf
import utils
import facenet
import numpy as np
import random
import train_new_model
import os
import argparse
import time



model_path='/home/shikigan/kiwi_fung/label_web/models/facenet_cow/20180821-130405/'

def main(args):
    subdir = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Write arguments to a text file
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(model_path)
            images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
            embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')

            target=tf.placeholder(dtype=tf.int64,shape=(None,2),name='target_output')

            fc1=tf.layers.dense(inputs=embeddings,units=64,activation='relu',use_bias=True)
            fc2=tf.layers.dense(inputs=fc1,units=2,use_bias=True)
            res=tf.nn.softmax(logits=fc2)

            loss=tf.reduce_sum(tf.multiply(tf.subtract(res,target),tf.subtract(res,target)))

            train_step=tf.train.GradientDescentOptimizer(0.5).minimize(loss)

            tf.global_variables_initializer().run()


def make_train_batch(file_path,per_tf):
    train_set=[]
    label_set=[]
    dataset=utils.get_dataset_predit(file_path)

    for _class in dataset:
        for i in range(per_tf):
            train_set
