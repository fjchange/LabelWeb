#coding=utf-8
import facenet
import os
import sys
import math
import pickle
from scipy import misc
from six.moves import xrange
import argparse
import tensorflow as tf
import numpy as np
import utils
import random
path_pre='/home/shikigan/res/'
fence_list=[['03251704','03261556','03270845'],['03251053','03251734','03261040','03261601','03270849'],
            ['03241052','03241626','03241636','03251035','03251713','03251739','03261044','03261619','03270857','03271042'],
            ['03241051','03241634','03251055','03251737','03261042','03261615','03270854','03271028']]

output_path_pre='/home/shikigan/cow_head_set/emb/'
gap=0.

def cosine(q,a):
    pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 1))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 1))
    pooled_mul_12 = tf.reduce_sum(q * a, 1)
    score = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 +1e-8, name="scores")
    return score

def l2_dis(q,a):
    score=tf.sqrt(tf.reduce_sum(tf.square(q-a),1),name='l2_scores')
    return score


#main 函数主要做的是使用已经训练好的模型产生对应的向量，并以向量之间的距离给出向量的关系
def main(args):
    pairs=[]
    for root, dirs, paths in os.walk(args.csv_path):
        for path in paths:
            path=os.path.join(args.csv_path,path)
            lists=utils.make_pairs(utils.csv_solver(path),prob=0.3)
            #pairs.append(list)
            print(len(lists))
            random.shuffle(lists)
            with tf.Graph().as_default():

                with tf.Session() as sess:

                    # Load the model
                    facenet.load_model(args.model)
                    # Get input and output tensors
                    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                    total_negative_distance=0.
                    total_positive_distance=0.
                    count=0.
                    count_right=0.
                    #for lists in list:
                    for pair in lists:
                        emb = []
                        for item in pair:
                            image=misc.imread(item)
                            #这里的变形大小可能影响了实际的输出
                            aligned=misc.imresize(image,[160,160],interp='bilinear')
                            prewhitened=facenet.prewhiten(aligned)
                            #images.append(prewhitened)
                            # Run forward pass to calculate embeddings
                            images = np.stack(prewhitened).reshape([-1,160,160,3])

                            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                            emb.append( sess.run(embeddings, feed_dict=feed_dict))
                            #这里我们获得了三个向量，那么将这三个向量计算相互的角度距离
                        [a,p,n]=emb
                        count+=1.

                        dis_p=l2_dis(a,p)
                        dis_n=l2_dis(a,n)
                        total_negative_distance+=dis_n
                        total_positive_distance+=dis_p
                        distance_p,distance_n,total_p_dis,total_n_dis=sess.run([dis_p,dis_n,total_positive_distance,total_negative_distance])
                        if distance_n-distance_p>gap:count_right+=1
                        print(count,'\ndistance between anchor and positive case',distance_p)
                        print('distance between anchor and negative case',distance_n)
                        print('avg_dis_pos : ',total_p_dis/count)
                        print ('ave_dis_neg : ',total_n_dis/count)
                        print('right rate:',count_right/count)

def get_emb(csv_paths,model_path):
    '''

    :param csv_paths: a list of given paths of csv
    :param model:
    :return: a list of embeddings of the images [video_set[image]]
    '''
    embeds=[]
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(model_path)
            images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
            embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
            for path in csv_paths:
                #获取到anchor下面的csv文件下对应的dict
                labeled_dict=utils.csv_solver(path)
                emb=[]
                #这里所获得的是一个list,按照key从小到大排列，[(id,[]),(id,[]),....]
                labeled_dict=sorted(labeled_dict.items(), key=lambda d:d[0],reverse=False)
                #读取图片，然后产生图片的向量
                for _class in labeled_dict:
                    images=[]
                    for item in _class[-1]:
                        image=misc.imread(item)
                        aligned=misc.imresize(image,[160,160],interp='bilinear')
                        prewhitened=facenet.prewhiten(aligned)
                        images.append(prewhitened)
                    images=np.stack(images).reshape([-1,160,160,3])
                    feed_dict={images_placeholder:images,phase_train_placeholder:False}
                    emb_res=sess.run(embeddings,feed_dict=feed_dict)
                    emb.append(emb_res)
                print('------> got embeddings of ',path.split('.')[0])
                embeds.append(emb)
    return embeds

def get_embedding(file_exp,model_path):
    emb=[]
    labels=[]
    with tf.Graph().as_default():
        with tf.Session()as sess:
            facenet.load_model(model_path)
            images_placeholder=tf.get_default_graph().get_tensor_by_name('input:0')
            embeddings=tf.get_default_graph().get_tensor_by_name('embeddings:0')
            phase_train_placeholder=tf.get_default_graph().get_tensor_by_name('phase_train:0')

            dataset=utils.get_dataset_predit(file_exp)

            print(len(dataset))
            for i,_class in enumerate(dataset):
                images=[]
                for item in _class:
                    image = misc.imread(item)
                    aligned = misc.imresize(image, [160, 160], interp='bilinear')
                    prewhitened = facenet.prewhiten(aligned)
                    images.append(prewhitened)
                images = np.stack(images).reshape([-1, 160, 160, 3])

                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_res=sess.run(embeddings,feed_dict=feed_dict)
                emb.append(emb_res)
                labels.append(i)
    return emb,labels

def get_emb_txt(csv_paths,model_path):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            facenet.load_model(model_path)
            images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
            embeddings = tf.get_default_graph().get_tensor_by_name('embeddings:0')
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
            for path in csv_paths:
                #获取到anchor下面的csv文件下对应的dict
                labeled_dict=utils.csv_solver(path)
                emb=[]
                #这里所获得的是一个list,按照key从小到大排列，[(id,[]),(id,[]),....]
                labeled_dict=sorted(labeled_dict.items(), key=lambda d:d[0],reverse=False)
                #读取图片，然后产生图片的向量
                for _class in labeled_dict:
                    images=[]
                    for item in _class[-1]:
                        image=misc.imread(item)
                        aligned=misc.imresize(image,[160,160],interp='bilinear')
                        prewhitened=facenet.prewhiten(aligned)
                        images.append(prewhitened)
                    images=np.stack(images).reshape([-1,160,160,3])
                    feed_dict={images_placeholder:images,phase_train_placeholder:False}
                    emb_res=sess.run(embeddings,feed_dict=feed_dict)
                    emb.append(emb_res)
                print('------> got embeddings of ',path.split('.')[0])
                output_path = output_path_pre + path.split('/')[-1].split('.')[0] + '.txt'
                emb_txt(output_path, emb)

def emb_txt(output_path,emb):
    fp=open(output_path,'w')
    id=1
    for _class in emb:
        fp.writelines([str(id)])
        for c in _class:
            for item in c:
                fp.write(str(item))
                fp.write(',')
        id+=1
    fp.close()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                        default='/home/shikigan/kiwi_fung/label_web/models/facenet_cow/20180719-174547/')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--csv_path',type=str,
                        help='The dir of the labeled csv',default='/home/shikigan/kiwi_fung/label_data')
    return parser.parse_args(argv)


if __name__ == '__main__':
    #main(parse_arguments(sys.argv[1:]))
    for root,dirs,paths in os.walk('/home/shikigan/kiwi_fung/label_data/'):
        new_paths=[]
        for path in paths:
            new_paths.append('/home/shikigan/kiwi_fung/label_data/'+path)
        get_emb_txt(new_paths,'/home/shikigan/kiwi_fung/label_web/models/facenet_cow/20180719-174547/')
