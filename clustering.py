#coding=utf-8
import predit
import tensorflow as tf
import numpy as np
import utils
import random
from tkinter import *
from PIL import Image,ImageTk


csv_exp='/home/shikigan/kiwi_fung/labeled_data/'
model='/home/shikigan/kiwi_fung/label_web/models/facenet_cow/20180821-130405/'
#输出的特征向量的长度（如果模型不一样，这个需要作为参数）
BOTTLENECK_WIDTH=128
#计算每一个视频截图集下同一组图片的group center，评测group_center 距离各图片的距离
'''
def grouping(csv_exp,model):
    dict=predit.get_emb(csv_exp,model)
    groups={}
    for key in dict.keys():
        group_cens = []
        count=len(dict[key])
        print('-------------------')
        print(key,count)
        for _class in dict[key]:
            group_sum =np.zeros(128)
            for item in _class:
                group_sum+=np.array(item)
            avg_group=group_sum/float(count)
            #print(avg_group)
            total_distance=0.
            for item in _class:
                total_distance+=l2_dis(avg_group,item)
            avg_distance=total_distance/float(count)
            print(avg_distance)
            group_cens.append(avg_group)
        groups[key]=group_cens
    return groups
'''

def l2_dis(q,a):
    #score=tf.sqrt(tf.reduce_sum(tf.square(q-a),1),name='l2_scores')
    #a=np.repeat(a,q.shape[0],axis=1)
    score=np.sqrt(np.sum(np.square(q-a)))
    return score
#无法使用马氏距离，毕竟样本的维度128远大于样本的数量，


def distance_ranking(anchor_group_cen,gallery_group_list):
    '''

    :param anchor_group_cen: 给定的目标向量的中心
    :param gallery_group_list: 待选集合的list
    :return: 返回的是筛选好的字典，里面把待选集合以id：路径 记录下来
    '''
    gallery_group_dict={}
    count=0
    for item in gallery_group_list:
        gallery_group_dict[count]=l2_dis(anchor_group_cen,item)
        count+=1
    #Attention ! 这里返回的是一个list: 这个list是包括[(id,distance)...](距离从小到大）
    sorted_dict=sorted(gallery_group_dict.items(),key=lambda d:d[1],reverse=False)
    return sorted_dict

#以重心作为每一类的中心，这可能有更好的选择方法
def get_groups_cens(groups_path):
    embeddings_res=predit.get_emb(groups_path,model_path=model)
    groups_cens=[]
    for _subset in embeddings_res:
        group_cens=[]
        for group in _subset:
            group_sum = np.zeros(BOTTLENECK_WIDTH)
            for item in group:
                group_sum+=np.array(item)
            group_cen=group_sum/float(len(group))
            group_cens.append(group_cen)
        groups_cens.append(group_cens)
    return groups_cens

#K 是一个超参,是指我们先尝试对前K个最有可能的向量进行人工标定在进行下一组，
def matching_process(anchor_groups_path,gallery_groups_path,K=1):
    #匹配后的结果
    group_pairs=[]
    anchor_groups_cens,gallery_groups_cens=get_groups_cens([anchor_groups_path,gallery_groups_path])

    anchor_dict=utils.csv_solver(anchor_groups_path)
    gallery_dict=utils.csv_solver(gallery_groups_path)

    #如果命中的话就置0
    #print(len(gallery_dict.keys()))
    shot_gallery_index_list=np.ones(len(gallery_dict.keys()))
    #print(shot_gallery_index_list)

    distance_rankings=[]
    #产生一个m*n的矩阵
    for item in anchor_groups_cens:
        distance_rankings.append(distance_ranking(item,gallery_groups_cens))

    #非空的数目用来
    not_empty_count=len(distance_rankings)

    while not_empty_count:
        #通过给定的path找到对应的group中的图片，实现挑出两个group中的任意图片给用户进行匹配
        for i in range (len(distance_rankings)):
            dis_vector=distance_rankings[i]
            if len(dis_vector)!=0:
                shoot = False
                for j in range(0,min(K,len(dis_vector))):
                    #item （id,distance）
                    item=dis_vector[j]
                    #！！！这里的i对应的吗
                    anchor_image_path=anchor_dict[i][int(random.random()*len(anchor_dict[i]))]

                    gallery_image_path=gallery_dict[item[0]][int(random.random()*len(gallery_dict[item[0]]))]
                    #interact with front end
                    label_result=interaction_with_human_labeling(anchor_image_path,gallery_image_path)

                    if label_result:
                        #剪枝
                        distance_rankings[i]=[]
                        not_empty_count-=1
                        temp_distance_rankings=[]
                        for dis_vec in distance_rankings:
                            temp_dis_ranking=[]
                            for _item in dis_vec:
                                if _item[0]!=item[0]:
                                    temp_dis_ranking.append(_item)
                            temp_distance_rankings.append(temp_dis_ranking)
                        distance_rankings=temp_distance_rankings
                        #pairs 匹配的是（anchor_id,gallery_id）
                        group_pairs.append((i,item[0]))
                        shoot=True
                        #命中就在index_list中剔除那一项
                        print(item[0])
                        shot_gallery_index_list[item[0]]=0
                        break
                #如果都没有命中的话，那么就要把K之前的都去掉
                if not shoot:
                    if K<len(dis_vector):
                        distance_rankings[i]=distance_rankings[i][K:]
                    else:
                        distance_rankings[i]=[]
                        not_empty_count-=1
                        group_pairs.append((i,None))

    #gallery中没有命中的都会以(None,id)加入到group_pairs中
    for i in range(len(shot_gallery_index_list)) :
        if shot_gallery_index_list[i]:
            group_pairs.append((None,i))


    #后续可能需要写回csv文件中
    return group_pairs,anchor_dict,gallery_dict

#最好写回的是相对路径，不然以后没法用
def write_res(res,csv_path):
    with open(csv_path)as c:
        for key in res.keys():
            for item in res[key]:
                c.writelines([item,key])

def match(anchor_groups_path,gallery_groups_path,output_csv_path,K=5):
    res_pairs,anchor_dict,gallery_dict=matching_process(anchor_groups_path,gallery_groups_path,K)
    #整合两个dict
    res_dict={}
    for i in range(len(res_pairs)):
        if res_pairs[i][0]!=None and res_pairs[i][-1]!=None:
            res_dict[i]=anchor_dict[res_pairs[i][0]]+gallery_dict[res_pairs[i][-1]]
        elif res_pairs[i][0]==None:
            res_dict[i]=gallery_dict[res_pairs[i][-1]]
        else:
            res_dict[i]=gallery_dict[res_pairs[i][0]]
    write_res(res_dict,output_csv_path)

class simple_front_end():
    def __init__(self):
        self.root=Tk()
        self.root.title('label_web_simple_interact.v0.1')
        self.root.geometry('1080x920')
        self.build_win()

    def set_image(self,anchor_path,gallery_path):
        self.ans=None
        self.anchor = anchor_path
        self.gallery = gallery_path
        print(anchor_path)
        print(gallery_path)
        anchor_jpg = Image.open(anchor_path)
        gallery_jpg = Image.open(gallery_path)
        anchor = ImageTk.PhotoImage(anchor_jpg)
        self.pic[0].configure(image=anchor)
        gallery = ImageTk.PhotoImage(gallery_jpg)
        self.pic[1].configure(image=gallery)
        self.root.mainloop()


    def click_same(self):
        self.ans=True
        self.root.destroy()

    def click_differ(self):
        self.ans=False
        self.root.destroy()
    def build_win(self):
        self.pic=list()
        self.pic.append(Label(self.root,text='anchor'))
        self.pic.append(Label(self.root,text='gallery'))

        self.pic[0].grid(row=0,column=0)
        self.pic[1].grid(row=0,column=1)

        self.but=list()
        self.but.append(Button(self.root,text='same',command=self.click_same))
        self.but.append(Button(self.root,text='differ',command=self.click_differ))

        self.but[0].grid(row=1,column=0)
        self.but[1].grid(row=1,column=1)





def interaction_with_human_labeling(anchor_image_path,gallery_image_path):
    #human do some action,and return the result of labeling
    #return True/False
    front_end=simple_front_end()
    front_end.set_image(anchor_image_path,gallery_image_path)
    print(front_end.ans)
    return front_end.ans

if __name__=='__main__':
    match('/home/shikigan/kiwi_fung/label_data_1/03270849.csv','/home/shikigan/kiwi_fung/label_data_1/03261601.csv','/home/shikigan/kiwi_fung/output.csv')