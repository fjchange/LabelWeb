#coding=utf-8
import os
import random
import csv
path_pre='/home/shikigan/out_res_1/'

def csv_solver(path):
    '''

    :param path:
    :return: path_dict: key是类序号，value是对应类下面的路径list
    '''
    with open(path)as c:
        path_dict={}
        lines=c.readlines()
        for line in lines:
            line_list=line.split(',')
            line_list[-1]=line_list[-1].split('\n')[0]
            if line_list[-1]=='' or int(line_list[-1])==-1:
                continue
            if int(line_list[-1])in path_dict.keys():
                path_dict[int(line_list[-1])].append(path_pre+line_list[0]+'/'+line_list[-1]+'/'+line_list[-2]+'.jpg')
            else:
                path_dict[int(line_list[-1])]=[path_pre+line_list[0]+'/'+line_list[-1]+'/'+line_list[-2]+'.jpg']
    return path_dict

#获取的是绝对路径
def csv_list_solver(path):
    '''
    利用ImageClass来存储csv文件对应的
    :param path:
    :return:
    '''
    with open(path)as c:
        path_dict={}
        path_list=[]
        lines=c.readlines()
        for line in lines:
            line_list=line.split(',')
            line_list[-1]=line_list[-1].split('\n')[0]
            if line_list[-1]=='' or int(line_list[-1])==-1:
                continue
            if int(line_list[-1])in path_dict.keys():
                path_dict[int(line_list[-1])].append(path_pre+line_list[0]+'/'+line_list[-1]+'/'+line_list[-2]+'.jpg')
            else:
                path_dict[int(line_list[-1])]=[path_pre+line_list[0]+'/'+line_list[-1]+'/'+line_list[-2]+'.jpg']
        for key in path_dict.keys():
             path_list.append(ImageClass(key,path_dict[key]))
    return path_list


#对于每一个视频文件夹中的图片进行配对（A,P,N)
'''
    如果在dict中共有m个label,总共有n个,选中的label中有n_i个，sum_i(n_i(n_i-1)*n)个匹配
'''
# solve the problem of 组合爆炸
#给定一个概率确定是否采用该图片，从而保证减少冗余的训练
def make_pairs(dict,prob=1.):
    pairs=[]
    for key in dict.keys():
        for item in dict[key]:
            #如果less than 阈值那么就采用this item
            if random.random()<prob:
                for _item in dict[key]:
                    #if less than
                    if _item!=item and random.random()<prob:
                        for _key in dict.keys():
                            if _key!=key:
                                for __item in dict[_key]:
                                    if  random.random()<prob:
                                        pairs.append((item,_item,__item))
    return pairs


#返回的dataset是[{},{}...]
def get_dataset(csv_exp):
    dataset=[]
    for root, dirs, paths in os.walk(csv_exp):
        for path in paths:
            path = os.path.join(csv_exp, path)
            data=csv_solver(path)
            dataset.append(data)
    return dataset

#返回的dataset是[[],[],[]...]
def _get_dataset(csv_exp):
    dataset=[]
    for root, dirs, paths in os.walk(csv_exp):
        for path in paths:
            path = os.path.join(csv_exp, path)
            data=csv_list_solver(path)
            print(len(data))
            dataset.append(data)
        print(len(dataset))
    return dataset

def __get_dataset(file_exp):
    dataset=[]
    for root,dirs,paths in os.walk(file_exp):
        for dir in dirs:
            path_lists=[]
            temp_path=os.path.join(file_exp,dir)
            for _root,_dirs,_paths in os.walk(temp_path):
                for _dir in _dirs:
                    _temp_path = os.path.join(temp_path,_dir )
                    for __root,__dirs,__paths in os.walk(_temp_path):
                        path_list=[]
                        for __path in __paths:
                            path_list.append(os.path.join(_temp_path,__path))
                        path_lists.append(ImageClass(_dir,path_list))
            dataset.append(path_lists)
    return dataset

#不同视频集的图片的个体不重叠
def ___get_dataset(file_exp):
    path_lists = []
    count=0
    for root,dirs,paths in os.walk(file_exp):
        for dir in dirs:
            temp_path=os.path.join(file_exp,dir)
            for _root,_dirs,_paths in os.walk(temp_path):
                for _dir in _dirs:
                    _temp_path = os.path.join(temp_path,_dir )
                    for __root,__dirs,__paths in os.walk(_temp_path):
                        path_list=[]
                        for __path in __paths:
                            path_list.append(os.path.join(_temp_path,__path))
                        path_lists.append(ImageClass(count,path_list))
                        count+=1
    return path_lists

def get_dataset_predit(file_exp):
    path_lists = []
    for root, dirs, paths in os.walk(file_exp):
        for dir in dirs:
            temp_path = os.path.join(file_exp, dir)
            for _root, _dirs, _paths in os.walk(temp_path):
                for _dir in _dirs:
                    _temp_path = os.path.join(temp_path, _dir)
                    for __root, __dirs, __paths in os.walk(_temp_path):
                        path_list = []
                        for __path in __paths:
                            path_list.append(os.path.join(_temp_path, __path))
                        path_lists.append( path_list)
    return path_lists

def write_csv(file_exp,out_csv_path):
    for root,dirs,paths in os.walk(file_exp):
        for dir in dirs:
            with open(os.path.join(out_csv_path,dir+'.csv'),'w')as c:
                writer=csv.writer(c)
                temp_path = os.path.join(file_exp, dir)
                for _root,_dirs,_paths in os.walk(temp_path):
                    for _dir in _dirs:
                        for __root,__dirs,__paths in os.walk(os.path.join(temp_path,_dir)):
                            for path in __paths:
                                writer.writerow([dir,path.split('.')[0],_dir])

class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

if __name__=='__main__':
    #_get_dataset('/home/shikigan/kiwi_fung/labeled_data')
    write_csv('/home/shikigan/out_res_1/','/home/shikigan/kiwi_fung/label_data_1/')