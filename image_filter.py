import os
import sys
file_path='/home/shikigan/out_res_3'
output_path='/home/shikigan/out_res_5'

def filter(file_path,output_path):
    for root,dirs,paths in os.walk(file_path):
        for dir in dirs:
            if dir!='..'or dir!='.':
                tmp_path=os.path.join(file_path,dir)
                out_tmp_path=os.path.join(output_path,dir)
                try:
                    os.system('mkdir '+out_tmp_path)
                except FileExistsError:
                    pass
                for _root,_dirs,_paths in os.walk(tmp_path):
                    for _dir in _dirs:
                        _tmp_path=os.path.join(tmp_path,_dir)
                        _out_tmp_path=os.path.join(out_tmp_path,_dir)
                        try:
                            os.system('mkdir '+_out_tmp_path)
                        except FileExistsError:
                            pass
                        for __root,__dirs,__paths in os.walk(_tmp_path):
                            if len(__paths)<10:
                                os.system('rm -rf '+_out_tmp_path)
                                break
                            sorted(__paths)
                            iter=0
                            while iter<len(__paths):
                                os.system('cp '+os.path.join(_tmp_path,__paths[iter])+' '+os.path.join(_out_tmp_path,__paths[iter]))
                                iter+=3

if __name__=='__main__':
    filter(file_path,output_path=output_path)
