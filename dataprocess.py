import os
import json
import time
import numpy as np
from PIL import Image
from shutil import copyfile

def read_cap(file):
    cap_dict={}
    with open(file) as f:
        for line in f:
            line_split=line.split('\t', 1)
            caption = line_split[1][:-1]
            imgid=line_split[0].split(sep='#')[0]
            if imgid not in cap_dict:
                cap_dict[imgid]=[caption]
            else:
                cap_dict[imgid].append(caption)
    return cap_dict

def getid(file):
    ids=[]
    with open(file) as f:
        for line in f:
            ids.append(line[:-1])
    return ids

def copyfiles(dir_out, dir_in, ids):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)
    for i in ids:
        pathin=os.path.join(dir_in, i)
        pathout=os.path.join(dir_out, i)
        copyfile(pathin, pathout)

def writecap(dir_out, ids, cap_dict):
    pathout=os.path.join(dir_out, 'captions.txt')
    output=[]
    for i in ids:
        dic={i: cap_dict[i]}
        output.append(json.dumps(dic))
    with open(pathout, mode='w') as file:
        file.write('\n'.join(output))

def re_allocate(dir_img, token, cap_path):
    dir_out={'train': 'train', 'val': 'val', 'test': 'test'}
    cap_dict= read_cap(token) #get caption dictionary
    img = os.listdir(dir_img)  # train, val, test mix; all img

    id_train = getid(cap_path['train'])  # ger ids
    id_val = getid(cap_path['val'])
    id_test = getid(cap_path['test'])

    copyfiles(dir_out['train'], dir_img, id_train)  # sort files to new dir
    copyfiles(dir_out['val'], dir_img, id_val)
    copyfiles(dir_out['test'], dir_img, id_test)

    writecap(dir_out['train'], id_train, cap_dict)
    writecap(dir_out['val'], id_val, cap_dict)
    writecap(dir_out['test'], id_test, cap_dict)

dir_img='images'
dir_text='text'
file_token='Flickr8k.token.txt'
file_train='Flickr_8k.trainImages.txt'
file_val='Flickr_8k.devImages.txt'
file_test='Flickr_8k.testImages.txt'
filepath_token=os.path.join(dir_text, file_token)

cap_path={'train': os.path.join(dir_text, file_train), 'val': os.path.join(dir_text, file_val),
         'test': os.path.join(dir_text, file_test)}

re_allocate(dir_img, filepath_token, cap_path)