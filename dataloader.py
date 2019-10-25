import os
import json
import nltk
import time
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms


class dataloader():
    def __init__(self, dir, vocab, transform):
        self.vocab=vocab
        self.transform = transform
        self.dir=dir
        self.loadcap(dir)
        self.loadimg(dir)
        #self.data_preshuf = self.get_data()

    def loadcap(self, dir):
        cap_file= os.path.join(dir, 'captions.txt')
        cap_dict={}
        with open(cap_file) as file:
            for line in file:
                cap=json.loads(line)
                for i ,j in cap.items():
                    cap_dict[i]=j
        self.cap_dict=cap_dict
        return cap_dict

    def loadimg(self, dir):
        img_dir=os.listdir(dir)
        img={}
        for file in img_dir:
            img_format=file.split('.')[1]
            if img_format == 'jpg':
                cur_img=Image.open(os.path.join(dir, file))
                img[file]= self.transform(cur_img)
        self.img=img
        return img

    def cap2id(self, cap):
        token=nltk.tokenize.word_tokenize(cap.lower())
        list=[]
        list.append(self.vocab.get_id('<start>'))
        list.extend([self.vocab.get_id(word) for word in token])
        list.append(self.vocab.get_id('<end>'))
        return list

    def get_img(self, imgid):
        return self.img[imgid]

    def get_data(self):
        img =[]
        cap = []
        for imgid, imgcap in self.cap_dict.items():
            num_cap = len(imgcap)
            img.extend([imgid] * num_cap) # generate image file name equalize number of captions
            for caption in imgcap:
                cap.append(self.cap2id(caption))
        data = img, cap
        return data

    '''def shuffle(self, seed):
        img, cap = self.data_preshuf
        img_shuffled=[]
        cap_shuffled=[]
        #num_img=len(img)
        torch.manual_seed(seed)
        perm = list(torch.randperm(len(img)))
        for i in range(len(img)):
            img_shuffled.append(img[perm[i]])
            cap_shuffled.append(cap[perm[i]])
        return img_shuffled, cap_shuffled'''

