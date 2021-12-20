# -*-coding: utf-8 -
'''
    @author: MD. Nazmuddoha Ansary
'''
#--------------------
# imports
#--------------------
import random
import os
import cv2
import numpy as np

from glob import glob
from tqdm import tqdm
from .config import config
from .word import create_word
from .utils import randColor

#--------------------
# background
#--------------------
def backgroundGenerator(ds,dim=(1024,1024)):
    '''
        generates random background
        args:
            ds   : dataset object
            dim  : the dimension for background
    '''
    # collect image paths
    _paths=[img_path for img_path in tqdm(glob(os.path.join(ds.common.background,"*.*")))]
    while True:
        _type=random.choice(["single","double","comb"])
        if _type=="single":
            img=cv2.imread(random.choice(_paths))
            img=cv2.resize(img,dim)
            yield img
        elif _type=="double":
            imgs=[]
            img_paths= random.sample(_paths, 2)
            for img_path in img_paths:
                img=cv2.imread(img_path)
                img=cv2.resize(img,dim)
                imgs.append(img)
            # randomly concat
            img=np.concatenate(imgs,axis=random.choice([0,1]))
            img=cv2.resize(img,dim)
            yield img
        else:
            imgs=[]
            img_paths= random.sample(_paths, 4)
            for img_path in img_paths:
                img=cv2.imread(img_path)
                img=cv2.resize(img,dim)
                imgs.append(img)
            seg1=imgs[:2]
            seg2=imgs[2:]
            seg1=np.concatenate(seg1,axis=0)
            seg2=np.concatenate(seg2,axis=0)
            img=np.concatenate([seg1,seg2],axis=1)
            img=cv2.resize(img,dim)
            yield img

#--------------------
# padding
#--------------------
def padPage(img):
    '''
        pads a page image to proper dimensions
    '''
    h,w=img.shape 
    if h>config.back_dim:
        # resize height
        height=config.back_dim
        width= int(height* w/h) 
        img=cv2.resize(img,(width,height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
        
        # pad width
        # mandatory check
        h,w=img.shape 
        # pad widths
        left_pad_width =random.randint(0,(config.back_dim-w))
        right_pad_width=config.back_dim-w-left_pad_width
        # pads
        left_pad =np.zeros((h,left_pad_width))
        right_pad=np.zeros((h,right_pad_width))
        # pad
        img =np.concatenate([left_pad,img,right_pad],axis=1)
        
    else:
        _type=random.choice(["top","bottom","middle"])
        if _type in ["top","bottom"]:
            pad_height=config.back_dim-h
            pad     =np.zeros((pad_height,config.back_dim))
            if _type=="top":
                img=np.concatenate([img,pad],axis=0)
            else:
                img=np.concatenate([pad,img],axis=0)
        else:
            # pad heights
            top_pad_height =(config.back_dim-h)//2
            bot_pad_height=config.back_dim-h-top_pad_height
            # pads
            top_pad =np.zeros((top_pad_height,w))
            bot_pad=np.zeros((bot_pad_height,w))
            # pad
            img =np.concatenate([top_pad,img,bot_pad],axis=0)
            
    # for error avoidance
    img=cv2.resize(img,(config.back_dim,config.back_dim),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    
    return img


def processLine(img):
    '''
        fixes a line image 
    '''
    h,w=img.shape 
    if w>config.back_dim:
        width=config.back_dim-random.randint(0,300)
        # resize
        height= int(width* h/w) 
        img=cv2.resize(img,(width,height),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
    # mandatory check
    h,w=img.shape 
    # pad widths
    left_pad_width =random.randint(0,(config.back_dim-w))
    right_pad_width=config.back_dim-w-left_pad_width
    # pads
    left_pad =np.zeros((h,left_pad_width),dtype=np.int64)
    right_pad=np.zeros((h,right_pad_width),dtype=np.int64)
    # pad
    img =np.concatenate([left_pad,img,right_pad],axis=1)
    
    return img 


#--------------------
# data
#--------------------
def createSceneData(ds,backgen):
    '''
        creates a scene image
        args:
            ds      :  the dataset object
            backgen :  background generator
        returns:
            back    :  the rendered image
            img     :  word level mapping 
    '''
    word_iden=2
    page_imgs=[]
    
    # select number of lines in an image
    num_lines=random.randint(config.min_num_lines,config.max_num_lines)
    for _ in range(num_lines):
        line_imgs=[]
        
        # select number of words
        num_words=random.randint(config.min_num_words,config.max_num_words)
        for _ in range(num_words):
            img,word_iden=create_word(  iden=word_iden,
                                        source_type=random.choice(config.data.sources),
                                        data_type=random.choice(config.data.formats),
                                        comp_type=random.choice(config.data.components), 
                                        ds=ds,
                                        use_dict=random.choice([True,False]))
            line_imgs.append(img)
            
        
        # reform
        rline_imgs=[]
        max_h=0
        # find max height
        for line_img in line_imgs:
            max_h=max(max_h,line_img.shape[0])
        
        # reform
        for line_img in line_imgs:
            h,w=line_img.shape 
            width= int(max_h* w/h) 
            line_img=cv2.resize(line_img,(width,max_h),fx=0,fy=0, interpolation = cv2.INTER_NEAREST)
            rline_imgs.append(line_img)
            

        # create the line image
        line_img=np.concatenate(rline_imgs,axis=1)
        

        line_img=processLine(line_img)
        # the page lines
        page_imgs.append(line_img)
        
    imgs=[]
    for img in page_imgs:
        # pad lines 
        pad_height=random.randint(config.vert_min_space,config.vert_max_space)
        pad     =np.zeros((pad_height,config.back_dim))
        img=np.concatenate([img,pad],axis=0)
        
        imgs.append(img)
        

    # page data img
    img=np.concatenate(imgs,axis=0)
    img=padPage(img)

    # scene
    back=next(backgen)
    vals=[v for v in np.unique(img) if v>0]

    for v in vals:
        col=randColor()
        back[img==v]=col
    
        
    return back,img
